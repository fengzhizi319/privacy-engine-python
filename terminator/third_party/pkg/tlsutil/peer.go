// Package tlsutil — 传输无关的对端证书视图。
//
// ==============================================================================
// 【为什么需要这一层】
//
// 鉴权层要在两种传输上做同一件事（客户端证书准入：cn / fp / spki），
// 而两种传输把对端证书暴露成**互不兼容的 Go 类型**：
//
//	标准 TLS   crypto/tls        → []*x509.Certificate
//	国密 TLCP  gitee.com/…/tlcp  → []*smx509.Certificate
//
// 二者无法互转：**Go 标准库根本解析不了 SM2 证书** ——
// `x509.ParseCertificate` 对 SM2 曲线直接返回 "unsupported elliptic curve"，
// 而 `smx509.Certificate` 是独立结构体（镜像字段，不内嵌 x509，也没有 ToX509）。
//
// 于是出现三个 producer × 两种证书类型的组合爆炸：
// REST 标准 TLS（http.Request.TLS）、REST TLCP（ConnContext 桥接）、gRPC（AuthInfo）。
// 本文件把它们收敛到**一个值类型**上，下游 verifier 各只需一条实现。
//
// ==============================================================================
// 【为什么是 struct 而不是 interface】
//
// 穷举鉴权层对证书的全部使用，只有六个**字段**，零个方法 ——
// 链校验在握手期由 TLS/TLCP 栈做完了，到鉴权层已是验过链的证书。
// interface 抽象的是行为，这里没有行为可抽象；用 interface 表达"取字段"
// 需要两个包装类型 + 十个转发方法，并引入 **typed-nil 陷阱**：
//
//	var p PeerCert = (*x509Peer)(nil)   // p != nil 为真
//
// 本仓库已因此吃过一次亏（`internal/anchor.NewFileBackend` 刻意返回接口
// 而非具体类型就是为了绕开它）。在鉴权路径上踩到它等于认证绕过。
// struct 可比较、可直接进日志、测试写字面量即可、没有 nil 歧义。
//
// ==============================================================================
// 【为什么不全局改用 smx509】
//
// smx509 确实是超集（SM2 与 RSA 都能解析，已实测）。但标准 TLS 路径
// **根本不经过 DER 解析** —— crypto/tls 握手完成后直接给出已解析的
// *x509.Certificate。为统一类型而把 cert.Raw 用 smx509 重解析一遍，
// 是每请求多一次 ASN.1 解析，还引入「两个解析器对同一份 DER 理解不一致」
// 这一新的静默安全差异面。
package tlsutil

import (
	"crypto/tls"
	"crypto/x509"

	"gitee.com/Trisia/gotlcp/tlcp"
	"github.com/emmansun/gmsm/smx509"
)

// PeerIdentity 是传输无关的对端证书视图：只装鉴权需要的**原始材料**。
//
// 刻意不含摘要，也不含任何策略判断：
//   - 摘要算法由凭据声明决定（见 pkg/auth 的自描述摘要），在此预算会算错或算多；
//   - 绝大多数请求并无 fp= / spki= 绑定，预算四个摘要是纯浪费。
//
// 两个 DER 字段是**唯一口径**，不要改用解析后的对象重新序列化：
// SM2 公钥过不了 `x509.MarshalPKIXPublicKey`，而原始 DER 对
// RSA/ECDSA 与 marshal 结果逐字节相等（已实测），是唯一同时覆盖国密与国际算法的口径。
type PeerIdentity struct {
	// CertDER 是叶证书完整 DER（x509 的 Raw），叶证书指纹的唯一口径。
	CertDER []byte
	// SPKIDER 是 SubjectPublicKeyInfo 的原始 DER（x509 的 RawSubjectPublicKeyInfo），
	// 公钥摘要的唯一口径。
	SPKIDER []byte

	CommonName string
	DNSNames   []string
	// URIs 存字符串而非 *url.URL：消费方只做大小写无关的字符串比对，
	// 带着 *url.URL 只是多一层解引用和一个 nil 风险。
	URIs []string

	// Verified 表示对端证书是否**已被校验到受信 CA**（握手期产生了验证链）。
	//
	// 【为什么必须单独表达，而不是"有证书就算认证过"】
	// Go 的 ClientAuth 有两档容易混淆的取值：
	//   - RequireAndVerifyClientCert：链校验失败即握手失败，VerifiedChains 非空；
	//   - RequireAnyClientCert（本仓库配置里的 "require"）：**只要求出示证书，
	//     不验链** —— 此时 PeerCertificates 有值而 VerifiedChains 为空。
	// 授权类检查（CN 白名单、指纹钉扎）若只看"有没有证书"，
	// 在后一档配置下会接受任意自签证书。故保留这一位，由调用方显式要求。
	Verified bool
}

// Empty 报告该视图是否**完全没有**可用的对端身份材料。
//
// 调用方据此走 fail-closed 分支：视图为空意味着这条链路上拿不到客户端证书
// （例如 TLCP 监听未做 ConnContext 桥接），此时任何 fp/spki/cn 绑定
// 都应判定为不满足，而不是跳过检查。
//
// 【为什么判据是"全都没有"而不是"没有 DER"】
// 两类消费者需要的材料不同：
//   - 摘要绑定（fp= / spki=）需要 DER；
//   - CN / SAN 白名单只需要 CommonName 与 SAN，**不需要 DER**。
//
// 若把"没有 DER"当成"没有身份"，只做 CN 校验的场景就会被误判为无证书。
// 摘要侧不依赖本函数兜底：digestOf 对空 DER 返回空串，
// 比对必然失配 —— 即每个检查各自对自己需要的材料负责。
func (p *PeerIdentity) Empty() bool {
	if p == nil {
		return true
	}
	return len(p.CertDER) == 0 && len(p.SPKIDER) == 0 &&
		p.CommonName == "" && len(p.DNSNames) == 0 && len(p.URIs) == 0
}

// PeerFromX509 由标准 TLS 链路的叶证书构造视图。
func PeerFromX509(c *x509.Certificate) *PeerIdentity {
	if c == nil {
		return nil
	}
	uris := make([]string, 0, len(c.URIs))
	for _, u := range c.URIs {
		if u != nil {
			uris = append(uris, u.String())
		}
	}
	return &PeerIdentity{
		CertDER:    c.Raw,
		SPKIDER:    c.RawSubjectPublicKeyInfo,
		CommonName: c.Subject.CommonName,
		DNSNames:   c.DNSNames,
		URIs:       uris,
	}
}

// PeerFromSMX509 由国密 TLCP 链路的叶证书构造视图。
//
// smx509 能同时解析 SM2 与 RSA/ECDSA，故本函数对两类证书都成立；
// 但标准 TLS 链路应走 PeerFromX509，避免无谓的重解析（见包头说明）。
func PeerFromSMX509(c *smx509.Certificate) *PeerIdentity {
	if c == nil {
		return nil
	}
	uris := make([]string, 0, len(c.URIs))
	for _, u := range c.URIs {
		if u != nil {
			uris = append(uris, u.String())
		}
	}
	return &PeerIdentity{
		CertDER:    c.Raw,
		SPKIDER:    c.RawSubjectPublicKeyInfo,
		CommonName: c.Subject.CommonName,
		DNSNames:   c.DNSNames,
		URIs:       uris,
	}
}

// PeerFromTLSState 由标准 TLS 连接状态取叶证书构造视图。
//
// 取 PeerCertificates[0] 而非 VerifiedChains[0][0]：前者是对端**实际出示**的
// 叶证书，后者是校验链的起点。二者通常相同，但在 ClientAuth 为
// RequestClientCert（不校验）时 VerifiedChains 为空 —— 那种配置下
// 仍应能取到证书做应用层准入，否则会静默退化为「没有客户端证书」。
func PeerFromTLSState(st *tls.ConnectionState) *PeerIdentity {
	if st == nil {
		return nil
	}
	// 优先取验证链的叶证书：正常 mTLS 下它与 PeerCertificates[0] 是同一张，
	// 但只有前者能证明"校验到了受信 CA"。
	if len(st.VerifiedChains) > 0 && len(st.VerifiedChains[0]) > 0 {
		p := PeerFromX509(st.VerifiedChains[0][0])
		if p != nil {
			p.Verified = true
		}
		return p
	}
	if len(st.PeerCertificates) == 0 {
		return nil
	}
	return PeerFromX509(st.PeerCertificates[0]) // Verified 保持 false
}

// PeerFromTLCPConn 由国密 TLCP 连接取叶证书构造视图。
//
// 【调用时机很关键】
// 必须在**握手完成后**调用。http.Server.ConnContext 在握手之前触发，
// 那时 ConnectionState().PeerCertificates 恒为空 —— 所以 ConnContext 里
// 应注入一个惰性 provider（在首个请求到达时才调本函数），
// 而不是当场提取。见 TLCPConnContext。
func PeerFromTLCPConn(c *tlcp.Conn) *PeerIdentity {
	if c == nil {
		return nil
	}
	st := c.ConnectionState()
	if len(st.VerifiedChains) > 0 && len(st.VerifiedChains[0]) > 0 {
		p := PeerFromSMX509(st.VerifiedChains[0][0])
		if p != nil {
			p.Verified = true
		}
		return p
	}
	if len(st.PeerCertificates) == 0 {
		return nil
	}
	return PeerFromSMX509(st.PeerCertificates[0]) // Verified 保持 false
}
