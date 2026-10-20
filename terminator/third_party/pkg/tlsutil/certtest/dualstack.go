// 双栈参数化测试助手：让同一套用例在**国密 TLCP** 与**标准 TLS** 两条路径上
// 各跑一遍，照 audit-log 的 forEachSQLBackend 范式。
//
// ==============================================================================
// 【为什么需要它】
// 「用国密就全部国密，否则全部国际算法」这条决策要成立，前提是两条路径都被
// 真正执行过。此前的实际情况是：国密路径有零星单向用例、国际路径靠既有用例
// 顺带覆盖，没有任何地方保证**同一个行为**在两种体系下结论一致。
// 于是「国密下客户端证书准入完全不执行」这类缺陷能长期存在 ——
// 跑的是国际路径的用例，国密路径压根没进去。
//
// 参数化之后，漏掉一条路径会表现为子测试缺失，而不是静默通过。
// 子测试名即体系名（tlcp / standard），失败时能一眼看出是哪种密码体系出的问题。
package certtest

import (
	"crypto/tls"
	"crypto/x509"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
)

// TLSMode 描述一种传输密码体系下的全套测试材料。
//
// 服务端与客户端材料都给齐，因为双向认证的用例两边都要用；
// DigestAlgo 一并给出，使「摘要算法随传输体系联动」可以被直接断言，
// 而不是让每个用例自己去推当前该用什么算法。
type TLSMode struct {
	// Name 是子测试名，也是失败时的唯一定位线索。
	Name string

	// NationalCipher 直接对应各服务的 *_TLS_NATIONAL_CIPHER。
	NationalCipher bool

	// Dir 是本模式证书所在目录（TLCP 为 <root>/tlcp，标准 TLS 为 <root>）。
	Dir string

	// CAFile 校验对端用的根 CA。
	CAFile string

	// ── 服务端材料 ──
	// TLCP 用 Sign*/Enc* 双对；标准 TLS 用 CertFile/KeyFile。
	// 另一套在当前模式下为空串 —— 取到空串就说明用错了模式。
	SignCertFile string
	SignKeyFile  string
	EncCertFile  string
	EncKeyFile   string
	CertFile     string
	KeyFile      string

	// ── 客户端材料 ──
	ClientSignCertFile string
	ClientSignKeyFile  string
	ClientEncCertFile  string
	ClientEncKeyFile   string
	ClientCertFile     string
	ClientKeyFile      string

	// DigestAlgo 是本体系应当使用的凭据摘要算法（"sm3" / "sha256"）。
	// 「用国密则全用国密」的联动断言直接用它，不要在用例里硬编码。
	DigestAlgo string
}

// ForEachTLSMode 在国密 TLCP 与标准 TLS 两种体系上各跑一遍 fn。
//
// 证书由 Generate 就地生成在 t.TempDir()，不依赖工作区状态 ——
// 任何机器、任何克隆都能跑，也不会与仓内固定开发证书相互污染。
//
// 用法：
//
//	certtest.ForEachTLSMode(t, func(t *testing.T, mode certtest.TLSMode) {
//	    srv := startServer(t, mode)           // 按 mode 装配监听
//	    client := mode.NewHTTPClient(t)       // 按 mode 取能握手的客户端
//	    // ... 同一套断言
//	})
func ForEachTLSMode(t *testing.T, fn func(t *testing.T, mode TLSMode)) {
	t.Helper()
	for _, mode := range TLSModes(t) {
		mode := mode
		t.Run(mode.Name, func(t *testing.T) {
			fn(t, mode)
		})
	}
}

// TLSModes 生成两种体系的材料并返回，供不便用闭包的场景（如需要在
// 子测试外先装配共享 fixture）自行遍历。
//
// 两种体系各用**独立目录**：共用一套目录会让「把 TLCP 证书喂给标准 TLS」
// 这种配错体系的情况偶然「跑通」，从而掩盖真实缺陷。
func TLSModes(t *testing.T) []TLSMode {
	t.Helper()

	tlcpRoot := t.TempDir()
	if err := Generate(tlcpRoot); err != nil {
		t.Fatalf("生成国密测试证书失败: %v", err)
	}
	stdRoot := t.TempDir()
	if err := Generate(stdRoot); err != nil {
		t.Fatalf("生成标准 TLS 测试证书失败: %v", err)
	}

	tlcpDir := filepath.Join(tlcpRoot, "tlcp")
	return []TLSMode{
		{
			Name:           "tlcp",
			NationalCipher: true,
			Dir:            tlcpDir,
			CAFile:         filepath.Join(tlcpDir, "ca.crt"),

			SignCertFile: filepath.Join(tlcpDir, "server-sign.crt"),
			SignKeyFile:  filepath.Join(tlcpDir, "server-sign.key"),
			EncCertFile:  filepath.Join(tlcpDir, "server-enc.crt"),
			EncKeyFile:   filepath.Join(tlcpDir, "server-enc.key"),

			ClientSignCertFile: filepath.Join(tlcpDir, "client-sign.crt"),
			ClientSignKeyFile:  filepath.Join(tlcpDir, "client-sign.key"),
			ClientEncCertFile:  filepath.Join(tlcpDir, "client-enc.crt"),
			ClientEncKeyFile:   filepath.Join(tlcpDir, "client-enc.key"),

			DigestAlgo: "sm3",
		},
		{
			Name:           "standard",
			NationalCipher: false,
			Dir:            stdRoot,
			CAFile:         filepath.Join(stdRoot, "ca.crt"),

			CertFile: filepath.Join(stdRoot, "server.crt"),
			KeyFile:  filepath.Join(stdRoot, "server.key"),

			ClientCertFile: filepath.Join(stdRoot, "client.crt"),
			ClientKeyFile:  filepath.Join(stdRoot, "client.key"),

			DigestAlgo: "sha256",
		},
	}
}

// ServerTLCPConfig 按本模式给出服务端 TLCP 配置；非国密模式返回 nil。
func (m TLSMode) ServerTLCPConfig(clientAuth string) *tlsutil.TLCPConfig {
	if !m.NationalCipher {
		return nil
	}
	return &tlsutil.TLCPConfig{
		Enabled:      true,
		SignCertFile: m.SignCertFile,
		SignKeyFile:  m.SignKeyFile,
		EncCertFile:  m.EncCertFile,
		EncKeyFile:   m.EncKeyFile,
		ClientCAFile: m.CAFile,
		ClientAuth:   clientAuth,
	}
}

// ClientTLCPConfig 按本模式给出客户端 TLCP 配置；非国密模式返回 nil。
func (m TLSMode) ClientTLCPConfig() *tlsutil.TLCPClientConfig {
	if !m.NationalCipher {
		return nil
	}
	return &tlsutil.TLCPClientConfig{
		CAFile:       m.CAFile,
		SignCertFile: m.ClientSignCertFile,
		SignKeyFile:  m.ClientSignKeyFile,
		EncCertFile:  m.ClientEncCertFile,
		EncKeyFile:   m.ClientEncKeyFile,
	}
}

// NewHTTPClient 返回能与本模式服务端完成握手的 HTTP 客户端（带客户端证书）。
//
// 国密模式走 tlsutil.TLCPClientDialer —— 不能用标准 http.Transport 的
// TLSClientConfig，因为 gotlcp 不是 crypto/tls。这一点是 TLCP 改造里
// 最容易写错的地方：配了 TLSClientConfig 看似「配了证书」，实际不生效。
func (m TLSMode) NewHTTPClient(t *testing.T) *http.Client {
	t.Helper()
	transport := &http.Transport{}

	if m.NationalCipher {
		clientCfg, err := tlsutil.BuildTLCPClientConfig(m.ClientTLCPConfig())
		if err != nil {
			t.Fatalf("构造 TLCP 客户端配置失败: %v", err)
		}
		transport.DialTLSContext = tlsutil.TLCPClientDialer(clientCfg)
	} else {
		cert, err := tls.LoadX509KeyPair(m.ClientCertFile, m.ClientKeyFile)
		if err != nil {
			t.Fatalf("加载标准 TLS 客户端证书失败: %v", err)
		}
		caPEM, err := os.ReadFile(m.CAFile)
		if err != nil {
			t.Fatalf("读取 CA 失败: %v", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caPEM) {
			t.Fatalf("解析 CA 失败: %s", m.CAFile)
		}
		transport.TLSClientConfig = &tls.Config{
			MinVersion:   tls.VersionTLS12,
			Certificates: []tls.Certificate{cert},
			RootCAs:      pool,
		}
	}
	return &http.Client{Transport: transport, Timeout: 5 * time.Second}
}
