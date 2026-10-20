// Package config provides shared environment helpers and the fail-closed security
// invariants enforced by every PrivShield service at startup.
// Package config 提供共享的环境变量读取助手，以及各服务启动时必须通过的 fail-closed 安全不变式校验。
package config

import (
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strings"
)

// Fail-closed 启动期安全不变式错误。任一命中即由服务的 Validate() 上抛、main() 终止进程，
// 避免「配置缺失 → 静默降级为无鉴权/明文传输」的可观测性盲区。
var (
	// ErrAPIKeyRequired 表示监听地址可被远端访问但未配置入站 API Key（原实现为空即放行）。
	ErrAPIKeyRequired = errors.New("inbound API key must not be empty when listening on a non-loopback address")

	// ErrAuthRequired 表示监听地址可被远端访问但未启用身份认证（AUTH_ENABLED=false）。
	ErrAuthRequired = errors.New("authentication must be enabled when listening on a non-loopback address (set *_AUTH_ENABLED=true)")

	// ErrTLSRequired 表示部署方声明必须启用 TLS，但 TLS 开关处于关闭状态。
	ErrTLSRequired = errors.New("TLS is required by configuration but not enabled")

	// ErrTLSRequiredForRemote 表示监听地址可被远端访问但未启用 TLS。
	ErrTLSRequiredForRemote = errors.New("TLS must be enabled when listening on a non-loopback address (set *_TLS_ENABLED=true)")

	// ErrMTLSWhitelistRequired 表示已启用 gRPC TLS 但未注入客户端证书 CN 白名单文件；
	// 此时白名单拦截器不会被注册，任何通过 CA 校验的客户端都可调用全部方法。
	ErrMTLSWhitelistRequired = errors.New("mTLS CN whitelist file is required when TLS is enabled on the gRPC server")
)

// SecurityRequirements 描述一次 fail-closed 校验所需的监听与开关状态。
type SecurityRequirements struct {
	// ServiceName 用于错误信息定位，例如 "audit-log"。
	ServiceName string
	// Hosts 是该进程将要监听的全部地址（HTTP 与 gRPC），用于判断是否存在非环回暴露面。
	Hosts []string
	// APIKey 是入站鉴权密钥（各服务前缀变量）。
	APIKey string
	// HasAuthCredential 表示除单一 APIKey 外，已配置 Scoped Keys、KeysFile 或动态用户凭据。
	// 当 HasAuthCredential 为真时，允许遗留单一 APIKey 为空（生产模式推荐 Scoped Keys / 动态用户体系）。
	HasAuthCredential bool
	// AuthEnabled 表示身份认证是否已启用。
	AuthEnabled bool
	// AdmissionOnly 声明本服务是凭据的**转交者**而非**校验者**（隐私安全网关形态：
	// scope 裁决在被代理的 Agent 端，网关从不校验凭据内容）。置真时「远程暴露必须
	// 有凭据可校验」这条红线改由准入姿态履行：AuthEnabled（逐请求必带凭据 + 透传）
	// 或 BackendCredentialConfigured（剔除入站凭据 + 回源统一注入）任一成立即可。
	// 缺省为假 —— 其余服务沿用原有判据，本字段不构成任何放宽。
	AdmissionOnly bool
	// BackendCredentialConfigured 声明已配置回源注入凭据（网关的
	// PRIVACY_GATEWAY_BACKEND_API_KEY）。它不证明本服务能校验入站凭据，只证明
	// 出站调用不会匿名 —— 因此只参与「不得匿名直传后端」这一条判据。
	BackendCredentialConfigured bool
	// TLSEnabled 表示服务端 TLS 是否已启用。
	TLSEnabled bool
	// RequireTLS 由部署方显式置真（生产编排），此时 TLSEnabled 必须为真。
	RequireTLS bool
	// SkipTLSForRemote 为真时跳过非环回 TLS 强制校验（用于不终止 TLS 的反向代理网关）。
	SkipTLSForRemote bool
	// GRPCEnabled 表示本进程是否监听 gRPC 端口。
	GRPCEnabled bool
	// MTLSWhitelistFile 是客户端证书 CN 白名单文件路径（空即拦截器不注册）。
	MTLSWhitelistFile string
	// HasPoPCredentials 声明已配置方案 C 客户端证书强绑定的凭据（PoP Binding）。
	// 方案 C 将客户端证书 CN、指纹与 SPKI 公钥 Hash 一体化内聚在凭据配置中，
	// 由应用层 DecidePoP 执行强类型物理绑定核验。当 HasPoPCredentials 为真时，
	// 允许 MTLSWhitelistFile 为空，实现准入与授权一体化。
	HasPoPCredentials bool
	// HasExternalAnchor 表示写入链式存证的服务是否配置了外部锚定目的地。
	//
	// 链密钥（AUDIT_LOG_HASH_KEY）已整体退场，存证哈希恒为公开的纯 SM3：
	// 任何拿到库写权限的人都能改完记录再自行重算全链，验真照常通过。
	// 防伪造能力因此整体移交外部锚定 —— 在攻击者控制范围之外留一份
	// 「某时刻链长什么样」的记录（docs/no_hmac_design.md §4.1）。
	//
	// **本字段只用于告警，不作为启动硬约束。** 这是明确的项目决策：
	// 开发阶段局方端点尚未就绪，锚定处于只写不可读状态，
	// 此时强制拒绝启动会让整条流程无法推进。
	// 代价是「无锚定 + 纯 SM3」这个组合可以启动，而它等于零防护
	// （§3.2）—— 因此必须在启动横幅与验真结论里如实呈现，
	// 绝不能让它看起来像有保护。
	HasExternalAnchor bool
	// WritesEvidenceChain 标记该服务写入链式存证（audit-log），
	// 用于决定是否需要就锚定缺失发出告警。
	WritesEvidenceChain bool
	// AllowedCIDRs 是 IP 白名单。空表示透传。
	// 非环回地址且为空时 ValidateFailClosed 输出 slog.Warn 警告。
	AllowedCIDRs []string
}

// ValidateFailClosed 强制「安全开关缺失即启动失败」，取代原先的空值静默放行。
//
// 判定规则：
//  1. 任一监听地址非环回（0.0.0.0 / 具体网卡 IP / 通配）→ 必须配置入站 API Key、Scoped Keys 或动态用户凭证；
//     纯 127.0.0.1 本地开发形态允许无密钥启动；
//  2. RequireTLS 为真而 TLSEnabled 为假 → 拒绝启动（防止生产编排漏配证书却照常服务）；
//  3. 启用 gRPC TLS 但未提供 CN 白名单文件 → 拒绝启动（防止「以为已做双向认证、实则未注册拦截器」）；
//  4. WritesEvidenceChain 为真且无外部锚定、且存在非环回监听 → **告警但放行**
//     （链密钥已退场，纯 SM3 可被重算伪造，此时无锚定即零防护；
//     按项目决策不做硬约束，但必须让运维看见）。
//
// 注意：本函数不再校验信封加密密钥。存证样本已改为明文落盘（见 audit-log
// design.md §6），加密不再是启动前提；防伪造改由第 4 条的外部锚定承担。
func ValidateFailClosed(req SecurityRequirements) error {
	name := req.ServiceName
	if name == "" {
		name = "service"
	}
	remoteExposed := false
	for _, h := range req.Hosts {
		if !IsLoopbackHost(h) {
			remoteExposed = true
			break
		}
	}

	// 空白密钥等同于没配：只判 != "" 时 APIKey=" " 会穿过这道门，
	// 服务照常起来，然后所有请求都拿着一个没人能提供的凭据被 401 ——
	// 门禁要拦的是「以为配了其实没配」。同函数对 MTLSWhitelistFile 就是先 Trim 的。
	hasAuth := strings.TrimSpace(req.APIKey) != "" || req.HasAuthCredential
	// 转交者形态（隐私安全网关）自身不校验凭据，「持有可校验入站凭据」这条对它
	// 没有意义：它的准入由逐请求「必带凭据」（AuthEnabled）或回源统一注入
	// （BackendCredentialConfigured）履行，两条都让后端看到一次认证过的调用。
	// 此前网关被迫同时配一把它永远不会用的 Agent 前缀 Key 才能非环回启动 ——
	// 那正是「配了但不生效」的形态，也堵死了本次裁定的 D1 姿态（演进纪实 §12）。
	if req.AdmissionOnly && (req.AuthEnabled || req.BackendCredentialConfigured) {
		hasAuth = true
	}
	if !hasAuth {
		if remoteExposed {
			return fmt.Errorf("%s: %w (set the service *_API_KEY or *_API_KEYS variable; bind to 127.0.0.1 for local development)", name, ErrAPIKeyRequired)
		}
	}

	// 回源注入是「关闭逐请求准入」时的既有合法姿态：入站凭据被剔除、后端凭网关
	// Key 认证，暴露面依旧不是匿名。除此之外，声明了 AuthEnabled=false 却要远程
	// 监听 = 无人把关的入口，拒启。
	if !req.AuthEnabled && !(req.AdmissionOnly && req.BackendCredentialConfigured) && remoteExposed {
		return fmt.Errorf("%s: %w", name, ErrAuthRequired)
	}

	if !req.SkipTLSForRemote && !req.TLSEnabled && remoteExposed {
		return fmt.Errorf("%s: %w", name, ErrTLSRequiredForRemote)
	}

	if req.RequireTLS && !req.TLSEnabled {
		return fmt.Errorf("%s: %w", name, ErrTLSRequired)
	}

	// 当启用 gRPC TLS 时，必须具备客户端证书准入鉴权：
	// 1. 显式提供 mTLS CN 白名单文件（MTLSWhitelistFile）；或
	// 2. 启用了具备证书公钥/指纹绑定的方案 C 凭据（HasPoPCredentials 为真）。
	// 方案 C（RFC 8705 / PoP 绑定）将客户端证书 CN、指纹与 SPKI 公钥 Hash 深度内聚在凭据库中，
	// 由应用层 DecidePoP 执行强类型物理绑定核验，满足连接准入与最小权限授权一体化。
	// 两者均缺失时，阻断启动以防未授权客户端调用 gRPC。
	//
	// AdmissionOnly 例外：网关的 gRPC 面是透明流代理，根本没有 CN 白名单/PoP 校验器
	// （它只把凭据转交给 Agent），这条断言在它身上没有执行点 —— 逼它配一个白名单文件，
	// 得到的是一张永远不会被读的表。网关在该面的准入是 CIDR + 逐请求必带凭据。
	// 此前这条恰好被仓内 config/mtls-whitelist.yaml 的存在蒙过：容器里没有该文件时，
	// 声明了 PRIVACY_GATEWAY_AUTH_ENABLED=true 的网关会在 ErrMTLSWhitelistRequired 上拒启。
	if req.TLSEnabled && req.GRPCEnabled && !req.AdmissionOnly &&
		strings.TrimSpace(req.MTLSWhitelistFile) == "" && !req.HasPoPCredentials {
		return fmt.Errorf("%s: %w", name, ErrMTLSWhitelistRequired)
	}

	// 【这里刻意不是 fail-closed，与本文件其余判据不同】
	// 存证哈希恒为公开的纯 SM3，链自身不再防伪造；外部锚定是唯一信任根。
	// 无锚定 + 纯 SM3 = 零防护：拿到库写权限的人可改写全部历史，
	// 而 VerifyChain 依然返回 ok。
	//
	// 按项目决策（开发阶段局方端点未就绪）此处只告警不拒启。
	// 这意味着这条告警是运维唯一的提示 —— 措辞必须直白到不会被当成噪音。
	if req.WritesEvidenceChain && !req.HasExternalAnchor && remoteExposed {
		slog.Warn("NO TAMPER PROTECTION: evidence hashes are un-keyed SM3 (publicly re-computable) "+
			"and no external anchor is configured; anyone with database write access can rewrite "+
			"the entire history and chain verification will still report ok. "+
			"Set AUDIT_LOG_ANCHOR_BACKEND to an external destination.",
			"service", name)
	}

	if remoteExposed && len(req.AllowedCIDRs) == 0 {
		slog.Warn("IP allowlist is empty; all client IPs are accepted. Set IP allowlist to restrict access",
			"service", name)
	}

	return nil
}

// IsLoopbackHost reports whether a listen host only accepts local connections.
//
// IsLoopbackHost 判断监听地址是否仅接受本机连接："localhost"、127.0.0.0/8 与 ::1
// 视为本地；"0.0.0.0"、"::"、空主机（":8080" 或 ""）与具体网卡地址视为对外暴露。
//
// 空主机必须是「对外暴露」：`net.Listen("tcp", ":8080")` 绑定的是全部网卡，
// 与 "0.0.0.0:8080" 是同一个监听面。此前空串走前置短路判成环回、而带端口的
// ":8080" 走 SplitHostPort 判成非环回 —— 同一语义两种安全结论，且前者更宽松：
// `PRIVACY_GATEWAY_REST_HOST=`（显式置空）会让网关以全网卡监听却按开发态
// 匿名开放 /metrics 与 /gateway/backends。判据只能有一个方向：宁可让运维
// 收到「请绑 127.0.0.1 或配 Key」的启动错误，也不能让暴露面自认为安全。
func IsLoopbackHost(host string) bool {
	h := strings.TrimSpace(host)
	if h == "" {
		return false
	}
	if h == "localhost" {
		return true
	}
	// 允许 "host:port" 形式的入参，取主机部分判断。
	if splitHost, _, err := net.SplitHostPort(h); err == nil {
		h = splitHost
	}
	if h == "" || h == "*" {
		return false
	}
	if ip := net.ParseIP(h); ip != nil {
		return ip.IsLoopback()
	}
	// 无法解析为主机名时按对外暴露处理（fail-closed）。
	return false
}

// productionEnvValues 是被认定为「生产」的环境标签取值（小写比对）。
var productionEnvValues = map[string]struct{}{
	"prod":       {},
	"production": {},
}

// IsProductionLabeled 报告运维是否**显式**把本进程标注为生产环境。
//
// envVars 为待检查的环境变量名（按入参顺序读取，任一命中即为真），典型用法：
//
//	pkgconfig.IsProductionLabeled("APP_ENV", "SERVICE_HUB_ENV")
//	pkgconfig.IsProductionLabeled("APP_ENV", "PRIVACY_ENV")
//
// 为何需要一个共享实现（G-16）：本判定原先在 service-hub 与 privacy-engine 中被
// 手写了 6 份，每份都是 4 行 strings.EqualFold 或 strings.ToLower 的组合。副本越多，
// 漏一份就意味着一道生产门禁恒不触发（实际已发生：APP_ENV 在全编排中零注入，
// 而 6 份副本全部只读它，因此所有生产 fail-closed 门禁静默失效）。
//
// 语义边界：本函数只看**显式标签**，不包含 GIN_MODE、监听地址是否环回、TLS 是否
// 启用等推断条件 —— 那些属于「生产敏感姿态」的宽定义（如 service-hub 的
// Config.IsProduction），两者不可互换：把宽定义用在启动期 fail-closed 门禁上，会让
// 绑定 0.0.0.0 的开发用 compose 也被当成生产而拒绝启动。
func IsProductionLabeled(envVars ...string) bool {
	for _, name := range envVars {
		if name == "" {
			continue
		}
		v := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
		if _, ok := productionEnvValues[v]; ok {
			return true
		}
	}
	return false
}

// IsProductionPosture 是「生产敏感姿态」的**宽定义**统一实现（G-16 宽口径单一事实源）：
// 显式标签 ∨ GIN_MODE=release ∨ 任一监听地址非环回 ∨ 部署方声明必须加密（remoteDeclared）。
//
// 与 IsProductionLabeled（窄定义）的分工：窄定义用于**启动期 fail-closed 门禁**
// （宽定义会把绑 0.0.0.0 的开发 compose 误判成生产而拒绝启动）；宽定义用于
// **运行期保护姿态判定**（如 /metrics 免密分支的生产标注口径）——监听面一旦可被
// 远端触达，就不再存在「忘了配标签所以匿名可读」的豁免。
//
// 历史缺口（2026-09 metrics 复核）：宽定义此前只有 service-hub 一份私有实现，
// 另三个 A 档服务的 metrics 装配用窄定义，导致同一份非环回编排里
// service-hub 要凭据、其余三家匿名 200 —— 同一部署两种姿态。现统一收敛到本函数。
//
// hosts 传给本进程将要监听的地址（可带端口）；空串与 ":端口" 一律按对外暴露处理，
// 因为 net.Listen 对空主机的语义就是绑定全部网卡（见 IsLoopbackHost）。
// tlsEnabled 传本服务「当前确实以标准 TLS 加密监听」
// 的派生位——注意 privacy-engine 无明文档档（恒加密），该位无区分度，
// 调用方应只传 RequireTLS 类声明位。
func IsProductionPosture(hosts []string, tlsEnabled, requireTLS bool, labelVars ...string) bool {
	if IsProductionLabeled(labelVars...) {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(os.Getenv("GIN_MODE")), "release") {
		return true
	}
	if tlsEnabled || requireTLS {
		return true
	}
	for _, h := range hosts {
		if !IsLoopbackHost(h) {
			return true
		}
	}
	return false
}
