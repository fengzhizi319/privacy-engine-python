package tlsutil

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"

	"gitee.com/Trisia/gotlcp/pa"
	"gitee.com/Trisia/gotlcp/tlcp"
	"google.golang.org/grpc/credentials"
)

// 服务端协议栈：一个开关，三档，没有明文。
//
// 【为什么是一个开关而不是两个布尔】
// 此前各服务用 *_TLS_ENABLED 与 *_TLS_NATIONAL_CIPHER 两个布尔组合出三个互斥
// 分支（TLCP / 标准 TLS / 明文），判定散在四个服务的监听代码里各写一遍。
// 两个布尔的四种组合里有一种（TLS=false 且 NATIONAL_CIPHER=true）语义含混，
// 而「国密默认开启」的翻转在其中一个服务上就曾因分支读 env 而非读 cfg 静默失效。
// 收敛为一个取值是协议栈**名字**的开关：写 tls 的人知道自己在写什么，
// 写错即拒绝启动，不存在含混组合。
//
// 【为什么业务口与运维口共用本实现】
// 运维口的客户端里有 kubelet 与 Prometheus，它们是 Go 标准库 TLS，握手层面
// 不会 TLCP，因此运维口需要同一端口自适应两种协议（gotlcp 的 pa 包）。
// 业务口默认只开国密，但过渡期（如替换中的真实数据源一时做不了 TLCP）也需要
// 双协议。两个口的差别只是默认值，故只写一份代码。
type ListenerStack string

const (
	// StackTLCP 仅国密 TLCP 双证书；标准 TLS 客户端在握手层被拒。生产基线。
	StackTLCP ListenerStack = "tlcp"
	// StackBoth 同一端口自适应 TLCP 与 TLS 1.3，按 ClientHello 记录层头分流。
	StackBoth ListenerStack = "both"
	// StackTLS 仅 TLS 1.3。境外部署 / 客户明确接受的非国密口径。
	StackTLS ListenerStack = "tls"
)

// ParseListenerStack 解析开关取值。空串即默认 tlcp；其他任何非法值报错——
// 协议栈选错是「国密静默失效」这一类缺陷的入口，必须 fail-fast。
func ParseListenerStack(raw string) (ListenerStack, error) {
	switch ListenerStack(strings.ToLower(strings.TrimSpace(raw))) {
	case "":
		return StackTLCP, nil
	case StackTLCP:
		return StackTLCP, nil
	case StackBoth:
		return StackBoth, nil
	case StackTLS:
		return StackTLS, nil
	default:
		return "", fmt.Errorf("tlsutil: invalid listener stack %q (want one of: tlcp, both, tls)", raw)
	}
}

// HasTLCP 报告本协议栈是否接受 TLCP 客户端。
func (s ListenerStack) HasTLCP() bool { return s == StackTLCP || s == StackBoth }

// HasTLS 报告本协议栈是否接受标准 TLS 客户端。
func (s ListenerStack) HasTLS() bool { return s == StackTLS || s == StackBoth }

// String 便于日志与 /readyz 上报。
func (s ListenerStack) String() string { return string(s) }

// ServerStack 是装配一个协议栈监听所需的全部材料。
// 所选协议栈需要的那一侧配置缺失即 BuildServerStack 报错；另一侧忽略。
type ServerStack struct {
	Stack ListenerStack
	// TLCP 在 HasTLCP() 时必填。
	TLCP *TLCPConfig
	// TLS 在 HasTLS() 时必填。
	TLS *ServerTLSConfig
}

// BuiltStack 是装配完成、可直接监听的协议栈。
type BuiltStack struct {
	Stack ListenerStack
	tlcp  *tlcp.Config
	tls   *tls.Config
}

// BuildServerStack 按开关装配所需的一侧或两侧配置。
//
// 缺证书在这里报错而不是在首个连接时——服务应当拒绝启动，而不是起来后
// 对每个连接握手失败。
func BuildServerStack(cfg *ServerStack) (*BuiltStack, error) {
	if cfg == nil {
		return nil, errors.New("tlsutil: nil server stack config")
	}
	stack, err := ParseListenerStack(string(cfg.Stack))
	if err != nil {
		return nil, err
	}
	out := &BuiltStack{Stack: stack}
	if stack.HasTLCP() {
		if cfg.TLCP == nil {
			return nil, fmt.Errorf("tlsutil: stack %s requires TLCP server config", stack)
		}
		// BuildTLCPConfig 以 Enabled 位为门，这里由协议栈决定它必然开启。
		tc := *cfg.TLCP
		tc.Enabled = true
		built, err := BuildTLCPConfig(&tc)
		if err != nil {
			return nil, fmt.Errorf("tlsutil: stack %s: %w", stack, err)
		}
		out.tlcp = built
	}
	if stack.HasTLS() {
		if cfg.TLS == nil {
			return nil, fmt.Errorf("tlsutil: stack %s requires TLS server config", stack)
		}
		tc := *cfg.TLS
		tc.Enabled = true
		built, err := BuildServerTLSConfig(&tc)
		if err != nil {
			return nil, fmt.Errorf("tlsutil: stack %s: %w", stack, err)
		}
		out.tls = built
	}
	return out, nil
}

// TLSConfig 返回标准 TLS 侧配置（仅 HasTLS 时非 nil）。供需要 http.Server.TLSConfig 的调用方使用。
func (b *BuiltStack) TLSConfig() *tls.Config { return b.tls }

// TLCPConfig 返回 TLCP 侧配置（仅 HasTLCP 时非 nil）。
func (b *BuiltStack) TLCPConfig() *tlcp.Config { return b.tlcp }

// Listen 按协议栈建立监听。
//
//   - tlcp：gotlcp 原生监听，返回 *tlcp.Conn；
//   - tls ：标准 tls.NewListener，返回 *tls.Conn（net/http 能据此填充 r.TLS）；
//   - both：pa.NewListener，返回 *pa.ProtocolSwitchServerConn，首个 Read 时检测
//     记录层版本号并包装为 *tlcp.Conn 或 *tls.Conn。
//
// 三种连接类型都由 StackConnContext 桥接成同一个 PeerIdentity。
func (b *BuiltStack) Listen(network, address string) (net.Listener, error) {
	switch b.Stack {
	case StackTLCP:
		return tlcp.Listen(network, address, b.tlcp)
	case StackTLS:
		inner, err := net.Listen(network, address)
		if err != nil {
			return nil, err
		}
		return tls.NewListener(inner, b.tls), nil
	case StackBoth:
		inner, err := net.Listen(network, address)
		if err != nil {
			return nil, err
		}
		l := pa.NewListener(inner, b.tlcp, b.tls)
		if l == nil {
			_ = inner.Close()
			return nil, errors.New("tlsutil: pa listener rejected config (both sides nil)")
		}
		return l, nil
	default:
		return nil, fmt.Errorf("tlsutil: unknown stack %q", b.Stack)
	}
}

// GRPCServerCredentials 返回与协议栈对应的 gRPC 服务端凭据。
//
// both 档需要在握手前探测协议——gRPC 的 TransportCredentials 拿到的是裸连接，
// 与 pa 的做法相同：读记录层头 5 字节判版本，再交给对应实现握手。
func (b *BuiltStack) GRPCServerCredentials() (credentials.TransportCredentials, error) {
	switch b.Stack {
	case StackTLCP:
		return NewTLCPServerCredentials(b.tlcp)
	case StackTLS:
		return credentials.NewTLS(b.tls), nil
	case StackBoth:
		// Unlike credentials.NewTLS, stackCreds does not inject ALPN automatically.
		// Clone to keep the HTTP listener config independent of gRPC's h2 contract.
		grpcTLS := b.tls.Clone()
		grpcTLS.NextProtos = []string{"h2"}
		return &stackCreds{tlcp: b.tlcp, tls: grpcTLS}, nil
	default:
		return nil, fmt.Errorf("tlsutil: unknown stack %q", b.Stack)
	}
}

// ───────────────────────── 对端证书桥接（三种连接类型） ─────────────────────────

// stackPeerProvider 惰性地从连接取对端证书。
//
// 【为什么必须惰性】
// http.Server.ConnContext 在 Accept 之后、握手之前触发；此时证书还没到。
// pa 更进一步：连协议都还没检测出来（首个 Read 才检测）。所以只能把连接
// 存下来，等鉴权中间件真正要用时（请求已到达 = 握手必已完成）再取。
//
// 【为什么必须认三种类型】
// 桥接只认 *tlcp.Conn 时，both 档下 PoP 会**静默不执行**——pa 返回的是它
// 自己的包装类型，类型断言不中就当作「无对端证书」放过去，而其余一切正常。
// 这正是本仓库在 TLCP 上踩过的系统性缺陷的同形复现，用例 TestStackBridge_*
// 逐一钉住三种类型。
type stackPeerProvider struct {
	conn net.Conn
	once sync.Once
	peer *PeerIdentity
}

func (p *stackPeerProvider) get() *PeerIdentity {
	if p == nil || p.conn == nil {
		return nil
	}
	p.once.Do(func() { p.peer = peerFromAnyConn(p.conn) })
	return p.peer
}

// peerFromAnyConn 从三种服务端连接类型中提取对端身份；未知类型返回 nil。
func peerFromAnyConn(c net.Conn) *PeerIdentity {
	switch v := c.(type) {
	case *tlcp.Conn:
		return PeerFromTLCPConn(v)
	case *tls.Conn:
		st := v.ConnectionState()
		return PeerFromTLSState(&st)
	case *pa.ProtocolSwitchServerConn:
		// 检测尚未发生时 ProtectedConn 为 nil：请求到达时不可能如此，
		// 但取到 nil 就返回 nil，绝不把「还没检测」误判成「无证书且合法」。
		inner := v.ProtectedConn()
		if inner == nil {
			return nil
		}
		return peerFromAnyConn(inner)
	}
	return nil
}

// StackConnContext 返回可挂到 http.Server.ConnContext 的桥接函数，
// 使 PeerFromContext 在三种协议栈下都能取到对端证书。
// 取代 TLCPConnContext（后者只认 *tlcp.Conn）。
func StackConnContext(base func(context.Context, net.Conn) context.Context) func(context.Context, net.Conn) context.Context {
	return func(ctx context.Context, c net.Conn) context.Context {
		if base != nil {
			ctx = base(ctx, c)
		}
		switch c.(type) {
		case *tlcp.Conn, *tls.Conn, *pa.ProtocolSwitchServerConn:
			return context.WithValue(ctx, peerContextKey{}, &stackPeerProvider{conn: c})
		}
		return ctx
	}
}

// ───────────────────────── gRPC both 档凭据 ─────────────────────────

// stackCreds 是 both 档的 gRPC 服务端凭据：探测记录层版本后分派握手。
type stackCreds struct {
	tlcp *tlcp.Config
	tls  *tls.Config
}

// peekConn 读走记录层头 5 字节判版本，再把这 5 字节原样回放给后续读者。
// 与 pa.ProtocolDetectConn 同一思路；不直接用它是因为其版本字段未导出。
type peekConn struct {
	net.Conn
	head []byte
}

func (p *peekConn) Read(b []byte) (int, error) {
	if len(p.head) == 0 {
		return p.Conn.Read(b)
	}
	n := copy(b, p.head)
	p.head = p.head[n:]
	return n, nil
}

func (c *stackCreds) ServerHandshake(raw net.Conn) (net.Conn, credentials.AuthInfo, error) {
	head := make([]byte, 5)
	if _, err := io.ReadFull(raw, head); err != nil {
		_ = raw.Close()
		return nil, nil, fmt.Errorf("tlsutil: read record header: %w", err)
	}
	pc := &peekConn{Conn: raw, head: head}
	// 记录层头：type(1) major(1) minor(1) length(2)。TLCP 版本 0x0101，TLS 0x03xx。
	switch head[1] {
	case 0x01:
		conn := tlcp.Server(pc, c.tlcp)
		if err := conn.Handshake(); err != nil {
			_ = raw.Close()
			return nil, nil, fmt.Errorf("tlsutil: TLCP server handshake: %w", err)
		}
		st := conn.ConnectionState()
		return conn, TLCPInfo{
			State: st,
			Peer:  PeerFromTLCPConn(conn),
			CommonAuthInfo: credentials.CommonAuthInfo{
				SecurityLevel: credentials.PrivacyAndIntegrity,
			},
		}, nil
	case 0x03:
		conn := tls.Server(pc, c.tls)
		if err := conn.Handshake(); err != nil {
			_ = raw.Close()
			return nil, nil, fmt.Errorf("tlsutil: TLS server handshake: %w", err)
		}
		return conn, credentials.TLSInfo{
			State: conn.ConnectionState(),
			CommonAuthInfo: credentials.CommonAuthInfo{
				SecurityLevel: credentials.PrivacyAndIntegrity,
			},
		}, nil
	default:
		_ = raw.Close()
		return nil, nil, fmt.Errorf("tlsutil: unknown record-layer major version 0x%02x", head[1])
	}
}

func (c *stackCreds) ClientHandshake(context.Context, string, net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return nil, nil, errors.New("tlsutil: stack credentials are server-only")
}

func (c *stackCreds) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{SecurityProtocol: "tlcp+tls", SecurityVersion: "GM/T 0024 | TLS 1.3"}
}

func (c *stackCreds) Clone() credentials.TransportCredentials {
	return &stackCreds{tlcp: c.tlcp, tls: c.tls}
}

func (c *stackCreds) OverrideServerName(string) error { return nil }

// ValidateMTLSClientAuthMode 校验业务口/运维口标准 TLS 侧的客户端认证档位。
//
// 只接受「必须出示且校验」这一族写法。verify_if_given / request / 空串在
// BuildServerTLSConfig 里是合法的（供其他场景），但在本项目的两个口上不是：
// 目标态规定三档协议栈都是 RequireAndVerifyClientCert，放松即是降级。
//
// 词表**只写这一处**。audit-log 曾自写一份只认 "require"/"verify" 的白名单，
// 把 k8s 清单里合法的 "requireandverify" 挡在启动期——与 TLCP 侧曾出过的
// 缺陷同形。各服务一律委托本函数，杜绝再次漂移。
func ValidateMTLSClientAuthMode(mode string) error {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "require", "requireandverify", "require_and_verify":
		return nil
	default:
		return fmt.Errorf("tlsutil: client auth mode %q not allowed; mTLS is mandatory (use require / requireandverify)", mode)
	}
}
