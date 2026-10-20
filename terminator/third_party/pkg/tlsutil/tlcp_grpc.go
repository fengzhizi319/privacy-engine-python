package tlsutil

import (
	"context"
	"errors"
	"fmt"
	"net"

	"gitee.com/Trisia/gotlcp/tlcp"
	"google.golang.org/grpc/credentials"
)

// ==============================================================================
// gRPC over 国密 TLCP（GM/T 0024）
//
// 【为什么要自己实现 TransportCredentials】
// gRPC-Go 内建的 credentials.NewTLS 只认 crypto/tls。gotlcp 是独立协议栈，
// 连接类型是 *tlcp.Conn，握手结果是 tlcp.ConnectionState —— 两者无法互换。
// 而 gRPC 的扩展点恰好是 credentials.TransportCredentials 接口，
// 实现它即可让 gRPC 跑在任意自定义传输上。
//
// 【为什么不能复用 credentials.TLSInfo】
// TLSInfo 只装 tls.ConnectionState，而 TLCP 的对端证书是
// []*smx509.Certificate —— **Go 标准库根本解析不了 SM2 证书**
// （x509: unsupported elliptic curve），塞不进 tls.ConnectionState。
// 故另立 TLCPInfo，并让它直接携带传输无关的 PeerIdentity，
// 使上层鉴权拦截器不必关心传输类型。
// ==============================================================================

// TLCPInfo 是国密 TLCP 连接的 gRPC AuthInfo。
//
// 除 gRPC 要求的 AuthType 外，它直接携带 PeerIdentity ——
// 上层用 PeerFromGRPCAuthInfo 取用，无需再碰 smx509 类型。
type TLCPInfo struct {
	// State 是完整握手状态，供需要协商套件等细节的诊断使用。
	State tlcp.ConnectionState
	// Peer 是传输无关的对端证书视图；对端未出示证书时为 nil。
	Peer *PeerIdentity

	credentials.CommonAuthInfo
}

// AuthType 实现 credentials.AuthInfo。
// 返回 "tlcp" 而非 "tls"：让日志与诊断能区分两种传输，
// 也避免上层误以为可以把它当 TLSInfo 断言。
func (TLCPInfo) AuthType() string { return "tlcp" }

// PeerFromGRPCAuthInfo 从 gRPC AuthInfo 取对端证书视图，**同时覆盖两种传输**。
//
// 这是 gRPC 侧的唯一入口，与 REST 侧的 pkg/auth.PeerFromRequest 对称：
// 收敛之后「有没有接上 TLCP」是一处实现，而不是每个拦截器各自的疏漏。
//
// 返回 nil 表示对端未出示证书 —— 调用方必须按 fail-closed 处理。
func PeerFromGRPCAuthInfo(info credentials.AuthInfo) *PeerIdentity {
	switch v := info.(type) {
	case TLCPInfo:
		return v.Peer
	case *TLCPInfo:
		if v == nil {
			return nil
		}
		return v.Peer
	case credentials.TLSInfo:
		return PeerFromTLSState(&v.State)
	}
	return nil
}

// tlcpCreds 是 TLCP 的 gRPC 传输凭据，服务端与客户端共用一套实现。
type tlcpCreds struct {
	cfg    *tlcp.Config
	server bool
	// serverName 仅客户端侧使用；gRPC 会通过 OverrideServerName 或
	// ClientHandshake 的 authority 参数告知目标名。
	serverName string
}

// NewTLCPServerCredentials 构造服务端 TLCP 传输凭据。
//
// cfg 由 BuildTLCPConfig 产出（已完成 sign/enc 双证书装配、KeyUsage 强校验
// 与国密套件白名单）。ClientAuth 是否要求客户端证书同样由 cfg 决定。
func NewTLCPServerCredentials(cfg *tlcp.Config) (credentials.TransportCredentials, error) {
	if cfg == nil {
		return nil, errors.New("tlsutil: nil TLCP config for gRPC server credentials")
	}
	return &tlcpCreds{cfg: cfg, server: true}, nil
}

// NewTLCPClientCredentials 构造客户端 TLCP 传输凭据。
func NewTLCPClientCredentials(cfg *tlcp.Config) (credentials.TransportCredentials, error) {
	if cfg == nil {
		return nil, errors.New("tlsutil: nil TLCP config for gRPC client credentials")
	}
	return &tlcpCreds{cfg: cfg}, nil
}

// ServerHandshake 实现服务端握手。
func (c *tlcpCreds) ServerHandshake(raw net.Conn) (net.Conn, credentials.AuthInfo, error) {
	conn := tlcp.Server(raw, c.cfg)
	if err := conn.Handshake(); err != nil {
		// 握手失败必须关掉底层连接：gRPC 不会替我们收拾一个半开的 conn。
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
}

// ClientHandshake 实现客户端握手。
//
// authority 是 gRPC 给出的目标名。仅当 cfg 未显式设置 ServerName 时才用它 ——
// 显式配置优先，避免「配了 ServerName 却被 authority 悄悄覆盖」这种
// 排查起来毫无线索的情形。
func (c *tlcpCreds) ClientHandshake(ctx context.Context, authority string, raw net.Conn) (net.Conn, credentials.AuthInfo, error) {
	cfg := c.cfg.Clone()
	if cfg.ServerName == "" {
		name := c.serverName
		if name == "" {
			name = authority
		}
		// authority 可能带端口（host:port），TLCP 的 ServerName 只要主机名。
		if host, _, err := net.SplitHostPort(name); err == nil {
			name = host
		}
		cfg.ServerName = name
	}

	conn := tlcp.Client(raw, cfg)
	// 尊重 ctx 取消/超时：gotlcp 的 Handshake 本身不接受 ctx，
	// 故在另一 goroutine 里握手并由 ctx 决定是否放弃。
	// 不这么做，一个不响应的对端会让 gRPC 的 DialContext 超时形同虚设。
	errCh := make(chan error, 1)
	go func() { errCh <- conn.Handshake() }()
	select {
	case err := <-errCh:
		if err != nil {
			_ = raw.Close()
			return nil, nil, fmt.Errorf("tlsutil: TLCP client handshake: %w", err)
		}
	case <-ctx.Done():
		_ = raw.Close()
		return nil, nil, fmt.Errorf("tlsutil: TLCP client handshake canceled: %w", ctx.Err())
	}

	st := conn.ConnectionState()
	return conn, TLCPInfo{
		State: st,
		Peer:  PeerFromTLCPConn(conn),
		CommonAuthInfo: credentials.CommonAuthInfo{
			SecurityLevel: credentials.PrivacyAndIntegrity,
		},
	}, nil
}

// Info 实现 credentials.TransportCredentials。
//
// SecurityProtocol 报 "tlcp" 而非 "tls"：gRPC 会把它用于日志与
// xDS 等场景的协议判定，报错会让诊断把国密链路误认成标准 TLS。
func (c *tlcpCreds) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{
		SecurityProtocol: "tlcp",
		SecurityVersion:  "1.1",
		ServerName:       c.serverName,
	}
}

// Clone 实现 credentials.TransportCredentials。
func (c *tlcpCreds) Clone() credentials.TransportCredentials {
	return &tlcpCreds{cfg: c.cfg.Clone(), server: c.server, serverName: c.serverName}
}

// OverrideServerName 实现 credentials.TransportCredentials。
func (c *tlcpCreds) OverrideServerName(name string) error {
	c.serverName = name
	return nil
}
