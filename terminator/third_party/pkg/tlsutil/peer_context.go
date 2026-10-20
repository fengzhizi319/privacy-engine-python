package tlsutil

import (
	"context"
	"net"
	"sync"

	"gitee.com/Trisia/gotlcp/tlcp"
)

// peerContextKey 是 PeerIdentity 在 context 中的键。
// 用私有空结构体做键：避免与任何其它包的 context 键冲突。
type peerContextKey struct{}

// peerProvider 延迟提供对端视图。
//
// 【为什么必须惰性】
// http.Server.ConnContext 在**握手之前**被调用 —— 那时
// tlcp.Conn.ConnectionState().PeerCertificates 恒为空。
// 若在 ConnContext 里当场提取，拿到的永远是「没有客户端证书」，
// 于是国密链路下所有证书约束都会被判失配（fail-closed 方向，
// 但服务对合法调用方也不可用）。
//
// 故这里只登记一个 provider，等**首个请求到达时**（握手已完成）才真正提取。
type peerProvider struct {
	conn *tlcp.Conn
	once sync.Once
	peer *PeerIdentity
}

func (p *peerProvider) get() *PeerIdentity {
	if p == nil || p.conn == nil {
		return nil
	}
	p.once.Do(func() { p.peer = PeerFromTLCPConn(p.conn) })
	return p.peer
}

// TLCPConnContext 返回可直接赋给 http.Server.ConnContext 的函数，
// 把国密 TLCP 连接的对端证书桥接进 request context。
//
// ==============================================================================
// 【为什么这一步不可省 —— 不接它，国密模式下证书准入全部失效】
//
// gotlcp 的连接不是 *tls.Conn，因此 net/http **无法填充 http.Request.TLS**
// —— 在 TLCP 监听下它恒为 nil。而应用层的客户端证书准入
// （cn= / fp= / spki=，见 pkg/auth）正是从 r.TLS 取对端证书的。
//
// 后果分两种，都不可接受：
//   - 若准入逻辑把「取不到证书」当成「无约束」→ 静默放行，
//     security.md 声称"彻底封堵同 CA 冒名"的机制一条都不执行；
//   - 若当成失配（本仓库现在的口径）→ 凡绑定了证书约束的凭据全部 401，
//     服务对合法调用方也不可用。
//
// 本仓库此前只有 audit-log 接了这个桥接；service-hub 与 privacy-engine
// 都没接，而这两个服务的生产配置里国密**已经是开启状态**。
//
// 用法（三个服务一致，不要各抄一份）：
//
//	httpServer.ConnContext = tlsutil.TLCPConnContext(nil)
//
// base 非 nil 时先执行它，便于与其它 ConnContext 逻辑串联。
func TLCPConnContext(base func(context.Context, net.Conn) context.Context) func(context.Context, net.Conn) context.Context {
	return func(ctx context.Context, c net.Conn) context.Context {
		if base != nil {
			ctx = base(ctx, c)
		}
		if tc, ok := c.(*tlcp.Conn); ok {
			return context.WithValue(ctx, peerContextKey{}, &peerProvider{conn: tc})
		}
		return ctx
	}
}

// ContextWithPeer 直接注入一个已知的对端视图（测试与非 TLCP 场景用）。
func ContextWithPeer(ctx context.Context, p *PeerIdentity) context.Context {
	return context.WithValue(ctx, peerContextKey{}, p)
}

// PeerFromContext 取出 context 中的对端视图；不存在时返回 nil。
//
// 同时接受两种载荷：直接放入的 *PeerIdentity，以及 TLCPConnContext
// 登记的惰性 provider —— 后者在此刻才真正读取握手结果。
func PeerFromContext(ctx context.Context) *PeerIdentity {
	if ctx == nil {
		return nil
	}
	switch v := ctx.Value(peerContextKey{}).(type) {
	case *PeerIdentity:
		return v
	case *peerProvider:
		return v.get()
	case *stackPeerProvider:
		// 三协议栈桥接（stack.go）。漏掉这一分支的后果与漏掉连接类型相同：
		// 请求照常处理、PeerFromContext 恒 nil、PoP 静默不执行。
		return v.get()
	}
	return nil
}
