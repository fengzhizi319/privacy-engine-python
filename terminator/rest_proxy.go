// Package main — REST / 运维口 L7 反向代理。
//
// 终结器先完成 mTLS 握手，再从连接 context 取对端证书身份，剥离客户端伪造的
// 身份头，然后为回环上游签发身份头。上游（Python）只信任这类带签名的回环请求。
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
)

type restProxy struct {
	target              *url.URL
	proxy               *httputil.ReverseProxy
	signer              *identitySigner
	whitelist           *tlsutil.DynamicWhitelist
	requireVerifiedPeer bool
}

func newRESTProxy(
	target *url.URL,
	signer *identitySigner,
	whitelist *tlsutil.DynamicWhitelist,
	requireVerifiedPeer bool,
) *restProxy {
	p := &restProxy{
		target:              target,
		signer:              signer,
		whitelist:           whitelist,
		requireVerifiedPeer: requireVerifiedPeer,
	}
	transport := &http.Transport{
		Proxy: nil,
		DialContext: (&net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          200,
		MaxIdleConnsPerHost:   100,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	p.proxy = &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(p.target)
			pr.Out.Host = p.target.Host
			// 先剥离客户端伪造的身份头，再注入终结器签名头。
			stripRESTIdentityHeaders(pr.Out.Header)
			if p.signer != nil {
				peer := tlsutil.PeerFromContext(pr.In.Context())
				for name, value := range p.signer.restHeaders(pr.In.Method, pr.In.URL.Path, peer) {
					pr.Out.Header.Set(name, value)
				}
			}
		},
		Transport:     transport,
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			slog.Warn("upstream request failed",
				"path", r.URL.Path, "upstream", p.target.String(), "err", err)
			writeProxyError(w, r, http.StatusBadGateway, "UNAVAILABLE", "upstream unavailable")
		},
	}
	return p
}

func (p *restProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	peer := tlsutil.PeerFromContext(r.Context())
	if p.requireVerifiedPeer && (peer == nil || peer.Empty() || !peer.Verified) {
		writeProxyError(w, r, http.StatusUnauthorized, "UNAUTHENTICATED", "client certificate required")
		return
	}
	if p.whitelist != nil && peer != nil && !peer.Empty() {
		identity, ok := authorizePeer(p.whitelist, peer)
		if !ok {
			slog.Warn("mTLS client rejected by whitelist",
				"identities", peerIdentities(peer), "path", r.URL.Path)
			writeProxyError(w, r, http.StatusForbidden, "PERMISSION_DENIED", "client certificate is not authorized")
			return
		}
		slog.Debug("mTLS client authorized", "identity", identity, "path", r.URL.Path)
	}
	p.proxy.ServeHTTP(w, r)
}

// writeProxyError 输出与 Python 引擎一致的 5 字段错误信封。
func writeProxyError(w http.ResponseWriter, r *http.Request, status int, code, message string) {
	traceID := ""
	if r != nil {
		traceID = r.Header.Get("X-Request-ID")
		if traceID == "" {
			traceID = r.Header.Get("X-Trace-ID")
		}
	}
	payload := map[string]string{
		"code":      code,
		"message":   message,
		"detail":    message,
		"trace_id":  traceID,
		"timestamp": time.Now().UTC().Format("2006-01-02T15:04:05.000Z"),
	}
	body, err := json.Marshal(payload)
	if err != nil {
		body = []byte(fmt.Sprintf(`{"code":%q,"message":%q}`, code, message))
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(append(body, '\n'))
}

// opsReadinessHandler 让终结器暴露本地 /health、/livez 与聚合 /readyz；
// 其余运维路径（/metrics、/ops/diagnostics、/debug/pprof/）原样代理到 Python
// 运维口。上游未就绪时 /readyz 返回 503，而不是反向代理的 502。
func opsReadinessHandler(
	proxy *restProxy,
	opsUpstream *url.URL,
	restUpstream *url.URL,
	grpcUpstream string,
	draining func() bool,
) http.Handler {
	client := &http.Client{Timeout: 3 * time.Second}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health", "/healthz", "/livez":
			if draining() {
				writeProxyError(w, r, http.StatusServiceUnavailable, "UNAVAILABLE", "terminator is draining")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("{\"status\":\"ok\"}\n"))
			return
		case "/readyz":
			if draining() {
				writeProxyError(w, r, http.StatusServiceUnavailable, "UNAVAILABLE", "terminator is draining")
				return
			}
			if !upstreamsReady(client, opsUpstream, restUpstream, grpcUpstream) {
				writeProxyError(w, r, http.StatusServiceUnavailable, "UNAVAILABLE", "upstream not ready")
				return
			}
			proxy.ServeHTTP(w, r)
			return
		default:
			proxy.ServeHTTP(w, r)
		}
	})
}

func upstreamsReady(client *http.Client, opsUpstream, restUpstream *url.URL, grpcUpstream string) bool {
	if !httpReady(client, opsUpstream) || !httpReady(client, restUpstream) {
		return false
	}
	conn, err := net.DialTimeout("tcp", grpcUpstream, 2*time.Second)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

func httpReady(client *http.Client, base *url.URL) bool {
	target := *base
	target.Path = "/readyz"
	target.RawQuery = ""
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}
