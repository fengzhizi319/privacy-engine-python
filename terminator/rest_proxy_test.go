package main

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
)

type capturedRequest struct {
	header http.Header
	path   string
	method string
	body   string
}

func TestRestProxyStripsForgeryAndInjectsSignedIdentity(t *testing.T) {
	requests := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		requests <- capturedRequest{
			header: r.Header.Clone(),
			path:   r.URL.Path,
			method: r.Method,
			body:   string(body),
		}
		w.Header().Set("X-Upstream", "ok")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	target, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parse upstream: %v", err)
	}
	signer := testSigner(t)
	proxy := newRESTProxy(target, signer, nil, true)

	req := httptest.NewRequest(http.MethodPost, "/v1/privacy/mask", strings.NewReader(`{"value":"x"}`))
	req.Header.Set(headerClientCN, "attacker")
	req.Header.Set(headerClientCertSHA256, "deadbeef")
	req.Header.Set(headerIdentitySig, "forged")
	req = req.WithContext(tlsutil.ContextWithPeer(req.Context(), testPeer()))

	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("X-Upstream") != "ok" {
		t.Fatal("upstream response header was not forwarded")
	}

	got := <-requests
	if got.path != "/v1/privacy/mask" || got.method != http.MethodPost {
		t.Fatalf("upstream saw %s %s", got.method, got.path)
	}
	if got.header.Get(headerClientCN) != "privshield-test-client" {
		t.Fatalf("injected CN = %q", got.header.Get(headerClientCN))
	}
	if got.header.Get(headerClientCertSHA256) == "deadbeef" {
		t.Fatal("forged cert fingerprint reached upstream")
	}
	timestamp, err := strconv.ParseInt(got.header.Get(headerIdentityTimestamp), 10, 64)
	if err != nil {
		t.Fatalf("timestamp header = %q", got.header.Get(headerIdentityTimestamp))
	}
	wantSig := signer.sign(canonicalIdentity(
		got.method,
		got.path,
		got.header.Get(headerClientCN),
		got.header.Get(headerClientCertSHA256),
		got.header.Get(headerClientSPKI),
		timestamp,
	))
	if got.header.Get(headerIdentitySig) != wantSig {
		t.Fatalf("signature mismatch: got %q want %q", got.header.Get(headerIdentitySig), wantSig)
	}
}

func TestRestProxyRequiresVerifiedPeer(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Fatal("upstream must not be reached without a verified client certificate")
	}))
	defer upstream.Close()

	target, _ := url.Parse(upstream.URL)
	proxy := newRESTProxy(target, testSigner(t), nil, true)
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/privacy/mask", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	var envelope map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode envelope: %v", err)
	}
	if envelope["code"] != "UNAUTHENTICATED" {
		t.Fatalf("code = %q", envelope["code"])
	}

	unverified := testPeer()
	unverified.Verified = false
	req := httptest.NewRequest(http.MethodGet, "/v1/privacy/mask", nil)
	req = req.WithContext(tlsutil.ContextWithPeer(req.Context(), unverified))
	rec = httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("unverified peer status = %d, want 401", rec.Code)
	}
}

func TestRestProxyWhitelistRejectsUnknownCN(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Fatal("upstream must not be reached for an unauthorized CN")
	}))
	defer upstream.Close()

	path := writeWhitelistFile(t, `version: "1.0"
clients:
  - cn: some-other-service
    allowed_scopes: ["*"]
`)
	dw, err := tlsutil.NewDynamicWhitelist(path)
	if err != nil {
		t.Fatalf("NewDynamicWhitelist: %v", err)
	}
	defer dw.Close()

	target, _ := url.Parse(upstream.URL)
	proxy := newRESTProxy(target, testSigner(t), dw, true)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req = req.WithContext(tlsutil.ContextWithPeer(req.Context(), testPeer()))
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"PERMISSION_DENIED"`) {
		t.Fatalf("unexpected envelope: %s", rec.Body.String())
	}
}

func TestOpsProxyDoesNotRequirePeerButStripsForgery(t *testing.T) {
	requests := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- capturedRequest{header: r.Header.Clone()}
		_, _ = w.Write([]byte("metrics"))
	}))
	defer upstream.Close()

	target, _ := url.Parse(upstream.URL)
	proxy := newRESTProxy(target, testSigner(t), nil, false)
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.Header.Set(headerClientCN, "attacker")
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	got := <-requests
	if got.header.Get(headerClientCN) != "" {
		t.Fatalf("forged identity header reached ops upstream: %q", got.header.Get(headerClientCN))
	}
	if got.header.Get(headerIdentitySig) != "" {
		t.Fatal("ops request without peer must not carry an identity signature")
	}
}

func TestRestProxyUpstreamUnavailableReturns502(t *testing.T) {
	target, _ := url.Parse("http://127.0.0.1:1")
	proxy := newRESTProxy(target, testSigner(t), nil, true)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req = req.WithContext(tlsutil.ContextWithPeer(req.Context(), testPeer()))
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"UNAVAILABLE"`) {
		t.Fatalf("unexpected envelope: %s", rec.Body.String())
	}
}

func TestOpsReadinessHandler(t *testing.T) {
	opsUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/readyz" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"ok","from":"python"}`))
			return
		}
		_, _ = w.Write([]byte("ops"))
	}))
	defer opsUpstream.Close()
	restUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/readyz" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer restUpstream.Close()

	grpcListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	grpcAddr := grpcListener.Addr().String()
	defer grpcListener.Close()

	opsURL, _ := url.Parse(opsUpstream.URL)
	restURL, _ := url.Parse(restUpstream.URL)
	proxy := newRESTProxy(opsURL, testSigner(t), nil, false)

	var draining bool
	handler := opsReadinessHandler(proxy, opsURL, restURL, grpcAddr, func() bool { return draining })

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"from":"python"`) {
		t.Fatalf("readyz status=%d body=%s", rec.Code, rec.Body.String())
	}

	draining = true
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("draining readyz status = %d, want 503", rec.Code)
	}
	draining = false

	_ = grpcListener.Close()
	// 关闭监听后端口可能短暂仍可连接（内核回收时序）；先确认它确实已拒绝连接，
	// 否则这条断言失败会被误读成 readiness 逻辑缺陷。
	waitDialRefused(t, grpcAddr)
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("readyz without grpc upstream = %d, want 503", rec.Code)
	}

	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("health status = %d, want 200", rec.Code)
	}
}

// waitDialRefused 等待 address 不再接受 TCP 连接。监听器刚关闭时内核可能仍
// 短暂完成握手，直接断言会让测试偶发失败。
func waitDialRefused(t *testing.T, address string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		conn, err := net.DialTimeout("tcp", address, 100*time.Millisecond)
		if err != nil {
			return
		}
		_ = conn.Close()
		if time.Now().After(deadline) {
			t.Fatalf("%s still accepts connections after listener close", address)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
