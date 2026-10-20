package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil/certtest"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const (
	methodUnary  = "/test.Echo/Unary"
	methodStream = "/test.Echo/Stream"
	methodFail   = "/test.Echo/Fail"
)

type capturedCall struct {
	method string
	md     metadata.MD
}

type grpcUpstream struct {
	mu    sync.Mutex
	calls []capturedCall
}

func (u *grpcUpstream) record(method string, md metadata.MD) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.calls = append(u.calls, capturedCall{method: method, md: md.Copy()})
}

func (u *grpcUpstream) only(t *testing.T) capturedCall {
	t.Helper()
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(u.calls) != 1 {
		t.Fatalf("upstream saw %d calls, want exactly 1", len(u.calls))
	}
	return u.calls[0]
}

func (u *grpcUpstream) count() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return len(u.calls)
}

func (u *grpcUpstream) handle(_ any, stream grpc.ServerStream) error {
	ctx := stream.Context()
	method, _ := grpc.Method(ctx)
	md, _ := metadata.FromIncomingContext(ctx)
	u.record(method, md)

	switch method {
	case methodFail:
		return status.Error(codes.PermissionDenied, "denied by upstream")
	case methodStream:
		if err := stream.SendHeader(metadata.Pairs("x-upstream-header", "stream")); err != nil {
			return err
		}
		stream.SetTrailer(metadata.Pairs("x-upstream-trailer", "stream-done"))
		for i := 0; i < 3; i++ {
			if err := stream.SendMsg(&rawMessage{data: []byte(fmt.Sprintf("chunk-%d", i))}); err != nil {
				return err
			}
		}
		return nil
	default:
		var msg rawMessage
		if err := stream.RecvMsg(&msg); err != nil {
			return err
		}
		if err := stream.SendHeader(metadata.Pairs("x-upstream-header", "echo")); err != nil {
			return err
		}
		stream.SetTrailer(metadata.Pairs("x-upstream-trailer", "echo-done"))
		return stream.SendMsg(&rawMessage{data: append([]byte("echo:"), msg.data...)})
	}
}

func startGRPCUpstream(t *testing.T, upstream *grpcUpstream) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen upstream: %v", err)
	}
	srv := grpc.NewServer(
		grpc.ForceServerCodec(rawCodec{}),
		grpc.UnknownServiceHandler(upstream.handle),
	)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(func() {
		srv.Stop()
		_ = lis.Close()
	})
	return lis.Addr().String()
}

// startTerminator 按测试证书体系装配一个只监听 gRPC 的终结器。
func startTerminator(t *testing.T, mode certtest.TLSMode, upstreamAddr, whitelistFile string) string {
	t.Helper()
	serverStack := &tlsutil.ServerStack{
		Stack: tlsutil.StackTLCP,
		TLCP: &tlsutil.TLCPConfig{
			SignCertFile: mode.SignCertFile,
			SignKeyFile:  mode.SignKeyFile,
			EncCertFile:  mode.EncCertFile,
			EncKeyFile:   mode.EncKeyFile,
			ClientCAFile: mode.CAFile,
			ClientAuth:   "requireandverify",
		},
		TLS: &tlsutil.ServerTLSConfig{
			CertFile:   mode.CertFile,
			KeyFile:    mode.KeyFile,
			CAFile:     mode.CAFile,
			ClientAuth: "requireandverify",
		},
	}
	if !mode.NationalCipher {
		serverStack.Stack = tlsutil.StackTLS
	}
	built, err := tlsutil.BuildServerStack(serverStack)
	if err != nil {
		t.Fatalf("build server stack (%s): %v", mode.Name, err)
	}

	cfg := &config{grpcUpstream: upstreamAddr, whitelistFile: whitelistFile}
	srv, proxy, dw, err := newGRPCServer(built, cfg, testSigner(t))
	if err != nil {
		t.Fatalf("build gRPC terminator: %v", err)
	}

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		_ = proxy.Close()
		t.Fatalf("listen terminator: %v", err)
	}
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(func() {
		srv.Stop()
		_ = proxy.Close()
		if dw != nil {
			dw.Close()
		}
		_ = lis.Close()
	})
	return lis.Addr().String()
}

func dialTerminator(t *testing.T, mode certtest.TLSMode, addr string) *grpc.ClientConn {
	t.Helper()
	var creds credentials.TransportCredentials
	if mode.NationalCipher {
		clientCfg, err := tlsutil.BuildTLCPClientConfig(&tlsutil.TLCPClientConfig{
			CAFile:       mode.CAFile,
			SignCertFile: mode.ClientSignCertFile,
			SignKeyFile:  mode.ClientSignKeyFile,
			EncCertFile:  mode.ClientEncCertFile,
			EncKeyFile:   mode.ClientEncKeyFile,
		})
		if err != nil {
			t.Fatalf("build TLCP client config: %v", err)
		}
		creds, err = tlsutil.NewTLCPClientCredentials(clientCfg)
		if err != nil {
			t.Fatalf("build TLCP client credentials: %v", err)
		}
	} else {
		cert, err := tls.LoadX509KeyPair(mode.ClientCertFile, mode.ClientKeyFile)
		if err != nil {
			t.Fatalf("load client cert: %v", err)
		}
		caPEM, err := os.ReadFile(mode.CAFile)
		if err != nil {
			t.Fatalf("read CA: %v", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caPEM) {
			t.Fatalf("parse CA: %s", mode.CAFile)
		}
		creds = credentials.NewTLS(&tls.Config{
			MinVersion:   tls.VersionTLS12,
			Certificates: []tls.Certificate{cert},
			RootCAs:      pool,
			ServerName:   "localhost",
		})
	}

	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(creds),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(rawCodec{})),
	)
	if err != nil {
		t.Fatalf("dial terminator: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

// clientCN 是各测试证书体系的客户端证书 CN。
func clientCN(mode certtest.TLSMode) string {
	if mode.NationalCipher {
		return "privshield-test-client-sign"
	}
	return "privshield-test-client"
}

func whitelistFor(t *testing.T, cn string) string {
	t.Helper()
	return writeWhitelistFile(t, fmt.Sprintf(`version: "1.0"
clients:
  - cn: %s
    allowed_scopes: ["*"]
`, cn))
}

func mdFirst(md metadata.MD, key string) string {
	values := md.Get(key)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

// assertSignedIdentity 校验上游收到的身份 metadata 是终结器签名的那一份，
// 且客户端伪造值已被剥离。
func assertSignedIdentity(t *testing.T, call capturedCall, signer *identitySigner, wantCN string) {
	t.Helper()
	cn := mdFirst(call.md, mdClientCN)
	if cn != wantCN {
		t.Fatalf("injected CN = %q, want %q (forged identity must be stripped)", cn, wantCN)
	}
	if values := call.md.Get(mdClientCN); len(values) != 1 {
		t.Fatalf("CN metadata has %d values, want 1: %v", len(values), values)
	}
	timestamp, err := strconv.ParseInt(mdFirst(call.md, mdIdentityTimestamp), 10, 64)
	if err != nil {
		t.Fatalf("identity timestamp = %q", mdFirst(call.md, mdIdentityTimestamp))
	}
	sig := mdFirst(call.md, mdIdentitySig)
	want := signer.sign(canonicalIdentity(
		"POST",
		call.method,
		cn,
		mdFirst(call.md, mdClientCertSHA256),
		mdFirst(call.md, mdClientSPKI),
		timestamp,
	))
	if sig != want {
		t.Fatalf("identity signature = %q, want %q", sig, want)
	}
}

func TestGRPCProxyTLSModes(t *testing.T) {
	for _, mode := range certtest.TLSModes(t) {
		mode := mode
		t.Run(mode.Name, func(t *testing.T) {
			t.Run("unary", func(t *testing.T) {
				upstream := &grpcUpstream{}
				upstreamAddr := startGRPCUpstream(t, upstream)
				terminatorAddr := startTerminator(t, mode, upstreamAddr, whitelistFor(t, clientCN(mode)))
				conn := dialTerminator(t, mode, terminatorAddr)

				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				ctx = metadata.AppendToOutgoingContext(ctx,
					mdClientCN, "attacker",
					mdClientCertSHA256, "deadbeef",
					mdIdentitySig, "forged",
					"x-request-id", "req-1",
				)

				var header, trailer metadata.MD
				var resp rawMessage
				if err := conn.Invoke(ctx, methodUnary, &rawMessage{data: []byte("hello")}, &resp,
					grpc.Header(&header), grpc.Trailer(&trailer)); err != nil {
					t.Fatalf("unary through terminator: %v", err)
				}
				if string(resp.data) != "echo:hello" {
					t.Fatalf("response = %q, want echo:hello", resp.data)
				}
				if mdFirst(header, "x-upstream-header") != "echo" {
					t.Fatalf("header = %v", header)
				}
				if mdFirst(trailer, "x-upstream-trailer") != "echo-done" {
					t.Fatalf("trailer = %v", trailer)
				}

				call := upstream.only(t)
				if call.method != methodUnary {
					t.Fatalf("upstream method = %q", call.method)
				}
				assertSignedIdentity(t, call, testSigner(t), clientCN(mode))
				if mdFirst(call.md, "x-request-id") != "req-1" {
					t.Fatalf("application metadata not forwarded: %v", call.md)
				}
				for _, value := range call.md.Get(mdClientCN) {
					if value == "attacker" {
						t.Fatal("forged CN metadata reached upstream")
					}
				}
			})

			t.Run("status_preserved", func(t *testing.T) {
				upstream := &grpcUpstream{}
				upstreamAddr := startGRPCUpstream(t, upstream)
				terminatorAddr := startTerminator(t, mode, upstreamAddr, whitelistFor(t, clientCN(mode)))
				conn := dialTerminator(t, mode, terminatorAddr)

				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				var resp rawMessage
				err := conn.Invoke(ctx, methodFail, &rawMessage{data: []byte("x")}, &resp)
				if status.Code(err) != codes.PermissionDenied {
					t.Fatalf("status = %v, want PermissionDenied (upstream status must not be downgraded)", status.Code(err))
				}
				if st, _ := status.FromError(err); st.Message() != "denied by upstream" {
					t.Fatalf("status message = %q", st.Message())
				}
			})

			t.Run("server_stream", func(t *testing.T) {
				upstream := &grpcUpstream{}
				upstreamAddr := startGRPCUpstream(t, upstream)
				terminatorAddr := startTerminator(t, mode, upstreamAddr, whitelistFor(t, clientCN(mode)))
				conn := dialTerminator(t, mode, terminatorAddr)

				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				stream, err := conn.NewStream(ctx, &grpc.StreamDesc{ServerStreams: true}, methodStream)
				if err != nil {
					t.Fatalf("open stream: %v", err)
				}
				if err := stream.SendMsg(&rawMessage{data: []byte("go")}); err != nil {
					t.Fatalf("send: %v", err)
				}
				if err := stream.CloseSend(); err != nil {
					t.Fatalf("close send: %v", err)
				}
				header, err := stream.Header()
				if err != nil {
					t.Fatalf("stream header: %v", err)
				}
				if mdFirst(header, "x-upstream-header") != "stream" {
					t.Fatalf("stream header = %v", header)
				}

				var chunks []string
				for {
					var msg rawMessage
					recvErr := stream.RecvMsg(&msg)
					if errors.Is(recvErr, io.EOF) {
						break
					}
					if recvErr != nil {
						t.Fatalf("recv: %v", recvErr)
					}
					chunks = append(chunks, string(msg.data))
				}
				if len(chunks) != 3 || chunks[0] != "chunk-0" || chunks[2] != "chunk-2" {
					t.Fatalf("chunks = %v", chunks)
				}
				if mdFirst(stream.Trailer(), "x-upstream-trailer") != "stream-done" {
					t.Fatalf("stream trailer = %v", stream.Trailer())
				}
				assertSignedIdentity(t, upstream.only(t), testSigner(t), clientCN(mode))
			})

			t.Run("whitelist_denied", func(t *testing.T) {
				upstream := &grpcUpstream{}
				upstreamAddr := startGRPCUpstream(t, upstream)
				terminatorAddr := startTerminator(t, mode, upstreamAddr, whitelistFor(t, "some-other-service"))
				conn := dialTerminator(t, mode, terminatorAddr)

				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				var resp rawMessage
				err := conn.Invoke(ctx, methodUnary, &rawMessage{data: []byte("x")}, &resp)
				if status.Code(err) != codes.PermissionDenied {
					t.Fatalf("status = %v, want PermissionDenied", status.Code(err))
				}
				if upstream.count() != 0 {
					t.Fatal("unauthorized client reached the upstream")
				}
			})
		})
	}
}

// TestGRPCProxyRejectsClientWithoutCert 锁定三档协议栈都是 mTLS：
// 无客户端证书的握手必须失败，而不是"只在传输层加密"。
func TestGRPCProxyRejectsClientWithoutCert(t *testing.T) {
	for _, mode := range certtest.TLSModes(t) {
		mode := mode
		t.Run(mode.Name, func(t *testing.T) {
			upstream := &grpcUpstream{}
			upstreamAddr := startGRPCUpstream(t, upstream)
			terminatorAddr := startTerminator(t, mode, upstreamAddr, "")

			var creds credentials.TransportCredentials
			if mode.NationalCipher {
				clientCfg, err := tlsutil.BuildTLCPClientConfig(&tlsutil.TLCPClientConfig{CAFile: mode.CAFile})
				if err != nil {
					t.Fatalf("build TLCP client config: %v", err)
				}
				creds, err = tlsutil.NewTLCPClientCredentials(clientCfg)
				if err != nil {
					t.Fatalf("build TLCP client credentials: %v", err)
				}
			} else {
				caPEM, err := os.ReadFile(mode.CAFile)
				if err != nil {
					t.Fatalf("read CA: %v", err)
				}
				pool := x509.NewCertPool()
				if !pool.AppendCertsFromPEM(caPEM) {
					t.Fatalf("parse CA")
				}
				creds = credentials.NewTLS(&tls.Config{
					MinVersion: tls.VersionTLS12,
					RootCAs:    pool,
					ServerName: "localhost",
				})
			}

			conn, err := grpc.NewClient(terminatorAddr,
				grpc.WithTransportCredentials(creds),
				grpc.WithDefaultCallOptions(grpc.ForceCodec(rawCodec{})),
			)
			if err != nil {
				t.Fatalf("dial: %v", err)
			}
			defer conn.Close()

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var resp rawMessage
			if err := conn.Invoke(ctx, methodUnary, &rawMessage{data: []byte("x")}, &resp); err == nil {
				t.Fatal("client without certificate completed an RPC; the stack is not mTLS")
			}
			if upstream.count() != 0 {
				t.Fatal("unauthenticated client reached the upstream")
			}
		})
	}
}
