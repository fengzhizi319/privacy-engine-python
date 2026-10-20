// Package main 是 PrivShield Python 引擎的 TLCP / mTLS 终结器（唯一 TLS 边界）。
//
//	client ──TLCP / both / TLS-mTLS──▶ terminator ──loopback──▶ Python REST/gRPC/ops
//
// 终结器复用 Go 引擎同款 pkg/tlsutil（gotlcp + gmsm），不重复实现国密协议栈；
// 对回环上游做 L7 反代并注入 HMAC 签名的客户端证书身份，供 Python 侧做防御性
// 二次校验。协议栈三档（tlcp / both / tls）均为 mTLS，没有明文档。
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
	"google.golang.org/grpc"
)

// Version 由 -ldflags 注入。
var Version = "dev"

func main() {
	initLogger()
	cfg, err := loadConfig()
	if err != nil {
		slog.Error("terminator configuration rejected", "err", err)
		os.Exit(1)
	}
	if err := run(cfg); err != nil {
		slog.Error("terminator stopped with error", "err", err)
		os.Exit(1)
	}
}

func run(cfg *config) error {
	businessStack, err := tlsutil.BuildServerStack(cfg.businessServerStack())
	if err != nil {
		return fmt.Errorf("business server stack: %w", err)
	}
	opsStack, err := tlsutil.BuildServerStack(cfg.opsServerStack())
	if err != nil {
		return fmt.Errorf("ops server stack: %w", err)
	}

	var signer *identitySigner
	if cfg.identityEnabled {
		signer, err = newIdentitySigner(cfg.identityKey, cfg.identityAlgo)
		if err != nil {
			return err
		}
	} else {
		slog.Warn("identity header injection is DISABLED; Python can only authenticate with API keys")
	}

	var restWhitelist *tlsutil.DynamicWhitelist
	if cfg.whitelistFile != "" {
		restWhitelist, err = tlsutil.NewDynamicWhitelist(cfg.whitelistFile)
		if err != nil {
			return fmt.Errorf("load mTLS whitelist %s: %w", cfg.whitelistFile, err)
		}
		defer restWhitelist.Close()
	} else {
		slog.Warn("PRIVACY_AUTH_MTLS_WHITELIST_FILE is empty; CN authorization is left to the Python engine")
	}

	restTarget, err := parseHTTPUpstream("PRIVACY_TERMINATOR_REST_UPSTREAM", cfg.restUpstream)
	if err != nil {
		return err
	}
	opsTarget, err := parseHTTPUpstream("PRIVACY_TERMINATOR_OPS_UPSTREAM", cfg.opsUpstream)
	if err != nil {
		return err
	}

	restProxy := newRESTProxy(restTarget, signer, restWhitelist, true)
	opsProxy := newRESTProxy(opsTarget, signer, restWhitelist, false)

	grpcServer, grpcProxy, grpcWhitelist, err := newGRPCServer(businessStack, cfg, signer)
	if err != nil {
		return err
	}
	defer func() { _ = grpcProxy.Close() }()
	if grpcWhitelist != nil {
		defer grpcWhitelist.Close()
	}

	restListener, err := businessStack.Listen("tcp", cfg.restListen)
	if err != nil {
		return fmt.Errorf("listen REST %s: %w", cfg.restListen, err)
	}
	// gRPC 必须绑定裸 TCP：grpc.Creds 自己在 Accept 之后完成协议栈握手。
	// 用 businessStack.Listen 会把已包装的连接再交给凭据层，形成双重握手。
	grpcListener, err := net.Listen("tcp", cfg.grpcListen)
	if err != nil {
		_ = restListener.Close()
		return fmt.Errorf("listen gRPC %s: %w", cfg.grpcListen, err)
	}
	opsListener, err := opsStack.Listen("tcp", cfg.opsListen)
	if err != nil {
		_ = restListener.Close()
		_ = grpcListener.Close()
		return fmt.Errorf("listen ops %s: %w", cfg.opsListen, err)
	}

	var draining atomic.Bool
	restServer := &http.Server{
		Handler:           restProxy,
		ConnContext:       tlsutil.StackConnContext(nil),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	opsServer := &http.Server{
		Handler: opsReadinessHandler(
			opsProxy, opsTarget, restTarget, cfg.grpcUpstream, draining.Load,
		),
		ConnContext:       tlsutil.StackConnContext(nil),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	slog.Info("PrivShield Python terminator starting",
		"version", Version,
		"rest_listen", cfg.restListen,
		"grpc_listen", cfg.grpcListen,
		"ops_listen", cfg.opsListen,
		"tls_stack", businessStack.Stack.String(),
		"ops_tls_stack", opsStack.Stack.String(),
		"rest_upstream", restTarget.String(),
		"grpc_upstream", cfg.grpcUpstream,
		"ops_upstream", opsTarget.String(),
		"identity_enabled", cfg.identityEnabled,
		"identity_algo", cfg.identityAlgo,
		"whitelist_file", cfg.whitelistFile,
	)

	serveErr := make(chan error, 3)
	go func() {
		if err := restServer.Serve(restListener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- fmt.Errorf("REST server: %w", err)
		}
	}()
	go func() {
		if err := opsServer.Serve(opsListener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- fmt.Errorf("ops server: %w", err)
		}
	}()
	go func() {
		if err := grpcServer.Serve(grpcListener); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			serveErr <- fmt.Errorf("gRPC server: %w", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	select {
	case sig := <-quit:
		slog.Info("shutdown signal received, draining", "signal", sig.String())
	case err := <-serveErr:
		slog.Error("listener failed, shutting down", "err", err)
	}
	signal.Stop(quit)

	// 先置 unready，再排空流量。
	draining.Store(true)
	if cfg.drainSeconds > 0 {
		slog.Info("draining in-flight traffic", "seconds", cfg.drainSeconds)
		time.Sleep(time.Duration(cfg.drainSeconds) * time.Second)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := restServer.Shutdown(shutdownCtx); err != nil {
		slog.Warn("REST shutdown warning", "err", err)
	}
	if err := opsServer.Shutdown(shutdownCtx); err != nil {
		slog.Warn("ops shutdown warning", "err", err)
	}

	gracefulDone := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(gracefulDone)
	}()
	select {
	case <-gracefulDone:
	case <-time.After(time.Duration(cfg.grpcGraceSeconds) * time.Second):
		slog.Warn("gRPC graceful stop timed out, forcing stop", "seconds", cfg.grpcGraceSeconds)
		grpcServer.Stop()
	}
	slog.Info("PrivShield Python terminator stopped")
	return nil
}

func initLogger() {
	level := slog.LevelInfo
	switch strings.ToUpper(strings.TrimSpace(os.Getenv("PRIVACY_LOG_LEVEL"))) {
	case "DEBUG":
		level = slog.LevelDebug
	case "WARN":
		level = slog.LevelWarn
	case "ERROR":
		level = slog.LevelError
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level})))
}
