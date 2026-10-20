// Package main — 终结器配置加载。
//
// 公开端口默认 :8079（REST）/ :50051（gRPC）/ :9079（运维），回环上游默认
// 18079 / 15051 / 19079。TLS 材料沿用 Go 引擎的 PRIVACY_* 约定；终结器专属
// 开关使用 PRIVACY_TERMINATOR_* 前缀，避免与 Python 引擎同名变量冲突。
package main

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	pkgconfig "github.com/fengzhizi319/PrivShield-go/pkg/config"
	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
)

const (
	defaultRESTListen = ":8079"
	defaultGRPCListen = ":50051"
	defaultOpsListen  = ":9079"

	defaultRESTUpstream = "http://127.0.0.1:18079"
	defaultGRPCUpstream = "127.0.0.1:15051"
	defaultOpsUpstream  = "http://127.0.0.1:19079"

	maxGRPCMessageSize = 64 << 20
)

type config struct {
	serviceName string

	restListen string
	grpcListen string
	opsListen  string

	restUpstream string
	grpcUpstream string
	opsUpstream  string

	stack    tlsutil.ListenerStack
	opsStack tlsutil.ListenerStack

	tlsCertFile    string
	tlsKeyFile     string
	tlsKeyPassword string
	tlsCAFile      string
	tlsClientAuth  string

	tlcpSignCertFile string
	tlcpSignKeyFile  string
	tlcpEncCertFile  string
	tlcpEncKeyFile   string
	tlcpClientCAFile string
	tlcpClientAuth   string

	opsTLSCertFile string
	opsTLSKeyFile  string
	opsTLSCAFile   string

	whitelistFile string

	identityEnabled bool
	identityKey     []byte
	identityAlgo    string
	identityMaxSkew time.Duration

	drainSeconds     int
	grpcGraceSeconds int
}

func loadConfig() (*config, error) {
	stack, err := tlsutil.ParseListenerStack(envString("PRIVACY_TLS_STACK", ""))
	if err != nil {
		return nil, fmt.Errorf("PRIVACY_TLS_STACK: %w", err)
	}
	opsStack, err := tlsutil.ParseListenerStack(envString("PRIVACY_OPS_TLS_STACK", string(tlsutil.StackBoth)))
	if err != nil {
		return nil, fmt.Errorf("PRIVACY_OPS_TLS_STACK: %w", err)
	}

	cfg := &config{
		serviceName: envString("PRIVACY_TERMINATOR_SERVICE_NAME", "privshield-python-terminator"),

		restListen: envString("PRIVACY_TERMINATOR_REST_LISTEN", defaultRESTListen),
		grpcListen: envString("PRIVACY_TERMINATOR_GRPC_LISTEN", defaultGRPCListen),
		opsListen:  envString("PRIVACY_TERMINATOR_OPS_LISTEN", defaultOpsListen),

		restUpstream: envString("PRIVACY_TERMINATOR_REST_UPSTREAM", defaultRESTUpstream),
		grpcUpstream: envString("PRIVACY_TERMINATOR_GRPC_UPSTREAM", defaultGRPCUpstream),
		opsUpstream:  envString("PRIVACY_TERMINATOR_OPS_UPSTREAM", defaultOpsUpstream),

		stack:    stack,
		opsStack: opsStack,

		tlsCertFile:    envString("PRIVACY_TLS_CERT_FILE", ""),
		tlsKeyFile:     envString("PRIVACY_TLS_KEY_FILE", ""),
		tlsKeyPassword: envString("PRIVACY_TLS_KEY_PASSWORD", ""),
		tlsCAFile:      envString("PRIVACY_TLS_CA_FILE", ""),
		tlsClientAuth:  envString("PRIVACY_TLS_CLIENT_AUTH", "requireandverify"),

		tlcpSignCertFile: repoRelative(envString("PRIVACY_TLCP_SIGN_CERT_FILE", "config/certs/tlcp/server-sign.crt")),
		tlcpSignKeyFile:  repoRelative(envString("PRIVACY_TLCP_SIGN_KEY_FILE", "config/certs/tlcp/server-sign.key")),
		tlcpEncCertFile:  repoRelative(envString("PRIVACY_TLCP_ENC_CERT_FILE", "config/certs/tlcp/server-enc.crt")),
		tlcpEncKeyFile:   repoRelative(envString("PRIVACY_TLCP_ENC_KEY_FILE", "config/certs/tlcp/server-enc.key")),
		tlcpClientCAFile: repoRelative(envString("PRIVACY_TLCP_CLIENT_CA_FILE", "config/certs/tlcp/ca.crt")),
		tlcpClientAuth:   envString("PRIVACY_TLCP_CLIENT_AUTH", "requireandverify"),

		opsTLSCertFile: envString("PRIVACY_OPS_TLS_CERT_FILE", ""),
		opsTLSKeyFile:  envString("PRIVACY_OPS_TLS_KEY_FILE", ""),
		opsTLSCAFile:   envString("PRIVACY_OPS_TLS_CA_FILE", ""),

		whitelistFile: strings.TrimSpace(os.Getenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE")),

		identityEnabled: envBool("PRIVACY_TERMINATOR_IDENTITY_ENABLED", true),
		identityAlgo:    envString("PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO", "sm3"),
		identityMaxSkew: time.Duration(envInt("PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS", 60)) * time.Second,

		drainSeconds:     envInt("PRIVACY_SHUTDOWN_DRAIN_SECONDS", 5),
		grpcGraceSeconds: envInt("PRIVACY_GRPC_GRACEFUL_STOP_SECONDS", 15),
	}

	if cfg.opsTLSCertFile == "" {
		cfg.opsTLSCertFile = firstNonEmpty(cfg.tlsCertFile, repoRelative("config/certs/server.crt"))
	}
	if cfg.opsTLSKeyFile == "" {
		cfg.opsTLSKeyFile = firstNonEmpty(cfg.tlsKeyFile, repoRelative("config/certs/server.key"))
	}
	if cfg.opsTLSCAFile == "" {
		cfg.opsTLSCAFile = firstNonEmpty(cfg.tlsCAFile, repoRelative("config/certs/ca.crt"))
	}

	if cfg.identityEnabled {
		key, err := loadIdentityKey()
		if err != nil {
			return nil, err
		}
		cfg.identityKey = key
		if _, err := newIdentitySigner(key, cfg.identityAlgo); err != nil {
			return nil, err
		}
		if cfg.identityMaxSkew <= 0 {
			return nil, fmt.Errorf("PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS must be > 0")
		}
	}

	if err := validateListen("PRIVACY_TERMINATOR_REST_LISTEN", cfg.restListen); err != nil {
		return nil, err
	}
	if err := validateListen("PRIVACY_TERMINATOR_GRPC_LISTEN", cfg.grpcListen); err != nil {
		return nil, err
	}
	if err := validateListen("PRIVACY_TERMINATOR_OPS_LISTEN", cfg.opsListen); err != nil {
		return nil, err
	}
	if cfg.restListen == cfg.grpcListen || cfg.restListen == cfg.opsListen || cfg.grpcListen == cfg.opsListen {
		return nil, fmt.Errorf("terminator listeners must be distinct: rest=%s grpc=%s ops=%s",
			cfg.restListen, cfg.grpcListen, cfg.opsListen)
	}
	if _, err := parseHTTPUpstream("PRIVACY_TERMINATOR_REST_UPSTREAM", cfg.restUpstream); err != nil {
		return nil, err
	}
	if _, err := parseHTTPUpstream("PRIVACY_TERMINATOR_OPS_UPSTREAM", cfg.opsUpstream); err != nil {
		return nil, err
	}
	if err := requireLoopbackUpstream("PRIVACY_TERMINATOR_GRPC_UPSTREAM", cfg.grpcUpstream); err != nil {
		return nil, err
	}
	return cfg, nil
}

// businessServerStack 装配业务口 REST 与 gRPC 共用的协议栈。
func (c *config) businessServerStack() *tlsutil.ServerStack {
	return &tlsutil.ServerStack{
		Stack: c.stack,
		TLCP: &tlsutil.TLCPConfig{
			SignCertFile: c.tlcpSignCertFile,
			SignKeyFile:  c.tlcpSignKeyFile,
			EncCertFile:  c.tlcpEncCertFile,
			EncKeyFile:   c.tlcpEncKeyFile,
			ClientCAFile: c.tlcpClientCAFile,
			ClientAuth:   c.tlcpClientAuth,
		},
		TLS: &tlsutil.ServerTLSConfig{
			CertFile:    c.tlsCertFile,
			KeyFile:     c.tlsKeyFile,
			KeyPassword: c.tlsKeyPassword,
			CAFile:      c.tlsCAFile,
			ClientAuth:  c.tlsClientAuth,
		},
	}
}

// opsServerStack 装配运维口协议栈。TLCP 材料与业务口同一份；标准 TLS 使用
// 独立的 OpsTLS* 材料，客户端认证档位固定为 verify-if-given（kubelet 探针
// 出示不了证书，探针路径本身免凭证；数据路径仍由 Python 的 metrics 门禁裁决）。
func (c *config) opsServerStack() *tlsutil.ServerStack {
	return &tlsutil.ServerStack{
		Stack: c.opsStack,
		TLCP: &tlsutil.TLCPConfig{
			SignCertFile: c.tlcpSignCertFile,
			SignKeyFile:  c.tlcpSignKeyFile,
			EncCertFile:  c.tlcpEncCertFile,
			EncKeyFile:   c.tlcpEncKeyFile,
			ClientCAFile: c.tlcpClientCAFile,
			ClientAuth:   c.tlcpClientAuth,
		},
		TLS: &tlsutil.ServerTLSConfig{
			CertFile:   c.opsTLSCertFile,
			KeyFile:    c.opsTLSKeyFile,
			CAFile:     c.opsTLSCAFile,
			ClientAuth: "verify",
		},
	}
}

func loadIdentityKey() ([]byte, error) {
	if path := strings.TrimSpace(os.Getenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE")); path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE %s: %w", path, err)
		}
		key := []byte(strings.TrimSpace(string(raw)))
		if len(key) == 0 {
			return nil, fmt.Errorf("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE %s is empty", path)
		}
		return key, nil
	}
	if value := strings.TrimSpace(os.Getenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY")); value != "" {
		return []byte(value), nil
	}
	return nil, fmt.Errorf("terminator identity injection is enabled but no HMAC key is configured " +
		"(set PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE or PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY, " +
		"or explicitly disable with PRIVACY_TERMINATOR_IDENTITY_ENABLED=false)")
}

func validateListen(envName, address string) error {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("%s=%q is not host:port: %w", envName, address, err)
	}
	if port == "" {
		return fmt.Errorf("%s=%q has an empty port", envName, address)
	}
	if n, err := strconv.Atoi(port); err != nil || n < 0 || n > 65535 {
		return fmt.Errorf("%s=%q has an invalid port", envName, address)
	}
	_ = host
	return nil
}

func parseHTTPUpstream(envName, raw string) (*url.URL, error) {
	target, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("%s=%q: %w", envName, raw, err)
	}
	if target.Scheme != "http" && target.Scheme != "https" {
		return nil, fmt.Errorf("%s=%q must be an http(s) URL", envName, raw)
	}
	if target.Host == "" {
		return nil, fmt.Errorf("%s=%q has no host", envName, raw)
	}
	if err := requireLoopbackUpstream(envName, target.Host); err != nil {
		return nil, err
	}
	return target, nil
}

// requireLoopbackUpstream 是本拓扑的安全不变量：Python 引擎不得直接暴露
// 非回环地址。终结器是唯一 TLS 边界，上游必须是同 Pod / 同机的回环端口。
func requireLoopbackUpstream(envName, hostPort string) error {
	host, _, err := net.SplitHostPort(hostPort)
	if err != nil {
		return fmt.Errorf("%s=%q is not host:port: %w", envName, hostPort, err)
	}
	if strings.EqualFold(host, "localhost") {
		return nil
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("%s=%q must point at a loopback upstream; the Python engine must not be "+
			"reachable without passing through the terminator", envName, hostPort)
	}
	return nil
}

func envString(name, def string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return def
}

func envBool(name string, def bool) bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	if value == "" {
		return def
	}
	return value == "true" || value == "1" || value == "yes" || value == "on"
}

func envInt(name string, def int) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return def
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return def
	}
	return parsed
}

func repoRelative(path string) string {
	if path == "" {
		return ""
	}
	return pkgconfig.RepoRelativeDefault(path)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
