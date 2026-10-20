package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
)

// clearTerminatorEnv 把 loadConfig 读取的变量全部置空，避免宿主环境影响断言。
func clearTerminatorEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"PRIVACY_TERMINATOR_SERVICE_NAME",
		"PRIVACY_TERMINATOR_REST_LISTEN",
		"PRIVACY_TERMINATOR_GRPC_LISTEN",
		"PRIVACY_TERMINATOR_OPS_LISTEN",
		"PRIVACY_TERMINATOR_REST_UPSTREAM",
		"PRIVACY_TERMINATOR_GRPC_UPSTREAM",
		"PRIVACY_TERMINATOR_OPS_UPSTREAM",
		"PRIVACY_TLS_STACK",
		"PRIVACY_OPS_TLS_STACK",
		"PRIVACY_TLS_CERT_FILE",
		"PRIVACY_TLS_KEY_FILE",
		"PRIVACY_TLS_KEY_PASSWORD",
		"PRIVACY_TLS_CA_FILE",
		"PRIVACY_TLS_CLIENT_AUTH",
		"PRIVACY_TLCP_SIGN_CERT_FILE",
		"PRIVACY_TLCP_SIGN_KEY_FILE",
		"PRIVACY_TLCP_ENC_CERT_FILE",
		"PRIVACY_TLCP_ENC_KEY_FILE",
		"PRIVACY_TLCP_CLIENT_CA_FILE",
		"PRIVACY_TLCP_CLIENT_AUTH",
		"PRIVACY_OPS_TLS_CERT_FILE",
		"PRIVACY_OPS_TLS_KEY_FILE",
		"PRIVACY_OPS_TLS_CA_FILE",
		"PRIVACY_AUTH_MTLS_WHITELIST_FILE",
		"PRIVACY_TERMINATOR_IDENTITY_ENABLED",
		"PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO",
		"PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY",
		"PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE",
		"PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS",
		"PRIVACY_SHUTDOWN_DRAIN_SECONDS",
		"PRIVACY_GRPC_GRACEFUL_STOP_SECONDS",
	} {
		t.Setenv(name, "")
	}
}

func TestLoadConfigRejectsNonLoopbackUpstreams(t *testing.T) {
	cases := []struct {
		name string
		key  string
		val  string
	}{
		{"rest", "PRIVACY_TERMINATOR_REST_UPSTREAM", "http://10.0.0.5:18079"},
		{"grpc", "PRIVACY_TERMINATOR_GRPC_UPSTREAM", "10.0.0.5:15051"},
		{"ops", "PRIVACY_TERMINATOR_OPS_UPSTREAM", "http://192.168.1.10:19079"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clearTerminatorEnv(t)
			t.Setenv("PRIVACY_TERMINATOR_IDENTITY_ENABLED", "false")
			t.Setenv(tc.key, tc.val)
			if _, err := loadConfig(); err == nil {
				t.Fatalf("%s=%s was accepted; the Python engine must stay loopback-only", tc.key, tc.val)
			} else if !strings.Contains(err.Error(), "loopback") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestLoadConfigAcceptsLoopbackUpstreams(t *testing.T) {
	for _, upstream := range []struct {
		rest string
		grpc string
		ops  string
	}{
		{"http://127.0.0.1:18079", "127.0.0.1:15051", "http://127.0.0.1:19079"},
		{"http://localhost:18079", "localhost:15051", "http://localhost:19079"},
		{"http://[::1]:18079", "[::1]:15051", "http://[::1]:19079"},
	} {
		t.Run(upstream.grpc, func(t *testing.T) {
			clearTerminatorEnv(t)
			t.Setenv("PRIVACY_TERMINATOR_IDENTITY_ENABLED", "false")
			t.Setenv("PRIVACY_TERMINATOR_REST_UPSTREAM", upstream.rest)
			t.Setenv("PRIVACY_TERMINATOR_GRPC_UPSTREAM", upstream.grpc)
			t.Setenv("PRIVACY_TERMINATOR_OPS_UPSTREAM", upstream.ops)
			cfg, err := loadConfig()
			if err != nil {
				t.Fatalf("loopback upstream rejected: %v", err)
			}
			if cfg.grpcUpstream != upstream.grpc {
				t.Fatalf("grpc upstream = %q", cfg.grpcUpstream)
			}
		})
	}
}

func TestLoadConfigRejectsBadUpstreamScheme(t *testing.T) {
	clearTerminatorEnv(t)
	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_ENABLED", "false")
	t.Setenv("PRIVACY_TERMINATOR_REST_UPSTREAM", "ftp://127.0.0.1:18079")
	if _, err := loadConfig(); err == nil {
		t.Fatal("non-http(s) upstream scheme was accepted")
	}
}

func TestLoadConfigRequiresIdentityKeyWhenEnabled(t *testing.T) {
	clearTerminatorEnv(t)
	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_ENABLED", "true")
	if _, err := loadConfig(); err == nil {
		t.Fatal("identity injection enabled without an HMAC key was accepted")
	}

	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY", "short")
	if _, err := loadConfig(); err == nil {
		t.Fatal("identity HMAC key shorter than 16 bytes was accepted")
	}

	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY", testIdentityKey)
	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("valid identity key rejected: %v", err)
	}
	if string(cfg.identityKey) != testIdentityKey || cfg.identityAlgo != "sm3" {
		t.Fatalf("identity config = %q/%s", cfg.identityKey, cfg.identityAlgo)
	}
}

func TestLoadConfigIdentityKeyFileTakesPrecedence(t *testing.T) {
	clearTerminatorEnv(t)
	path := filepath.Join(t.TempDir(), "identity.key")
	if err := os.WriteFile(path, []byte("file-key-0123456789\n"), 0o600); err != nil {
		t.Fatalf("write key file: %v", err)
	}
	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY", "env-key-0123456789")
	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE", path)
	key, err := loadIdentityKey()
	if err != nil {
		t.Fatalf("loadIdentityKey: %v", err)
	}
	if string(key) != "file-key-0123456789" {
		t.Fatalf("key = %q; key file must win over the inline env value", key)
	}
}

func TestLoadConfigRejectsInvalidStackAndListeners(t *testing.T) {
	cases := []struct {
		name string
		key  string
		val  string
	}{
		{"tls_stack", "PRIVACY_TLS_STACK", "ssl"},
		{"ops_stack", "PRIVACY_OPS_TLS_STACK", "gm"},
		{"rest_listen", "PRIVACY_TERMINATOR_REST_LISTEN", "8079"},
		{"grpc_listen", "PRIVACY_TERMINATOR_GRPC_LISTEN", "127.0.0.1:not-a-port"},
		{"ops_listen", "PRIVACY_TERMINATOR_OPS_LISTEN", "127.0.0.1:70000"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clearTerminatorEnv(t)
			t.Setenv("PRIVACY_TERMINATOR_IDENTITY_ENABLED", "false")
			t.Setenv(tc.key, tc.val)
			if _, err := loadConfig(); err == nil {
				t.Fatalf("%s=%q was accepted", tc.key, tc.val)
			}
		})
	}
}

func TestLoadConfigRejectsDuplicateListeners(t *testing.T) {
	clearTerminatorEnv(t)
	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_ENABLED", "false")
	t.Setenv("PRIVACY_TERMINATOR_GRPC_LISTEN", ":8079")
	if _, err := loadConfig(); err == nil {
		t.Fatal("duplicate REST/gRPC listeners were accepted")
	}
}

func TestLoadConfigRejectsNonPositiveSkew(t *testing.T) {
	clearTerminatorEnv(t)
	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY", testIdentityKey)
	t.Setenv("PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS", "0")
	if _, err := loadConfig(); err == nil {
		t.Fatal("zero identity skew was accepted")
	}
}

func TestParseListenerStackDefaults(t *testing.T) {
	stack, err := tlsutil.ParseListenerStack("")
	if err != nil {
		t.Fatalf("empty stack: %v", err)
	}
	if !stack.HasTLCP() {
		t.Fatal("default business stack must speak TLCP")
	}
	ops, err := tlsutil.ParseListenerStack("both")
	if err != nil {
		t.Fatalf("both: %v", err)
	}
	if !ops.HasTLCP() || !ops.HasTLS() {
		t.Fatal("both stack must accept TLCP and TLS")
	}
}
