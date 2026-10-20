package tlsutil

// gRPC 出站传输凭据的共用构造器（国密 TLCP / 标准 TLS / 明文三栈）。
//
// ==============================================================================
// 【为什么抽到 pkg】
// 引擎 gRPC 在 PRIVACY_TLS_NATIONAL_CIPHER 默认开启后是 TLCP + requireandverify，
// 于是每一个拨它的客户端都要会说国密。现有拨号方：
//
//   - 网关 gRPC 反代        services/privacy-engine/internal/gateway
//   - engine-console BFF    console/engine-console/bff-go/internal/agent
//   - hub → datasource      services/service-hub/internal/datasource（对端是模拟
//                           数据源，有意保持标准 TLS/明文，不上国密）
//
// 三处各写一遍「按体系分派 + 成对校验 + 不回落」必然漂移 —— TLCP 改造里
// 已经因为「同一段逻辑抄了三遍、只修了一遍」踩过坑（见 TLCPConnContext 的由来）。
// 故收敛到这里，各服务只负责从自己的环境变量前缀读配置。
//
// 【不回落是硬约束】
// 材料缺失一律返回错误，绝不静默退回明文：回落会让「以为在走国密、
// 实际在裸奔」成为可能，而调用方只看到请求成功。

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// GRPCClientCredsConfig 描述一次 gRPC 出站拨号的传输凭据材料。
//
// 三种体系互斥，优先级：国密 TLCP > 标准 TLS > 明文。
// 该优先级必须与服务端监听侧的分支判定一致（例如
// privacy-engine 的 cmd/privshield-engine/server_grpc.go）——
// 两边不同构时表现为拨号超时或 handshake failure，而非可读的配置错误。
type GRPCClientCredsConfig struct {
	// NationalCipher 为真时走国密 TLCP，优先于 TLSEnabled。
	NationalCipher bool

	// ── 国密 TLCP 客户端材料 ──
	// 客户端侧同为双证书：签名对用于身份认证，加密对用于会话密钥协商。
	// 服务端 ClientAuth=requireandverify 时签名对必填；加密对可选
	// （部分部署只给签名对，握手仍可完成）。
	TLCPCAFile       string
	TLCPSignCertFile string
	TLCPSignKeyFile  string
	TLCPEncCertFile  string
	TLCPEncKeyFile   string
	// TLCPInsecureSkipVerify 仅供本地联调；置真等于放弃对端身份校验。
	TLCPInsecureSkipVerify bool

	// ── 标准 TLS 客户端材料 ──
	TLSEnabled  bool
	TLSCAFile   string
	TLSCertFile string
	TLSKeyFile  string
	// TLSServerName 覆盖校验用主机名（连 127.0.0.1 但证书 SAN 只有 localhost 时）。
	TLSServerName string
	// TLSInsecureSkipVerify 仅供本地联调；生产严禁。
	TLSInsecureSkipVerify bool
	// TLSMinVersion 为 0 时取 tls.VersionTLS12。
	TLSMinVersion uint16
}

// BuildGRPCClientCredentials 按声明的密码体系构造 gRPC 出站传输凭据。
//
// 返回值直接传给 grpc.WithTransportCredentials。
// cfg 为 nil 时返回明文凭据（调用方明确不需要传输加密的场景）。
func BuildGRPCClientCredentials(cfg *GRPCClientCredsConfig) (credentials.TransportCredentials, error) {
	if cfg == nil {
		return insecure.NewCredentials(), nil
	}

	switch {
	case cfg.NationalCipher:
		// 签名对是必需的：服务端 requireandverify 会在握手期索要客户端证书，
		// 没有签名对无法完成双向认证。这里提前报错，避免留到握手期才暴露。
		if cfg.TLCPSignCertFile == "" || cfg.TLCPSignKeyFile == "" {
			return nil, fmt.Errorf("national cipher enabled but TLCP client sign cert/key not set " +
				"(need both *_TLCP_SIGN_CERT_FILE and *_TLCP_SIGN_KEY_FILE)")
		}
		clientCfg, err := BuildTLCPClientConfig(&TLCPClientConfig{
			CAFile:             cfg.TLCPCAFile,
			SignCertFile:       cfg.TLCPSignCertFile,
			SignKeyFile:        cfg.TLCPSignKeyFile,
			EncCertFile:        cfg.TLCPEncCertFile,
			EncKeyFile:         cfg.TLCPEncKeyFile,
			InsecureSkipVerify: cfg.TLCPInsecureSkipVerify,
		})
		if err != nil {
			return nil, fmt.Errorf("build TLCP client config: %w", err)
		}
		creds, err := NewTLCPClientCredentials(clientCfg)
		if err != nil {
			return nil, fmt.Errorf("build TLCP client credentials: %w", err)
		}
		return creds, nil

	case cfg.TLSEnabled:
		tlsCfg, err := buildStandardClientTLS(cfg)
		if err != nil {
			return nil, err
		}
		return credentials.NewTLS(tlsCfg), nil

	default:
		// 明文：仅当两个开关都显式关闭时才到这里。属于有意选择
		// （本地开发、或 Service Mesh 已在数据面加密），不是缺材料的兜底。
		return insecure.NewCredentials(), nil
	}
}

// buildStandardClientTLS 构造标准 TLS（国际算法）客户端配置。
func buildStandardClientTLS(cfg *GRPCClientCredsConfig) (*tls.Config, error) {
	minVersion := cfg.TLSMinVersion
	if minVersion == 0 {
		minVersion = tls.VersionTLS12
	}
	tlsCfg := &tls.Config{
		MinVersion:         minVersion,
		ServerName:         cfg.TLSServerName,
		InsecureSkipVerify: cfg.TLSInsecureSkipVerify, //nolint:gosec // 由调用方显式开启，默认关闭
	}
	if cfg.TLSCAFile != "" {
		caPEM, err := os.ReadFile(cfg.TLSCAFile)
		if err != nil {
			return nil, fmt.Errorf("read TLS CA file %s: %w", cfg.TLSCAFile, err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caPEM) {
			return nil, fmt.Errorf("failed to parse TLS CA certificate from %s", cfg.TLSCAFile)
		}
		tlsCfg.RootCAs = pool
	}
	// 客户端证书成对配置：只给一半按配置错误处理，而不是当作「不带客户端证书」。
	// 后者会让服务端 require 客户端证书时在握手期才失败，排查成本高得多。
	if (cfg.TLSCertFile == "") != (cfg.TLSKeyFile == "") {
		return nil, fmt.Errorf("TLS client cert and key must be provided together " +
			"(need both *_TLS_CERT_FILE and *_TLS_KEY_FILE)")
	}
	if cfg.TLSCertFile != "" {
		cert, err := tls.LoadX509KeyPair(cfg.TLSCertFile, cfg.TLSKeyFile)
		if err != nil {
			return nil, fmt.Errorf("load TLS client key pair (%s, %s): %w",
				cfg.TLSCertFile, cfg.TLSKeyFile, err)
		}
		tlsCfg.Certificates = []tls.Certificate{cert}
	}
	return tlsCfg, nil
}
