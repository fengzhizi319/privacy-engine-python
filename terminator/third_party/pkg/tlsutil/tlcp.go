// Package tlsutil provides shared TLS configuration utilities for building secure server credentials.
//
// 国密 TLCP (GB/T 38636-2020) 支持：
//  Go 标准库 crypto/tls 不原生支持 SM2/SM3/SM4 密码套件。本文件基于 Trisia/gotlcp
//  提供 TLCP 服务端配置构造，需配套 SM2 签名证书与加密证书。
//
//  TLCP 与 TLS 在证书体系上的差异：
//   - TLCP 需要「签名证书链」与「加密证书链」两套 SM2 证书；
//   - 单向 TLCP 至少配置签名证书；双向 TLCP 还需客户端 SM2 证书。
//   - 生产部署可在 Ingress/Envoy 层做国密 TLS 终结，本文件提供应用层替代方案。

package tlsutil

import (
	"encoding/pem"
	"fmt"
	"net"
	"os"
	"strings"

	"gitee.com/Trisia/gotlcp/tlcp"
	"github.com/emmansun/gmsm/sm2"
	"github.com/emmansun/gmsm/smx509"
)

// TLCPConfig holds the parameters for building a TLCP (national cipher) server config.
// TLCP 服务端配置参数，支持单向与双向国密 TLS。
type TLCPConfig struct {
	// Enabled 是否启用 TLCP。
	Enabled bool

	// SignCertFile 服务端 SM2 签名证书 PEM 路径。
	SignCertFile string

	// SignKeyFile 服务端 SM2 签名证书私钥 PEM 路径。
	SignKeyFile string

	// EncCertFile 服务端 SM2 加密证书 PEM 路径（双向/双证书场景）。
	EncCertFile string

	// EncKeyFile 服务端 SM2 加密证书私钥 PEM 路径。
	EncKeyFile string

	// ClientCAFile 客户端根 CA 证书 PEM 路径（启用双向认证时必填）。
	ClientCAFile string

	// ClientAuth 是否要求客户端证书："require" / "requireandverify" 开启双向。
	ClientAuth string
}

// BuildTLCPConfig 构造国密 TLCP 服务端配置。
// 若未启用或缺少签名证书，返回 nil（调用方应回退到标准 TLS）。
func BuildTLCPConfig(cfg *TLCPConfig) (*tlcp.Config, error) {
	if !cfg.Enabled {
		return nil, nil
	}
	if cfg.SignCertFile == "" || cfg.SignKeyFile == "" {
		return nil, fmt.Errorf("TLCP: sign cert file and sign key file are required")
	}

	signCertPEM, err := os.ReadFile(cfg.SignCertFile)
	if err != nil {
		return nil, fmt.Errorf("TLCP: read sign cert: %w", err)
	}
	signKeyPEM, err := os.ReadFile(cfg.SignKeyFile)
	if err != nil {
		return nil, fmt.Errorf("TLCP: read sign key: %w", err)
	}

	signCert, signKey, err := loadSM2CertFromPEM(signCertPEM, signKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("TLCP: load sign cert/key: %w", err)
	}
	// 签名证书槽位校验：签名位必须声明 KeyUsageDigitalSignature，否则 TLCP 签名用途不成立，启动即报错
	if signCert.KeyUsage&smx509.KeyUsageDigitalSignature == 0 {
		return nil, fmt.Errorf("TLCP: sign certificate %s lacks KeyUsageDigitalSignature (keyUsage=%d)", cfg.SignCertFile, signCert.KeyUsage)
	}

	var encCert *smx509.Certificate
	var encKey *sm2.PrivateKey
	if cfg.EncCertFile != "" && cfg.EncKeyFile != "" {
		encCertPEM, err := os.ReadFile(cfg.EncCertFile)
		if err != nil {
			return nil, fmt.Errorf("TLCP: read enc cert: %w", err)
		}
		encKeyPEM, err := os.ReadFile(cfg.EncKeyFile)
		if err != nil {
			return nil, fmt.Errorf("TLCP: read enc key: %w", err)
		}
		encCert, encKey, err = loadSM2CertFromPEM(encCertPEM, encKeyPEM)
		if err != nil {
			return nil, fmt.Errorf("TLCP: load enc cert/key: %w", err)
		}
		// 加密证书槽位校验：加密位必须声明 KeyUsageKeyEncipherment/DataEncipherment/KeyAgreement，
		// 否则密钥交换/数据加密封装不成立，启动即报错
		if encCert.KeyUsage&(smx509.KeyUsageKeyEncipherment|smx509.KeyUsageDataEncipherment|smx509.KeyUsageKeyAgreement) == 0 {
			return nil, fmt.Errorf("TLCP: enc certificate %s lacks KeyUsageKeyEncipherment/KeyUsageDataEncipherment/KeyUsageKeyAgreement (keyUsage=%d)", cfg.EncCertFile, encCert.KeyUsage)
		}
	}

	certs := []tlcp.Certificate{
		{
			Certificate: [][]byte{signCert.Raw},
			PrivateKey:  signKey,
			Leaf:        signCert,
		},
	}
	if encCert != nil && encKey != nil {
		certs = append(certs, tlcp.Certificate{
			Certificate: [][]byte{encCert.Raw},
			PrivateKey:  encKey,
			Leaf:        encCert,
		})
	}

	config := &tlcp.Config{
		// Certificates 顺序即槽位顺序：[签名证书, 加密证书]（gotlcp 约定）。
		Certificates: certs,
		// 显式密码套件白名单，仅允许 GM/T 0024 的 SM4-GCM 套件（AEAD，无 CBC 填充与
		// HMAC 拆分弱点），杜绝 SM1 / SHA1 / CBC 等遗留弱套件协商。
		// 无客户端证书的客户端将自动降级协商 ECC_SM4_GCM_SM3（SM2 静态密钥交换）；
		// 持有双证书的客户端可协商前向安全的 ECDHE_SM4_GCM_SM3。
		CipherSuites: []uint16{
			tlcp.ECDHE_SM4_GCM_SM3,
			tlcp.ECC_SM4_GCM_SM3,
		},
	}

	// 归一化后再查表：与 ValidateTLCPClientAuthMode 同口径。
	// 两边归一化方式若不一致，"Require" 这类大小写变体会通过启动校验，
	// 却在监听构造期才报 unknown client auth mode。
	clientAuthMode := strings.ToLower(strings.TrimSpace(cfg.ClientAuth))
	if clientAuthMode != "" {
		if cfg.ClientCAFile == "" {
			return nil, fmt.Errorf("TLCP: client CA file is required when client auth is enabled")
		}
		caPEM, err := os.ReadFile(cfg.ClientCAFile)
		if err != nil {
			return nil, fmt.Errorf("TLCP: read client CA file: %w", err)
		}
		caPool := smx509.NewCertPool()
		if !caPool.AppendCertsFromPEM(caPEM) {
			return nil, fmt.Errorf("TLCP: failed to parse client CA certificate")
		}
		config.ClientCAs = caPool
		mode, ok := tlcpClientAuthModes[clientAuthMode]
		if !ok {
			return nil, fmt.Errorf("TLCP: unknown client auth mode: %s", clientAuthMode)
		}
		config.ClientAuth = mode
	}

	return config, nil
}

// IsTLCPEnabled 判断环境是否显式要求启用国密 TLCP。
// 必须由调用方传入要检查的环境变量名（如 "PRIVACY_TLS_NATIONAL_CIPHER"）。
// 若 envKey 为空或未配置，则返回 false。
func IsTLCPEnabled(envKey string) bool {
	if envKey == "" {
		return false
	}
	return getEnvBool(envKey, false)
}

// TLCPConfigFromEnv 根据调用方显式传入的前缀从环境变量构建 TLCPConfig。
// 机制与策略完全分离：基础包自身不硬编码任何特定环境变量前缀，不维护次级兼容兜底。
func TLCPConfigFromEnv(prefix string) *TLCPConfig {
	if prefix == "" {
		return &TLCPConfig{}
	}
	return &TLCPConfig{
		Enabled:      getEnvBool(prefix+"TLS_NATIONAL_CIPHER", false),
		SignCertFile: getEnvString(prefix+"TLCP_SIGN_CERT_FILE", ""),
		SignKeyFile:  getEnvString(prefix+"TLCP_SIGN_KEY_FILE", ""),
		EncCertFile:  getEnvString(prefix+"TLCP_ENC_CERT_FILE", ""),
		EncKeyFile:   getEnvString(prefix+"TLCP_ENC_KEY_FILE", ""),
		ClientCAFile: getEnvString(prefix+"TLCP_CLIENT_CA_FILE", ""),
		ClientAuth:   getEnvString(prefix+"TLCP_CLIENT_AUTH", ""),
	}
}

// NewTLCPListener 创建 TLCP 国密 TLS 监听。
// 调用方可直接替换 net.Listen + tls 路径。
func NewTLCPListener(network, address string, config *tlcp.Config) (net.Listener, error) {
	return tlcp.Listen(network, address, config)
}

// getEnvString 读取环境变量，不存在返回默认值。
func getEnvString(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

// getEnvBool 读取布尔环境变量，支持 true/1/yes/on。
func getEnvBool(key string, defaultValue bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return defaultValue
	}
	switch v {
	case "true", "1", "yes", "on", "TRUE", "True":
		return true
	default:
		return false
	}
}

// loadSM2CertFromPEM 从 PEM 数据解析 SM2 证书与私钥（私钥支持 PKCS#8 格式）。
func loadSM2CertFromPEM(certPEM, keyPEM []byte) (*smx509.Certificate, *sm2.PrivateKey, error) {
	certBlock, _ := pem.Decode(certPEM)
	if certBlock == nil {
		return nil, nil, fmt.Errorf("failed to decode SM2 certificate PEM")
	}
	cert, err := smx509.ParseCertificate(certBlock.Bytes)
	if err != nil {
		return nil, nil, fmt.Errorf("parse SM2 certificate: %w", err)
	}

	keyBlock, _ := pem.Decode(keyPEM)
	if keyBlock == nil {
		return nil, nil, fmt.Errorf("failed to decode SM2 private key PEM")
	}
	keyAny, err := smx509.ParsePKCS8PrivateKey(keyBlock.Bytes)
	if err != nil {
		return nil, nil, fmt.Errorf("parse SM2 private key: %w", err)
	}
	key, ok := keyAny.(*sm2.PrivateKey)
	if !ok {
		return nil, nil, fmt.Errorf("private key is not SM2")
	}
	return cert, key, nil
}

// tlcpClientAuthModes 是 TLCP 客户端证书要求档位的**唯一词表**。
//
// ==============================================================================
// 【为什么要抽成表】
// 此前这份词表只存在于 BuildTLCPServerConfig 的 switch 里，而各服务 config 的
// Validate 各自又写了一份更窄的白名单 —— audit-log 只认 "require"/"verify"。
// 于是 k8s 清单里合法的 "requireandverify" 在启动校验处被拒，Pod 根本起不来；
// 反过来 Validate 放行的值若不在 switch 里，又会在监听构造期才炸。
// 两处词表必须同源，故对外暴露 ValidateTLCPClientAuthMode 供 Validate 复用。
var tlcpClientAuthModes = map[string]tlcp.ClientAuthType{
	"require":            tlcp.RequireAndVerifyClientCert,
	"requireandverify":   tlcp.RequireAndVerifyClientCert,
	"require_and_verify": tlcp.RequireAndVerifyClientCert,
	"verify":             tlcp.VerifyClientCertIfGiven,
	"verify_if_given":    tlcp.VerifyClientCertIfGiven,
	"request":            tlcp.RequestClientCert,
}

// ValidateTLCPClientAuthMode 校验客户端证书档位取值是否为 BuildTLCPServerConfig
// 真正认识的值。空串表示「未指定」，由调用方决定是否允许。
//
// 各服务 config.Validate 应调用本函数，而不是自己再写一份白名单 ——
// 自写白名单与真实解析词表漂移，正是 audit-log 曾经拒绝合法值的原因。
func ValidateTLCPClientAuthMode(mode string) error {
	normalized := strings.ToLower(strings.TrimSpace(mode))
	if normalized == "" {
		return nil
	}
	if _, ok := tlcpClientAuthModes[normalized]; !ok {
		return fmt.Errorf("unknown TLCP client auth mode %q (want one of: request, verify, verify_if_given, require, requireandverify, require_and_verify)", mode)
	}
	return nil
}
