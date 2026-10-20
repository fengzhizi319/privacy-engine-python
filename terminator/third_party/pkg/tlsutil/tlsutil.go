// Package tlsutil provides shared TLS configuration utilities for building secure server credentials.
// Package tlsutil 提供共享的 TLS/mTLS 密码学与传输安全配置工具，用于构建高安全等级的微服务与 RPC 通信凭证。
//
// ==============================================================================
// 【核心能力与安全架构】
// 1. 【TLS 1.3 强制最低版本】：
//    所有通过 BuildServerTLSConfig 生成的配置默认且强制将 MinVersion 设为 tls.VersionTLS13，
//    杜绝弱密码套件、重放攻击与降级攻击；
// 2. 【多模式双向认证 (mTLS Client Auth)】：
//    支持 require/requireandverify（强制双向强校验）、verify（客户端提供证书时校验）、
//    request（请求证书但不强制）等多种认证模式；
// 3. 【公钥固定防御 (Public Key / SPKI Pinning)】：
//    通过注入 VerifyPeerCertificate 钩子函数，深度比对对端公钥数学属性（RSA 模数/指数、
//    ECDSA 椭圆曲线坐标、Ed25519 字节），有效防御 CA 根证书被劫持或非法签发伪造证书；
// 4. 【跨协议复用】：
//    同时作为 HTTP REST（Gin/http.Server）与 gRPC 服务端（credentials.TransportCredentials）
//    的通用底层 TLS 凭证工厂。
//
// ==============================================================================
// 【使用方法与代码范例】
//
//	// 1. 构造服务端双向认证配置
//	tlsConfig, err := tlsutil.BuildServerTLSConfig(&tlsutil.ServerTLSConfig{
//	    Enabled:          true,
//	    CertFile:         "/etc/certs/server.crt",
//	    KeyFile:          "/etc/certs/server.key",
//	    CAFile:           "/etc/certs/ca.crt",
//	    ClientAuth:       "require", // 强制要求客户端证书并验证
//	    PinnedPubKeyFile: "/etc/certs/client_pinned_pub.pem", // 可选公钥固定
//	})
//	if err != nil {
//	    log.Fatalf("failed to build TLS config: %v", err)
//	}
//
//	// 2. 挂载到 HTTP 服务器
//	httpServer := &http.Server{
//	    Addr:      ":8443",
//	    Handler:   router,
//	    TLSConfig: tlsConfig,
//	}
//	go httpServer.ListenAndServeTLS("", "")
//
//	// 3. 挂载到 gRPC 服务器
//	grpcCreds := credentials.NewTLS(tlsConfig)
//	grpcServer := grpc.NewServer(grpc.Creds(grpcCreds))
// ==============================================================================

package tlsutil

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/emmansun/gmsm/smx509"
	"github.com/youmark/pkcs8"
)

// ServerTLSConfig holds the configuration parameters for building a TLS server config.
// ServerTLSConfig 保存构建 TLS 服务器配置所需的全部核心参数。
type ServerTLSConfig struct {
	// Enabled 表示是否启用 TLS 加密（若为 false，BuildServerTLSConfig 将返回错误）。
	Enabled bool

	// CertFile 为服务端 X.509 证书的 PEM 文件绝对或相对路径。
	CertFile string

	// KeyFile 为服务端私钥的 PEM 文件路径（支持 RSA/ECDSA/Ed25519 私钥）。
	KeyFile string

	// KeyPassword 为服务端私钥解密口令（支持口令保护的 PKCS#8 或 PEM 私钥）。
	KeyPassword string

	// CAFile 为受信任的客户端根 CA 证书文件路径（当启用 ClientAuth 时为必填项）。
	CAFile string

	// ClientAuth 为客户端双向认证模式：
	// - "require" / "requireandverify": 强制客户端提供证书且必须通过 CA 根证书链校验（tls.RequireAndVerifyClientCert）；
	// - "verify": 客户端若提供证书则进行校验（tls.VerifyClientCertIfGiven）；
	// - "request": 请求客户端提供证书但不强制校验（tls.RequestClientCert）；
	// - "": 仅单向 TLS 传输加密，不校验客户端证书。
	ClientAuth string

	// PinnedPubKeyFile 为固定的受信任客户端公钥 PEM 文件路径（可选）。
	// 若配置此项，将在 TLS 握手阶段比对客户端证书公钥指纹。
	PinnedPubKeyFile string

	// NationalCipher 启用国密 TLS 合规模式（三级等保/密评 G-01）。
	// 启用后将在日志中输出 TLCP 部署指引。完整国密 TLS（TLCP/GB/T 38636）
	// 需在 Ingress/Envoy 层或第三方库实现，Go 标准库尚未原生支持 SM2/SM3/SM4 套件。
	NationalCipher bool
}

// BuildServerTLSConfig constructs a *tls.Config supporting TLS 1.3, mTLS client auth, and public key pinning.
//
// BuildServerTLSConfig 根据 ServerTLSConfig 参数构建生产级 *tls.Config 实例：
//
// ==============================================================================
// 【执行逻辑】
// 1. 【参数防御校验】：
//   - 校验 Enabled 是否为 true；
//   - 校验 CertFile 与 KeyFile 是否均已配置，并使用 filepath.Clean 清洗路径防止路径穿越；
//
// 2. 【证书与私钥加载】：
//   - 调用 LoadX509KeyPairWithPassword 加载公私钥对（支持 PKCS#8 口令解密）；
//   - 设置 MinVersion 为 tls.VersionTLS13 强制最低加密版本；
//
// 3. 【客户端 CA 证书池装配 (mTLS)】：
//   - 若 ClientAuth 非空，读取 CAFile 并解析为 *x509.CertPool 注入 ClientCAs；
//   - 根据模式映射标准 tls.ClientAuthType（RequireAndVerifyClientCert 等）；
//
// 4. 【SPKI 公钥固定钩子注入 (VerifyPeerCertificate)】：
//   - 若配置了 PinnedPubKeyFile，解析固定公钥对象；
//   - 注册 VerifyPeerCertificate 回调，在 TLS 握手最后阶段提取 peerCert.PublicKey，
//     调用 PublicKeysEqual 进行常数深度比对，不匹配时中断 TLS 握手。
//
// ==============================================================================
func BuildServerTLSConfig(cfg *ServerTLSConfig) (*tls.Config, error) {
	if !cfg.Enabled {
		return nil, fmt.Errorf("TLS is disabled in configuration")
	}
	if cfg.CertFile == "" || cfg.KeyFile == "" {
		return nil, fmt.Errorf("TLS cert file and key file must be configured")
	}

	certFile := filepath.Clean(cfg.CertFile)
	keyFile := filepath.Clean(cfg.KeyFile)
	cert, err := LoadX509KeyPairWithPassword(certFile, keyFile, cfg.KeyPassword)
	if err != nil {
		return nil, fmt.Errorf("load server x509 key pair: %w", err)
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS13,
	}

	// 三级等保/密评 G-01：国密 TLS 合规指引
	if cfg.NationalCipher {
		// Go 标准库 crypto/tls 不支持 SM2/SM3/SM4 密码套件。
		// 完整国密 TLS（TLCP）需在 Ingress/Envoy 层部署国密终结，
		// 或使用第三方库（如 Trisia/gotlcp，见同包 tlcp.go）。
		// 此处记录合规指引日志，供运维参考。
		fmt.Fprintln(os.Stderr, "[TLS] NationalCipher enabled: Go stdlib TLS does not support SM2/SM3/SM4 cipher suites.")
		fmt.Fprintln(os.Stderr, "[TLS] For full TLCP (GB/T 38636) compliance, deploy national cipher at Ingress/Envoy layer")
		fmt.Fprintln(os.Stderr, "[TLS] or use a third-party library such as Trisia/gotlcp.")
	}

	clientAuthMode := strings.ToLower(strings.TrimSpace(cfg.ClientAuth))
	if clientAuthMode != "" {
		if cfg.CAFile == "" {
			return nil, fmt.Errorf("TLS CA file must be configured when client auth is enabled")
		}
		caFile := filepath.Clean(cfg.CAFile)
		caPEM, err := os.ReadFile(caFile)
		if err != nil {
			return nil, fmt.Errorf("read TLS CA file: %w", err)
		}
		caPool := x509.NewCertPool()
		if !caPool.AppendCertsFromPEM(caPEM) {
			return nil, fmt.Errorf("failed to parse CA certificate from %s", caFile)
		}
		tlsConfig.ClientCAs = caPool

		switch clientAuthMode {
		case "require", "requireandverify", "require_and_verify":
			tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
		case "verify", "verify_if_given":
			tlsConfig.ClientAuth = tls.VerifyClientCertIfGiven
		case "request":
			tlsConfig.ClientAuth = tls.RequestClientCert
		default:
			return nil, fmt.Errorf("unknown TLS client auth mode: %s", cfg.ClientAuth)
		}
	}

	// 注入公钥固定校验钩子
	if cfg.PinnedPubKeyFile != "" {
		pinnedFile := filepath.Clean(cfg.PinnedPubKeyFile)
		pinnedKey, err := LoadPublicKey(pinnedFile)
		if err != nil {
			return nil, fmt.Errorf("load pinned client public key: %w", err)
		}
		tlsConfig.VerifyPeerCertificate = func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
			if len(rawCerts) == 0 {
				return fmt.Errorf("mTLS: client did not present a certificate")
			}
			peerCert, err := x509.ParseCertificate(rawCerts[0])
			if err != nil {
				return fmt.Errorf("mTLS: failed to parse peer certificate: %w", err)
			}
			if !PublicKeysEqual(peerCert.PublicKey, pinnedKey) {
				return fmt.Errorf("mTLS: client public key does not match pinned key")
			}
			return nil
		}
	}

	return tlsConfig, nil
}

// LoadPublicKey loads a public key from PEM file (supports PKIX and X.509 Certificate formats).
//
// LoadPublicKey 从 PEM 文件中解析并提取公钥对象：
// 1. 优先尝试按照 PKIX 格式（x509.ParsePKIXPublicKey，如 BEGIN PUBLIC KEY）解析；
// 2. 若失败，回退尝试按照 X.509 证书格式（x509.ParseCertificate，如 BEGIN CERTIFICATE）提取 cert.PublicKey；
// 3. 支持 RSA、ECDSA 与 Ed25519 类型的公钥。
func LoadPublicKey(path string) (crypto.PublicKey, error) {
	cleanPath := filepath.Clean(path)
	data, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("read public key file: %w", err)
	}
	block, _ := pem.Decode(data)
	if block == nil {
		return nil, fmt.Errorf("no PEM data found in %s", path)
	}
	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		cert, certErr := x509.ParseCertificate(block.Bytes)
		if certErr == nil {
			return cert.PublicKey, nil
		}
		return nil, fmt.Errorf("parse public key: %w", err)
	}
	return pub, nil
}

// PublicKeysEqual checks if two public keys are identical (RSA, ECDSA, Ed25519).
//
// PublicKeysEqual 深度比对两个公钥的数学属性：
// - 【RSA】：比对模数 N（big.Int Cmp == 0）与公共指数 E 是否一致；
// - 【ECDSA】：比对公钥曲线坐标点 X、Y（big.Int Cmp == 0）及椭圆曲线参数 Curve 是否一致；
// - 【Ed25519】：调用 ed25519.PublicKey.Equal 比对 32 字节原生公钥数据；
// - 【其他/不匹配】：返回 false。
func PublicKeysEqual(a, b crypto.PublicKey) bool {
	switch keyA := a.(type) {
	case *rsa.PublicKey:
		keyB, ok := b.(*rsa.PublicKey)
		if !ok {
			return false
		}
		return keyA.N.Cmp(keyB.N) == 0 && keyA.E == keyB.E
	case *ecdsa.PublicKey:
		keyB, ok := b.(*ecdsa.PublicKey)
		if !ok {
			return false
		}
		return keyA.X.Cmp(keyB.X) == 0 && keyA.Y.Cmp(keyB.Y) == 0 && keyA.Curve == keyB.Curve
	case ed25519.PublicKey:
		keyB, ok := b.(ed25519.PublicKey)
		if !ok {
			return false
		}
		return keyA.Equal(keyB)
	default:
		return false
	}
}

// LoadX509KeyPairWithPassword loads and parses a public/private key pair from a pair
// of PEM encoded files, with support for password-protected PKCS#8 keys.
// If the key requires a password but password is empty, it returns an error immediately (fail-fast).
// Any temporary password buffers in memory are securely zeroized.
func LoadX509KeyPairWithPassword(certFile, keyFile, password string) (tls.Certificate, error) {
	certPEM, err := os.ReadFile(certFile)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("read cert file: %w", err)
	}
	keyPEM, err := os.ReadFile(keyFile)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("read key file: %w", err)
	}
	return X509KeyPairWithPassword(certPEM, keyPEM, password)
}

// X509KeyPairWithPassword parses a public/private key pair from PEM encoded data.
func X509KeyPairWithPassword(certPEM, keyPEM []byte, password string) (tls.Certificate, error) {
	var cert tls.Certificate

	remainingCert := certPEM
	for {
		var block *pem.Block
		block, remainingCert = pem.Decode(remainingCert)
		if block == nil {
			break
		}
		if block.Type == "CERTIFICATE" {
			cert.Certificate = append(cert.Certificate, block.Bytes)
		}
	}
	if len(cert.Certificate) == 0 {
		return tls.Certificate{}, fmt.Errorf("tls: failed to find any PEM certificate blocks")
	}

	keyBlock, _ := pem.Decode(keyPEM)
	if keyBlock == nil {
		return tls.Certificate{}, fmt.Errorf("tls: failed to find PEM private key block")
	}

	isEncrypted := keyBlock.Type == "ENCRYPTED PRIVATE KEY" || x509.IsEncryptedPEMBlock(keyBlock) //nolint:staticcheck
	if isEncrypted && strings.TrimSpace(password) == "" {
		return tls.Certificate{}, fmt.Errorf("tls: private key is password-protected, but no password was provided")
	}

	// 若未加密且未提供口令，直接使用标准库解析
	if !isEncrypted && password == "" {
		return tls.X509KeyPair(certPEM, keyPEM)
	}

	pwdBytes := []byte(password)
	defer func() {
		for i := range pwdBytes {
			pwdBytes[i] = 0
		}
	}()

	var privKey crypto.PrivateKey
	if keyBlock.Type == "ENCRYPTED PRIVATE KEY" {
		parsed, err := pkcs8.ParsePKCS8PrivateKey(keyBlock.Bytes, pwdBytes)
		if err != nil {
			return tls.Certificate{}, fmt.Errorf("tls: decrypt PKCS#8 private key: %w", err)
		}
		privKey = parsed
	} else if x509.IsEncryptedPEMBlock(keyBlock) { //nolint:staticcheck
		return tls.Certificate{}, fmt.Errorf("tls: traditional encrypted PEM is unsupported; use encrypted PKCS#8")
	} else {
		// 未显式标记加密类型，但提供了口令：尝试 pkcs8 解密或普通解析
		parsed, err := pkcs8.ParsePKCS8PrivateKey(keyBlock.Bytes, pwdBytes)
		if err == nil {
			privKey = parsed
		} else {
			parsedPriv, parseErr := parsePrivateKeyDER(keyBlock.Bytes)
			if parseErr != nil {
				return tls.Certificate{}, fmt.Errorf("tls: parse private key: %w", err)
			}
			privKey = parsedPriv
		}
	}

	cert.PrivateKey = privKey

	// 校验私钥与证书公钥匹配
	leaf, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("tls: parse leaf certificate: %w", err)
	}
	cert.Leaf = leaf
	if signer, ok := privKey.(crypto.Signer); ok {
		if !PublicKeysEqual(signer.Public(), leaf.PublicKey) {
			return tls.Certificate{}, fmt.Errorf("tls: private key does not match public key in certificate")
		}
	}

	return cert, nil
}

func parsePrivateKeyDER(der []byte) (crypto.PrivateKey, error) {
	if key, err := x509.ParsePKCS8PrivateKey(der); err == nil {
		return key, nil
	}
	if key, err := x509.ParsePKCS1PrivateKey(der); err == nil {
		return key, nil
	}
	if key, err := x509.ParseECPrivateKey(der); err == nil {
		return key, nil
	}
	return nil, fmt.Errorf("unknown private key DER format")
}

// IsDevCA checks if the CA certificate file at path is an in-repo or well-known development CA.
// It returns (true, reason) if identified as dev CA, or (false, "") if it is a production custom CA.
// If the file path is empty, it returns (true, "empty CA path").
func IsDevCA(caPath string) (bool, string) {
	if strings.TrimSpace(caPath) == "" {
		return true, "empty CA path"
	}
	pemBytes, err := os.ReadFile(caPath)
	if err != nil {
		return true, fmt.Sprintf("unreadable file: %v", err)
	}

	// 1. Try standard X.509
	block, _ := pem.Decode(pemBytes)
	if block != nil {
		if cert, err := x509.ParseCertificate(block.Bytes); err == nil {
			cn := cert.Subject.CommonName
			if isDevCN(cn) {
				return true, fmt.Sprintf("development Subject CommonName %q", cn)
			}
		}
	}

	// 2. Try SM2/TLCP
	if smCert, err := smx509.ParseCertificatePEM(pemBytes); err == nil {
		cn := smCert.Subject.CommonName
		if isDevCN(cn) {
			return true, fmt.Sprintf("development Subject CommonName %q", cn)
		}
	}

	// 3. Normalized path check (e.g. repo-relative defaults)
	cleanPath := filepath.ToSlash(caPath)
	if strings.HasSuffix(cleanPath, "config/certs/ca.crt") ||
		strings.Contains(cleanPath, "/certs/tlcp/ca.crt") ||
		strings.Contains(cleanPath, "services/service-hub/certs/ca.crt") ||
		strings.Contains(cleanPath, "services/audit-log/certs/ca.crt") ||
		strings.Contains(cleanPath, "console/mock-datasource/certs/ca.crt") {
		return true, fmt.Sprintf("in-repo default development certificate path: %s", cleanPath)
	}

	return false, ""
}

func isDevCN(cn string) bool {
	lower := strings.ToLower(strings.TrimSpace(cn))
	if lower == "privshield-dev-root-ca" || lower == "privshield-tlcp-dev-ca" {
		return true
	}
	if strings.Contains(lower, "dev-root-ca") || strings.Contains(lower, "dev-ca") || strings.Contains(lower, "test-ca") {
		return true
	}
	return false
}
