// Package certtest 为测试生成一套自洽的 TLS / TLCP 证书。
//
// ==============================================================================
// 【为什么需要它】
// 多处测试依赖仓库里的 config/certs/（client.key、tlcp/server-sign.key 等），
// 而 `*.key` 按约定被 .gitignore —— **私钥不入库是对的**。
// 后果是：数据使用者本地跑过 gen-certs.sh 之后这些测试一直是绿的，
// 而 CI 的干净克隆里根本没有那些文件，同一个提交必红。
//
// 本轮在三个不同位置各踩了一次同一个坑（service-hub TLS、service-hub TLCP、
// pkg/agent + audit-log）。单点修补挡不住第四次，故把证书生成收敛到这里：
// 测试不再依赖工作区状态，任何机器、任何克隆都能跑。
//
// 【生成什么】
//
//	<dir>/ca.crt                  ECDSA 根 CA（签发下面的 server / client）
//	<dir>/server.crt|.key         服务端证书对（EKU: serverAuth，SAN: localhost/127.0.0.1）
//	<dir>/client.crt|.key         客户端证书对（EKU: clientAuth）
//	<dir>/client.pub              客户端公钥 PEM（供应用层 SPKI Pinning）
//	<dir>/tlcp/ca.crt             SM2 根 CA（签发下面两张）
//	<dir>/tlcp/server-sign.crt|.key  服务端签名证书（KeyUsage: DigitalSignature）
//	<dir>/tlcp/server-enc.crt|.key   服务端加密证书（KeyUsage: KeyAgreement|DataEncipherment）
//	<dir>/tlcp/client-sign.crt|.key  客户端签名证书（双向 TLCP 必需）
//	<dir>/tlcp/client-enc.crt|.key   客户端加密证书
//
// TLCP 双证的 KeyUsage 必须分开：gotlcp 握手会强校验用途，用错证书会被拒绝。
// 这里按与 scripts/dev/tlcp-certgen 相同的口径生成，保证被测加载路径与生产一致。
// ==============================================================================
package certtest

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/emmansun/gmsm/sm2"
	"github.com/emmansun/gmsm/smx509"
)

// Generate 在 dir 下生成整套测试证书；目录不存在会被创建。
//
// 证书有效期 ±1 小时 —— 测试用证书没有长期有效的理由，
// 短有效期还能顺带暴露「把测试证书误用到别处」的情况。
func Generate(dir string) error {
	if err := os.MkdirAll(filepath.Join(dir, "tlcp"), 0o700); err != nil {
		return fmt.Errorf("create cert dir: %w", err)
	}
	if err := generateTLSSet(dir); err != nil {
		return fmt.Errorf("tls set: %w", err)
	}
	if err := generateTLCPSet(filepath.Join(dir, "tlcp")); err != nil {
		return fmt.Errorf("tlcp set: %w", err)
	}
	return nil
}

// generateTLSSet 生成标准 TLS 用的 CA 与客户端证书对。
func generateTLSSet(dir string) error {
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return err
	}
	caTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "privshield-test-ca"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTmpl, caTmpl, &caKey.PublicKey, caKey)
	if err != nil {
		return err
	}
	caCert, err := x509.ParseCertificate(caDER)
	if err != nil {
		return err
	}
	if err := writePEM(filepath.Join(dir, "ca.crt"), "CERTIFICATE", caDER); err != nil {
		return err
	}

	// 客户端叶子证书由上面的 CA 签发 —— 握手往返测试要求这条信任链成立。
	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return err
	}
	leafTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(2),
		Subject:               pkix.Name{CommonName: "privshield-test-client"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		DNSNames:              []string{"localhost"},
		BasicConstraintsValid: true,
	}
	leafDER, err := x509.CreateCertificate(rand.Reader, leafTmpl, caCert, &leafKey.PublicKey, caKey)
	if err != nil {
		return err
	}
	if err := writePEM(filepath.Join(dir, "client.crt"), "CERTIFICATE", leafDER); err != nil {
		return err
	}
	leafKeyDER, err := x509.MarshalECPrivateKey(leafKey)
	if err != nil {
		return err
	}
	if err := writePEM(filepath.Join(dir, "client.key"), "EC PRIVATE KEY", leafKeyDER); err != nil {
		return err
	}

	// client.pub：客户端公钥，供应用层 SPKI Pinning 使用。
	// 部分启动脚本按 {server.crt, server.key, ca.crt, client.pub} 四件齐全
	// 判断「证书链是否完整」，缺一就会自行重新生成 —— 那会绕过本包的生成结果。
	pubDER, err := x509.MarshalPKIXPublicKey(&leafKey.PublicKey)
	if err != nil {
		return err
	}
	if err := writePEM(filepath.Join(dir, "client.pub"), "PUBLIC KEY", pubDER); err != nil {
		return err
	}

	// 服务端证书：mTLS 往返测试里服务端要用它握手，
	// SAN 必须含 localhost 与 127.0.0.1，否则客户端校验主机名失败。
	srvKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return err
	}
	srvTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(3),
		Subject:               pkix.Name{CommonName: "privshield-test-server"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:              []string{"localhost"},
		IPAddresses:           []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")},
		BasicConstraintsValid: true,
	}
	srvDER, err := x509.CreateCertificate(rand.Reader, srvTmpl, caCert, &srvKey.PublicKey, caKey)
	if err != nil {
		return err
	}
	if err := writePEM(filepath.Join(dir, "server.crt"), "CERTIFICATE", srvDER); err != nil {
		return err
	}
	srvKeyDER, err := x509.MarshalECPrivateKey(srvKey)
	if err != nil {
		return err
	}
	return writePEM(filepath.Join(dir, "server.key"), "EC PRIVATE KEY", srvKeyDER)
}

// generateTLCPSet 生成国密 TLCP 用的 SM2 CA 与签名/加密双证。
func generateTLCPSet(dir string) error {
	caKey, err := sm2.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	caTmpl := &smx509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "privshield-test-tlcp-ca"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              smx509.KeyUsageCertSign | smx509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		SignatureAlgorithm:    smx509.SM2WithSM3,
	}
	caDER, err := smx509.CreateCertificate(rand.Reader, caTmpl, caTmpl, caKey.Public(), caKey)
	if err != nil {
		return err
	}
	caCert, err := smx509.ParseCertificate(caDER)
	if err != nil {
		return err
	}
	if err := writePEM(filepath.Join(dir, "ca.crt"), "CERTIFICATE", caDER); err != nil {
		return err
	}

	// TLCP 双证：签名证书与加密证书的 KeyUsage 必须分开，
	// gotlcp 握手会强校验用途 —— 用错证书会被直接拒绝。
	// 同时生成 server 端与 client 端双证书，以完整支持双向 TLCP mTLS 校验。
	for _, spec := range []struct {
		name     string
		usage    smx509.KeyUsage
		extUsage []smx509.ExtKeyUsage
		serial   int64
	}{
		{"server-sign", smx509.KeyUsageDigitalSignature, []smx509.ExtKeyUsage{smx509.ExtKeyUsageServerAuth}, 2},
		{"server-enc", smx509.KeyUsageKeyAgreement | smx509.KeyUsageDataEncipherment, []smx509.ExtKeyUsage{smx509.ExtKeyUsageServerAuth}, 3},
		{"client-sign", smx509.KeyUsageDigitalSignature, []smx509.ExtKeyUsage{smx509.ExtKeyUsageClientAuth}, 4},
		{"client-enc", smx509.KeyUsageKeyAgreement | smx509.KeyUsageDataEncipherment, []smx509.ExtKeyUsage{smx509.ExtKeyUsageClientAuth}, 5},
	} {
		key, kerr := sm2.GenerateKey(rand.Reader)
		if kerr != nil {
			return kerr
		}
		tmpl := &smx509.Certificate{
			SerialNumber:          big.NewInt(spec.serial),
			Subject:               pkix.Name{CommonName: "privshield-test-" + spec.name},
			NotBefore:             time.Now().Add(-time.Hour),
			NotAfter:              time.Now().Add(time.Hour),
			KeyUsage:              spec.usage,
			ExtKeyUsage:           spec.extUsage,
			DNSNames:              []string{"localhost"},
			IPAddresses:           []net.IP{net.ParseIP("127.0.0.1")},
			BasicConstraintsValid: true,
			SignatureAlgorithm:    smx509.SM2WithSM3,
		}
		der, cerr := smx509.CreateCertificate(rand.Reader, tmpl, caCert, key.Public(), caKey)
		if cerr != nil {
			return cerr
		}
		if werr := writePEM(filepath.Join(dir, spec.name+".crt"), "CERTIFICATE", der); werr != nil {
			return werr
		}
		keyDER, merr := smx509.MarshalPKCS8PrivateKey(key)
		if merr != nil {
			return merr
		}
		if werr := writePEM(filepath.Join(dir, spec.name+".key"), "PRIVATE KEY", keyDER); werr != nil {
			return werr
		}
	}
	return nil
}

// writePEM 以 0600 落盘一个 PEM 块。私钥与证书统一用同一权限，
// 避免「证书 0644、私钥 0600」这类不一致在测试里被当成正常。
func writePEM(path, blockType string, der []byte) error {
	return os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: blockType, Bytes: der}), 0o600)
}
