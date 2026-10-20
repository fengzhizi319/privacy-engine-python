// TLCP 国密客户端出站传输装配（GM/T 0024）。
//
// tlcp.go 提供的是服务端双证书配置；本文件补齐对称的**客户端**能力：
//   - BuildTLCPClientConfig：由显式参数构建 *tlcp.Config（CA 信任池 + 可选 SM2 双证书客户端身份）；
//   - TLCPClientDialer / SchemeAwareClientDialer：http.Transport.DialTLSContext 装配，
//     使 net/http 客户端可以对 tlcp:// 端点执行国密握手，并与 https 端点混部。
//
// 机制与策略分离：本文件不硬编码任何环境变量前缀，由调用方（如 pkg/auth 的
// <CALLER>_<TARGET>_TLCP_* 出站变量）读取后以显式参数传入。
//
// 本文件是 TLCP 客户端配置的唯一事实源：pkg/agent/transport.go 的 NewTLCPMTLSConfig
// 已委托 BuildTLCPClientConfig，改动校验逻辑只需维护本处。

package tlsutil

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"os"

	"gitee.com/Trisia/gotlcp/tlcp"
	"github.com/emmansun/gmsm/smx509"
)

// TLCPClientConfig holds the parameters for building a TLCP (national cipher) client config.
// TLCP 客户端配置参数：CA 信任池 + 可选 SM2 签名/加密双证书客户端身份（双向国密 mTLS）。
type TLCPClientConfig struct {
	// CAFile 服务端 SM2 根 CA 证书 PEM 路径（校验上游 TLCP 服务端证书）。
	// 为空且 InsecureSkipVerify 为 false 时回落 gotlcp 默认信任（系统 CA）。
	CAFile string

	// InsecureSkipVerify 跳过服务端证书校验（仅开发/演练）。
	InsecureSkipVerify bool

	// SignCertFile / SignKeyFile 客户端 SM2 签名证书 PEM 路径（双向认证必填）。
	SignCertFile string
	SignKeyFile  string

	// EncCertFile / EncKeyFile 客户端 SM2 加密证书 PEM 路径（可选；配置后可协商
	// 前向安全的 ECDHE_SM4_GCM_SM3，仅签名证书时协商 ECC_SM4_GCM_SM3）。
	EncCertFile string
	EncKeyFile  string
}

// BuildTLCPClientConfig 构造国密 TLCP 客户端配置。
//
// 证书槽位语义与 BuildTLCPConfig（服务端）完全对称：签名槽位必须声明
// KeyUsageDigitalSignature，加密槽位必须声明 KeyUsageKeyEncipherment/DataEncipherment/
// KeyAgreement；半对配置（只给 cert 不给 key 或反之）一律构造期报错。
// 未配置任何客户端证书时返回仅含信任材料的单向 TLCP 配置。
func BuildTLCPClientConfig(cfg *TLCPClientConfig) (*tlcp.Config, error) {
	if cfg == nil {
		return nil, fmt.Errorf("TLCP client: config is nil")
	}

	config := &tlcp.Config{
		InsecureSkipVerify: cfg.InsecureSkipVerify,
		// 与服务端同口径的 GCM AEAD 套件白名单，杜绝 SM1/SHA1/CBC 弱套件协商。
		CipherSuites: []uint16{
			tlcp.ECDHE_SM4_GCM_SM3,
			tlcp.ECC_SM4_GCM_SM3,
		},
	}

	// 先校验证书对完整性再读取信任材料：半对/槽位错误是配置错误，应优先于
	// CA 文件问题被报告，避免错误归因混乱。
	signCertFile, signKeyFile := cfg.SignCertFile, cfg.SignKeyFile
	encCertFile, encKeyFile := cfg.EncCertFile, cfg.EncKeyFile
	if (signCertFile == "") != (signKeyFile == "") {
		return nil, fmt.Errorf("TLCP client: sign cert file and sign key file must be configured together")
	}
	if (encCertFile == "") != (encKeyFile == "") {
		return nil, fmt.Errorf("TLCP client: enc cert file and enc key file must be configured together")
	}
	if encCertFile != "" && signCertFile == "" {
		return nil, fmt.Errorf("TLCP client: enc certificate requires a sign certificate (dual-cert client identity)")
	}

	if cfg.CAFile != "" {
		caPEM, err := os.ReadFile(cfg.CAFile)
		if err != nil {
			return nil, fmt.Errorf("TLCP client: read CA file: %w", err)
		}
		caPool := smx509.NewCertPool()
		if !caPool.AppendCertsFromPEM(caPEM) {
			return nil, fmt.Errorf("TLCP client: failed to parse CA certificate from %s", cfg.CAFile)
		}
		config.RootCAs = caPool
	}

	if signCertFile != "" {
		signCertPEM, err := os.ReadFile(signCertFile)
		if err != nil {
			return nil, fmt.Errorf("TLCP client: read sign cert: %w", err)
		}
		signKeyPEM, err := os.ReadFile(signKeyFile)
		if err != nil {
			return nil, fmt.Errorf("TLCP client: read sign key: %w", err)
		}
		signCert, signKey, err := loadSM2CertFromPEM(signCertPEM, signKeyPEM)
		if err != nil {
			return nil, fmt.Errorf("TLCP client: load sign cert/key: %w", err)
		}
		if signCert.KeyUsage&smx509.KeyUsageDigitalSignature == 0 {
			return nil, fmt.Errorf("TLCP client: sign certificate %s lacks KeyUsageDigitalSignature (keyUsage=%d)", signCertFile, signCert.KeyUsage)
		}
		config.Certificates = []tlcp.Certificate{{
			Certificate: [][]byte{signCert.Raw},
			PrivateKey:  signKey,
			Leaf:        signCert,
		}}

		if encCertFile != "" {
			encCertPEM, err := os.ReadFile(encCertFile)
			if err != nil {
				return nil, fmt.Errorf("TLCP client: read enc cert: %w", err)
			}
			encKeyPEM, err := os.ReadFile(encKeyFile)
			if err != nil {
				return nil, fmt.Errorf("TLCP client: read enc key: %w", err)
			}
			encCert, encKey, err := loadSM2CertFromPEM(encCertPEM, encKeyPEM)
			if err != nil {
				return nil, fmt.Errorf("TLCP client: load enc cert/key: %w", err)
			}
			if encCert.KeyUsage&(smx509.KeyUsageKeyEncipherment|smx509.KeyUsageDataEncipherment|smx509.KeyUsageKeyAgreement) == 0 {
				return nil, fmt.Errorf("TLCP client: enc certificate %s lacks KeyUsageKeyEncipherment/KeyUsageDataEncipherment/KeyUsageKeyAgreement (keyUsage=%d)", encCertFile, encCert.KeyUsage)
			}
			// Certificates 顺序即槽位顺序：[签名证书, 加密证书]（gotlcp 约定）。
			config.Certificates = append(config.Certificates, tlcp.Certificate{
				Certificate: [][]byte{encCert.Raw},
				PrivateKey:  encKey,
				Leaf:        encCert,
			})
		}
	}

	return config, nil
}

// TLCPClientDialer 构造 http.Transport.DialTLSContext：对每次拨号执行 GM/T 0024 国密握手。
// ServerName 缺省时以拨号地址的主机名填充（逐连接 Clone，避免污染共享配置）。
func TLCPClientDialer(cfg *tlcp.Config) func(ctx context.Context, network, addr string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		d := net.Dialer{}
		conn, err := d.DialContext(ctx, network, addr)
		if err != nil {
			return nil, err
		}
		cc := cfg.Clone()
		if cc.ServerName == "" {
			host, _, splitErr := net.SplitHostPort(addr)
			if splitErr == nil {
				cc.ServerName = host
			}
		}
		tlsConn := tlcp.Client(conn, cc)
		if err := tlsConn.HandshakeContext(ctx); err != nil {
			conn.Close()
			return nil, fmt.Errorf("TLCP handshake with %s: %w", addr, err)
		}
		return tlsConn, nil
	}
}

// SchemeAwareClientDialer 构造按对端地址分派的 DialTLSContext：
// hostSchemes 中标记为 "tlcp" 的对端走 GM/T 0024 国密握手，其余 https 对端走
// 标准 TLS 握手（tlsCfg 逐连接 Clone）。混部集群（https + tlcp 多节点）下同一条
// 传输上两种协议并存，互不用错套件。
func SchemeAwareClientDialer(tlsCfg *tls.Config, tlcpCfg *tlcp.Config, hostSchemes map[string]string) func(ctx context.Context, network, addr string) (net.Conn, error) {
	tlcpDial := TLCPClientDialer(tlcpCfg)
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		if hostSchemes[addr] == "tlcp" {
			return tlcpDial(ctx, network, addr)
		}
		d := net.Dialer{}
		conn, err := d.DialContext(ctx, network, addr)
		if err != nil {
			return nil, err
		}
		cc := &tls.Config{}
		if tlsCfg != nil {
			cc = tlsCfg.Clone()
		}
		if cc.ServerName == "" {
			if host, _, splitErr := net.SplitHostPort(addr); splitErr == nil {
				cc.ServerName = host
			}
		}
		tlsConn := tls.Client(conn, cc)
		if err := tlsConn.HandshakeContext(ctx); err != nil {
			conn.Close()
			return nil, fmt.Errorf("TLS handshake with %s: %w", addr, err)
		}
		return tlsConn, nil
	}
}
