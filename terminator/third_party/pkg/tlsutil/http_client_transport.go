// 出站 HTTP 传输的共用加装器（与 BuildGRPCClientCredentials 同一套三栈分派口径）。
//
// ==============================================================================
// 【为什么需要】
// BuildGRPCClientCredentials 收敛了 gRPC 拨号的三栈分派（国密 TLCP > 标准 TLS > 明文），
// 但 HTTP 侧长期没有对应物：网关 REST 反代把回源硬编码为明文 http:// + 共享明文连接池
// （pkg/gateway/balancer.go），而 PRIVACY_GATEWAY_BACKEND_TLCP_* 那套出站凭据只喂给了
// gRPC 代理 —— 后端业务口「没有明文档」（三档均 mTLS）时，REST 链路恒 502 EOF，
// 环境变量看起来配了、实际没人读。
//
// net/http 侧的国密握手机制与 gRPC 不同：gotlcp 不是 crypto/tls，
// 挂 TLSClientConfig 不生效，必须换 DialTLSContext（同 pkg/agent transport 的口径）。
//
// 【不回落是硬约束】与 gRPC 分支完全一致：声明了密码体系而材料缺失即报错，
// 绝不静默退回明文 —— 「以为在走国密、实际裸奔」比链路不通更危险。
package tlsutil

import (
	"fmt"
	"net/http"
)

// AugmentHTTPClientTransport 在既有 http.Transport 上按声明的密码体系加装握手层，
// 返回回源 URL 是否必须使用 https scheme。连接池等传输参数保持调用方原样。
//
// 明文分支（两个开关都显式关闭）不改动 tr，返回 (false, nil)。
// cfg 为 nil 等同明文分支。
func AugmentHTTPClientTransport(tr *http.Transport, cfg *GRPCClientCredsConfig) (bool, error) {
	if cfg == nil {
		return false, nil
	}

	switch {
	case cfg.NationalCipher:
		if cfg.TLCPSignCertFile == "" || cfg.TLCPSignKeyFile == "" {
			return false, fmt.Errorf("national cipher enabled but TLCP client sign cert/key not set " +
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
			return false, fmt.Errorf("build TLCP client config: %w", err)
		}
		// gotlcp 不是 crypto/tls：国密握手只能经 DialTLSContext 装配，
		// 挂 TLSClientConfig 会被标准握手静默顶替。
		tr.DialTLSContext = TLCPClientDialer(clientCfg)
		return true, nil

	case cfg.TLSEnabled:
		tlsCfg, err := buildStandardClientTLS(cfg)
		if err != nil {
			return false, err
		}
		tr.TLSClientConfig = tlsCfg
		return true, nil

	default:
		// 明文：与 gRPC 侧同口径，仅在两个开关都显式关闭时到达。属于有意选择
		// （本地开发、对端是测试替身、或 Service Mesh 已在数据面加密），不是缺材料的兜底。
		return false, nil
	}
}
