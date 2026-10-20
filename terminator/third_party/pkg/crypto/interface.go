// Package crypto provides unified cryptographic interfaces and primitives.
// Package crypto 提供国密密码学算法抽象接口与核心实现。
package crypto

import (
	"math/big"
)

// SM2 定义国密 SM2 椭圆曲线公钥密码体系的统一抽象接口。
//
// 符合国家标准 GB/T 32918-2016《信息安全技术 SM2 椭圆曲线公钥密码算法》与 GM/T 0003-2012。
// 抽象出密钥生命周期管理、数字签名、验签、公钥加密、私钥解密以及用户标识杂凑前缀 ZA 计算能力。
// 底层默认直连工业级、常量时间汇编硬件加速的 emmansun/gmsm 引擎，同时支持未来平滑接入加密机/HSM 驱动。
type SM2 interface {
	// GenerateKey 生成强随机 SM2 密钥对（256-bit 椭圆曲线 sm2p256v1）。
	GenerateKey() (*SM2PrivateKey, error)

	// SignMessage 使用 SM2 私钥对原始明文消息直接签名。
	// 内部自动执行 GM/T 0009 规定的默认 UID ("1234567812345678") 计算 ZA 与 SM3 杂凑，返回 (r, s)。
	SignMessage(priv *SM2PrivateKey, msg []byte) (r, s *big.Int, err error)

	// Sign 使用 SM2 私钥对已预先计算好的 32 字节杂凑值 hash 进行标量签名，返回 (r, s)。
	Sign(priv *SM2PrivateKey, hash []byte) (r, s *big.Int, err error)

	// SignASN1 使用 SM2 私钥对已预先计算好的 32 字节杂凑值 hash 进行签名，返回标准 ASN.1 DER 编码签名。
	SignASN1(priv *SM2PrivateKey, hash []byte) ([]byte, error)

	// Verify 使用 SM2 公钥对 32 字节预杂凑值 hash 上的签名 (r, s) 进行验签。
	// 验证通过返回 nil，若验签失败或参数非法则返回明确错误。
	Verify(pub *SM2PublicKey, hash []byte, r, s *big.Int) error

	// VerifyASN1 使用 SM2 公钥对预杂凑值 hash 上的 ASN.1 DER 编码签名进行验签。
	VerifyASN1(pub *SM2PublicKey, hash, sig []byte) bool

	// VerifyMessage 使用 SM2 公钥对原始明文消息及 (r, s) 进行验签（内部自动执行 ZA + SM3）。
	VerifyMessage(pub *SM2PublicKey, msg []byte, r, s *big.Int) bool

	// Encrypt 使用 SM2 公钥对明文进行非对称加密。
	// 输出符合新国标 GM/T 0003 的 C1‖C3‖C2 裸字节切片（C1 为 65 字节非压缩公钥 0x04 ‖ X ‖ Y）。
	Encrypt(pub *SM2PublicKey, plaintext []byte) ([]byte, error)

	// Decrypt 使用 SM2 私钥对 C1‖C3‖C2 密文字节流进行解密并核验 C3 杂凑。
	Decrypt(priv *SM2PrivateKey, ciphertext []byte) ([]byte, error)

	// ComputeZA 计算指定用户标识（UID）与公钥绑定的 32 字节杂凑前缀 ZA。
	// 若 uid 为空，遵循国密规范回退到默认 16 字节 UID "1234567812345678"。
	ComputeZA(pub *SM2PublicKey, uid []byte) []byte
}

// SignerVerifier 定义通用的数字签名与验签接口。
// 用于三级等保与审计存证不可否认性场景（对齐 pkg/store 审计存证注册规范）。
type SignerVerifier interface {
	// Sign 对输入字节切片进行数字签名，返回序列化后的签名字节流。
	Sign(data []byte) ([]byte, error)

	// Verify 验证输入字节切片上的数字签名是否合法有效。
	Verify(data, signature []byte) bool
}
