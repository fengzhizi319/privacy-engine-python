// Package crypto implements SM3 cryptographic hash and SM4 block cipher.
// SM3 cryptographic hash complies with Chinese National Standard GB/T 32905-2016 and GM/T 0004-2012.
// 底层接入业界主流、经过充分密码学验证的第三方商密库 github.com/emmansun/gmsm/sm3，
// 具备 CPU 硬件指令加速（AVX2 / ARM NEON）与高吞吐性能。
package crypto

import (
	"crypto/hmac"
	"encoding/hex"
	"hash"

	gmsm3 "github.com/emmansun/gmsm/sm3"
)

// SM3 算法核心常量（GB/T 32905-2016 / GM/T 0004-2012）。
const (
	// SM3BlockSize SM3 分组长度（字节），每 512-bit 为一个压缩函数输入块。
	SM3BlockSize = gmsm3.BlockSize
	// SM3Size SM3 摘要输出长度（字节），即 256-bit。
	SM3Size = gmsm3.Size
)

// NewSM3 returns a new hash.Hash computing the SM3 checksum.
// 基于第三方商用密码库 github.com/emmansun/gmsm/sm3 实现。
func NewSM3() hash.Hash {
	return gmsm3.New()
}

// SumSM3 returns the SM3 checksum of the data.
func SumSM3(data []byte) [SM3Size]byte {
	return gmsm3.Sum(data)
}

// SumSM3Hex returns the hex-encoded SM3 checksum of data.
func SumSM3Hex(data []byte) string {
	sum := SumSM3(data)
	return hex.EncodeToString(sum[:])
}

// HMACSM3 returns the keyed SM3 message authentication code (GM/T 0004-2012 SM3 with the RFC 2104 HMAC construction).
// HMACSM3 返回基于国密 SM3 的密钥化消息认证码。
//
// 【当前无生产调用方】
// 它原本服务于存证哈希链的密钥化摘要，链密钥退场后该用法已移除
// （见 pkg/store/audit_hash.go 文件头第 2 条）。pkg/auth 的 break-glass 签名
// 直接用 hmac.New(sm3.New, ...)，未经本函数。
//
// 保留它是作为通用国密原语，且 pkg/store 的用例仍用它构造「必须被拒绝的
// HMAC 口径摘要」来守那条禁止多口径回落的禁令。
func HMACSM3(key, data []byte) [SM3Size]byte {
	mac := hmac.New(NewSM3, key)
	mac.Write(data)
	sum := mac.Sum(nil)
	var out [SM3Size]byte
	copy(out[:], sum)
	return out
}

// HMACSM3Hex returns the hex-encoded keyed SM3 MAC of data.
func HMACSM3Hex(key, data []byte) string {
	sum := HMACSM3(key, data)
	return hex.EncodeToString(sum[:])
}
