// Package crypto implements SM4 block cipher and envelope encryption.
// SM4 block cipher complies with standard GB/T 32907-2016.
package crypto

import (
	"crypto/cipher"
	"encoding/binary"
	"errors"
	"math/bits"
)

// SM4 算法核心常量（GB/T 32907-2016）。
const (
	// BlockSize SM4 分组长度（字节），即 128-bit。
	BlockSize = 16
	// KeySize SM4 密钥长度（字节），即 128-bit。
	KeySize = 16
	// Rounds SM4 迭代轮数（32 轮非线性变换 + 反序变换）。
	Rounds = 32
)

// sbox 是 SM4 标准定义的 8-bit 非线性置换表（S 盒），提供算法的唯一非线性源。
var sbox = [256]uint8{
	0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
	0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
	0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
	0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
	0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
	0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
	0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
	0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
	0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
	0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
	0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
	0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
	0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
	0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
	0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
	0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
}

// fk 是 SM4 密钥扩展使用的系统参数 FK（4 个 32-bit 常数），用于与明文密钥异或启动迭代。
var fk = [4]uint32{0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc}

// ck 是 SM4 密钥扩展使用的常量参数 CK（32 个 32-bit 常数），每轮使用不同的常量。
var ck = [32]uint32{
	0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
	0x70777e85, 0x8c939a01, 0xa8afb6bd, 0xc4cbd2d9,
	0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
	0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
	0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
	0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
	0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
	0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279,
}

// sm4Cipher 实现 cipher.Block 接口的 SM4 分组密码实例。
type sm4Cipher struct {
	encRoundKeys [Rounds]uint32 // 加密轮密钥（正序 32 个 32-bit 子密钥）
	decRoundKeys [Rounds]uint32 // 解密轮密钥（逆序 32 个 32-bit 子密钥）
}

// NewCipher creates and returns a new cipher.Block implementing SM4.
func NewCipher(key []byte) (cipher.Block, error) {
	if len(key) != KeySize {
		return nil, errors.New("crypto/sm4: invalid key size, must be 16 bytes")
	}

	c := new(sm4Cipher)
	c.expandKey(key)
	return c, nil
}

func (c *sm4Cipher) BlockSize() int {
	return BlockSize
}

func (c *sm4Cipher) Encrypt(dst, src []byte) {
	if len(src) < BlockSize {
		panic("crypto/sm4: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/sm4: output smaller than input")
	}

	c.cryptBlock(dst, src, c.encRoundKeys)
}

func (c *sm4Cipher) Decrypt(dst, src []byte) {
	if len(src) < BlockSize {
		panic("crypto/sm4: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/sm4: output smaller than input")
	}

	c.cryptBlock(dst, src, c.decRoundKeys)
}

// cryptBlock 执行 SM4 单块 128-bit 加解密（32 轮复合变换 + 反序输出）。
// 算法流程（GB/T 32907-2016 §7）：
// 1. 将 16 字节明文拆为 4 个 32-bit 字 (X0, X1, X2, X3)；
// 2. 32 轮迭代：每轮对 Xi 执行 T 变换（S 盒置换 τ + 线性变换 L），结果异或到 Xi；
// 3. 输出时反序排列 (X3, X2, X1, X0)。
func (c *sm4Cipher) cryptBlock(dst, src []byte, rk [Rounds]uint32) {
	x0 := binary.BigEndian.Uint32(src[0:4])
	x1 := binary.BigEndian.Uint32(src[4:8])
	x2 := binary.BigEndian.Uint32(src[8:12])
	x3 := binary.BigEndian.Uint32(src[12:16])

	// 32 轮复合变换（每 4 轮一组展开，减少循环开销）
	for i := 0; i < Rounds; i += 4 {
		x0 ^= sm4T(x1 ^ x2 ^ x3 ^ rk[i+0])
		x1 ^= sm4T(x2 ^ x3 ^ x0 ^ rk[i+1])
		x2 ^= sm4T(x3 ^ x0 ^ x1 ^ rk[i+2])
		x3 ^= sm4T(x0 ^ x1 ^ x2 ^ rk[i+3])
	}

	// 反序变换（R32 变换）：输出时交换字序 (X3, X2, X1, X0)
	binary.BigEndian.PutUint32(dst[0:4], x3)
	binary.BigEndian.PutUint32(dst[4:8], x2)
	binary.BigEndian.PutUint32(dst[8:12], x1)
	binary.BigEndian.PutUint32(dst[12:16], x0)
}

// expandKey 从 128-bit 明文密钥生成 32 个轮密钥。
// 算法流程（GB/T 32907-2016 §8）：
// 1. 将密钥 MK 拆为 4 个 32-bit 字 (MK0, MK1, MK2, MK3)，与系统参数 FK 异或得 (K0, K1, K2, K3)；
// 2. 32 轮迭代：Ki+4 = Ki ⊕ T'(Ki+1 ⊕ Ki+2 ⊕ Ki+3 ⊕ CKi)，其中 T' 使用密钥专用线性变换 L'；
// 3. 加密轮密钥正序存储，解密轮密钥逆序存储（共享同一组子密钥）。
func (c *sm4Cipher) expandKey(key []byte) {
	mk0 := binary.BigEndian.Uint32(key[0:4])
	mk1 := binary.BigEndian.Uint32(key[4:8])
	mk2 := binary.BigEndian.Uint32(key[8:12])
	mk3 := binary.BigEndian.Uint32(key[12:16])

	k0 := mk0 ^ fk[0]
	k1 := mk1 ^ fk[1]
	k2 := mk2 ^ fk[2]
	k3 := mk3 ^ fk[3]

	for i := 0; i < Rounds; i++ {
		k4 := k0 ^ sm4KeyT(k1^k2^k3^ck[i])
		c.encRoundKeys[i] = k4
		c.decRoundKeys[Rounds-1-i] = k4 // 解密轮密钥为加密的逆序
		k0, k1, k2, k3 = k1, k2, k3, k4
	}
}

// sm4Tau 是 SM4 非线性置换 τ（S 盒逐字节替换），将 32-bit 字的每个字节经 sbox 映射。
func sm4Tau(a uint32) uint32 {
	return uint32(sbox[byte(a>>24)])<<24 |
		uint32(sbox[byte(a>>16)])<<16 |
		uint32(sbox[byte(a>>8)])<<8 |
		uint32(sbox[byte(a)])
}

// sm4L 是 SM4 数据加密用的线性变换 L：L(B) = B ⊕ (B<<<2) ⊕ (B<<<10) ⊕ (B<<<18) ⊕ (B<<<24)。
func sm4L(b uint32) uint32 {
	return b ^ bits.RotateLeft32(b, 2) ^ bits.RotateLeft32(b, 10) ^ bits.RotateLeft32(b, 18) ^ bits.RotateLeft32(b, 24)
}

// sm4KeyL 是 SM4 密钥扩展用的线性变换 L'：L'(B) = B ⊕ (B<<<13) ⊕ (B<<<23)。
func sm4KeyL(b uint32) uint32 {
	return b ^ bits.RotateLeft32(b, 13) ^ bits.RotateLeft32(b, 23)
}

// sm4T 是合成 T 变换（用于数据加密）：T(A) = L(τ(A))。
func sm4T(x uint32) uint32 {
	return sm4L(sm4Tau(x))
}

// sm4KeyT 是合成 T' 变换（用于密钥扩展）：T'(A) = L'(τ(A))。
func sm4KeyT(x uint32) uint32 {
	return sm4KeyL(sm4Tau(x))
}
