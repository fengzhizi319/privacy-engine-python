// Package crypto implements SM2 elliptic curve public key cryptography.
// SM2 非对称密码算法符合国家标准 GB/T 32918-2016 与 GM/T 0003-2012。
//
// ==============================================================================
// 【密码学标准与设计背景】
// SM2 算法基于椭圆曲线 sm2p256v1（256-bit 素域），提供：
//   - 数字签名：基于 SM3 杂凑的 ECDSA 变体，含用户标识 ZA 预处理（GM/T 0009）；
//   - 公钥加密：ECIES 风格，KDF 由 SM3 驱动，密文格式为 C1‖C3‖C2（新国标）；
//   - 统一抽象接口：定义 SM2 / SignerVerifier 接口，底层直连 emmansun/gmsm 高性能常数时间实现。
//
// ==============================================================================
package crypto

import (
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/subtle"
	"encoding/asn1"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"os"
	"strings"
	"sync"

	"github.com/emmansun/gmsm/sm2"
	"github.com/emmansun/gmsm/smx509"
)

// SM2 算法核心常量。
const (
	// SM2KeySize SM2 密钥与标量的字节长度（256-bit = 32 字节）。
	SM2KeySize = 32

	// SM2UncompressedLen SM2 非压缩公钥的字节长度（0x04 + 32 字节 X + 32 字节 Y = 65）。
	SM2UncompressedLen = 65

	// SM2SigPartLen SM2 签名分量 r / s 的固定字节长度（各 32 字节）。
	SM2SigPartLen = 32
)

// sm2DefaultUID GB/T 32918.2 规定的默认用户标识（16 字节 ASCII "1234567812345678"）。
var sm2DefaultUID = []byte("1234567812345678")

// ---------------------------------------------------------------------------
// 密钥结构体
// ---------------------------------------------------------------------------

// SM2PublicKey 表示 SM2 公钥，即曲线上的一个点 (X, Y)。
type SM2PublicKey struct {
	X, Y *big.Int
}

// ToECDSA 将 SM2PublicKey 转换为标准库/gmsm 兼容的 *ecdsa.PublicKey。
func (pub *SM2PublicKey) ToECDSA() *ecdsa.PublicKey {
	if pub == nil {
		return nil
	}
	return &ecdsa.PublicKey{
		Curve: sm2.P256(),
		X:     pub.X,
		Y:     pub.Y,
	}
}

// SM2PrivateKey 表示 SM2 私钥，包含标量 D 与对应公钥 PublicKey。
type SM2PrivateKey struct {
	D         *big.Int     // 私钥标量，满足 1 ≤ D ≤ n−2
	PublicKey SM2PublicKey // 对应的公钥 Q = D·G
}

// ToGMSM 将 SM2PrivateKey 转换为 emmansun/gmsm 兼容的 *sm2.PrivateKey。
func (priv *SM2PrivateKey) ToGMSM() *sm2.PrivateKey {
	if priv == nil || priv.D == nil {
		return nil
	}
	k, err := sm2.NewPrivateKeyFromInt(priv.D)
	if err != nil {
		return nil
	}
	return k
}

// FromGMSMPublicKey 从 ecdsa.PublicKey 构造 SM2PublicKey。
func FromGMSMPublicKey(pub *ecdsa.PublicKey) *SM2PublicKey {
	if pub == nil {
		return nil
	}
	return &SM2PublicKey{
		X: pub.X,
		Y: pub.Y,
	}
}

// FromGMSMPrivateKey 从 emmansun/gmsm 的 PrivateKey 构造 SM2PrivateKey。
func FromGMSMPrivateKey(priv *sm2.PrivateKey) *SM2PrivateKey {
	if priv == nil {
		return nil
	}
	return &SM2PrivateKey{
		D: priv.D,
		PublicKey: SM2PublicKey{
			X: priv.PublicKey.X,
			Y: priv.PublicKey.Y,
		},
	}
}

// ---------------------------------------------------------------------------
// 底层 emmansun/gmsm 驱动实现
// ---------------------------------------------------------------------------

type gmsmSM2 struct{}

var _ SM2 = (*gmsmSM2)(nil)

// GenerateKey 生成强随机 SM2 密钥对。
func (g *gmsmSM2) GenerateKey() (*SM2PrivateKey, error) {
	priv, err := sm2.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("crypto/sm2: generate key failed: %w", err)
	}
	return FromGMSMPrivateKey(priv), nil
}

// SignMessage 对原始明文消息签名（自动执行 ZA + SM3）。
func (g *gmsmSM2) SignMessage(priv *SM2PrivateKey, msg []byte) (r, s *big.Int, err error) {
	if priv == nil {
		return nil, nil, errors.New("crypto/sm2: nil private key")
	}
	gmsmPriv := priv.ToGMSM()
	if gmsmPriv == nil {
		return nil, nil, errors.New("crypto/sm2: invalid private key")
	}
	hash, err := sm2.CalculateSM2Hash(&gmsmPriv.PublicKey, msg, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("crypto/sm2: calculate sm2 hash failed: %w", err)
	}
	return g.Sign(priv, hash)
}

// Sign 对预杂凑值 hash 进行标量签名，返回 (r, s)。
// 内部使用具备抗随机数故障注入（Hedged RNG）的 sm2.SignASN1，并解析为 (r, s)。
func (g *gmsmSM2) Sign(priv *SM2PrivateKey, hash []byte) (r, s *big.Int, err error) {
	der, err := g.SignASN1(priv, hash)
	if err != nil {
		return nil, nil, err
	}
	var sig struct {
		R, S *big.Int
	}
	if _, err := asn1.Unmarshal(der, &sig); err != nil {
		return nil, nil, fmt.Errorf("crypto/sm2: unmarshal asn1 signature: %w", err)
	}
	return sig.R, sig.S, nil
}

// SignASN1 对预杂凑值 hash 进行签名，返回标准 ASN.1 DER 编码签名。
// 遵循 GB/T 32918.2-2016，具备抗 RNG 偏差的 Hedged 签名安全性。
func (g *gmsmSM2) SignASN1(priv *SM2PrivateKey, hash []byte) ([]byte, error) {
	if priv == nil {
		return nil, errors.New("crypto/sm2: nil private key")
	}
	gmsmPriv := priv.ToGMSM()
	if gmsmPriv == nil {
		return nil, errors.New("crypto/sm2: invalid private key")
	}
	return sm2.SignASN1(rand.Reader, gmsmPriv, hash, nil)
}

// Verify 对预杂凑值 hash 上的签名 (r, s) 进行验签。
// 内部打包为标准 ASN.1 DER 格式调用常量时间 sm2.VerifyASN1。
func (g *gmsmSM2) Verify(pub *SM2PublicKey, hash []byte, r, s *big.Int) error {
	if pub == nil {
		return errors.New("crypto/sm2: nil public key")
	}
	if r == nil || s == nil {
		return errors.New("crypto/sm2: nil signature component")
	}
	der, err := asn1.Marshal(struct {
		R, S *big.Int
	}{r, s})
	if err != nil {
		return fmt.Errorf("crypto/sm2: marshal asn1 signature: %w", err)
	}
	if !g.VerifyASN1(pub, hash, der) {
		return errors.New("crypto/sm2: signature verification failed")
	}
	return nil
}

// VerifyASN1 对预杂凑值 hash 上的 ASN.1 DER 编码签名进行验签。
func (g *gmsmSM2) VerifyASN1(pub *SM2PublicKey, hash, sig []byte) bool {
	if pub == nil || len(sig) == 0 {
		return false
	}
	gmsmPub := pub.ToECDSA()
	return sm2.VerifyASN1(gmsmPub, hash, sig)
}

// VerifyMessage 对原始明文消息进行验签（自动执行 ZA + SM3）。
func (g *gmsmSM2) VerifyMessage(pub *SM2PublicKey, msg []byte, r, s *big.Int) bool {
	if pub == nil || r == nil || s == nil {
		return false
	}
	gmsmPub := pub.ToECDSA()
	hash, err := sm2.CalculateSM2Hash(gmsmPub, msg, nil)
	if err != nil {
		return false
	}
	return g.Verify(pub, hash, r, s) == nil
}

// Encrypt 使用公钥对明文加密，输出 C1‖C3‖C2 格式密文。
func (g *gmsmSM2) Encrypt(pub *SM2PublicKey, plaintext []byte) ([]byte, error) {
	if pub == nil {
		return nil, errors.New("crypto/sm2: nil public key")
	}
	if len(plaintext) == 0 {
		return nil, nil
	}
	gmsmPub := pub.ToECDSA()
	opts := sm2.NewPlainEncrypterOpts(sm2.MarshalUncompressed, sm2.C1C3C2)
	ct, err := sm2.Encrypt(rand.Reader, gmsmPub, plaintext, opts)
	if err != nil {
		return nil, fmt.Errorf("crypto/sm2: encrypt failed: %w", err)
	}
	return ct, nil
}

// Decrypt 使用私钥对 C1‖C3‖C2 格式密文解密。
func (g *gmsmSM2) Decrypt(priv *SM2PrivateKey, ciphertext []byte) ([]byte, error) {
	if priv == nil {
		return nil, errors.New("crypto/sm2: nil private key")
	}
	if len(ciphertext) == 0 {
		return nil, nil
	}
	gmsmPriv := priv.ToGMSM()
	if gmsmPriv == nil {
		return nil, errors.New("crypto/sm2: invalid private key")
	}
	pt, err := sm2.Decrypt(gmsmPriv, ciphertext)
	if err != nil {
		return nil, fmt.Errorf("crypto/sm2: decrypt failed: %w", err)
	}
	return pt, nil
}

// ComputeZA 计算指定 UID 的用户身份杂凑前缀 ZA。
func (g *gmsmSM2) ComputeZA(pub *SM2PublicKey, uid []byte) []byte {
	if pub == nil {
		return nil
	}
	if len(uid) == 0 {
		uid = sm2DefaultUID
	}
	gmsmPub := pub.ToECDSA()
	za, err := sm2.CalculateZA(gmsmPub, uid)
	if err != nil {
		return nil
	}
	return za
}

// ---------------------------------------------------------------------------
// 默认引擎单例管理
// ---------------------------------------------------------------------------

var (
	defaultSM2Mu sync.RWMutex
	defaultSM2   SM2 = &gmsmSM2{}
)

// DefaultSM2 返回当前全局配置的 SM2 原语引擎实例（默认为 emmansun/gmsm 实现）。
func DefaultSM2() SM2 {
	defaultSM2Mu.RLock()
	defer defaultSM2Mu.RUnlock()
	return defaultSM2
}

// SetDefaultSM2 允许注入自定义的 SM2Provider 实例（如测试 Mock 或硬件 HSM 驱动）。
func SetDefaultSM2(engine SM2) {
	if engine == nil {
		return
	}
	defaultSM2Mu.Lock()
	defer defaultSM2Mu.Unlock()
	defaultSM2 = engine
}

// ---------------------------------------------------------------------------
// 包级别便捷函数（向后完全兼容现有代码）
// ---------------------------------------------------------------------------

// GenerateKey 使用默认 SM2 引擎生成强随机密钥对。
func GenerateKey() (*SM2PrivateKey, error) {
	return DefaultSM2().GenerateKey()
}

// SignMessage 使用默认 SM2 引擎对原始消息直接签名。
func SignMessage(priv *SM2PrivateKey, msg []byte) (r, s *big.Int, err error) {
	return DefaultSM2().SignMessage(priv, msg)
}

// Sign 使用默认 SM2 引擎对预杂凑值签名，返回 (r, s)。
func Sign(priv *SM2PrivateKey, hash []byte) (r, s *big.Int, err error) {
	return DefaultSM2().Sign(priv, hash)
}

// SignASN1 使用默认 SM2 引擎对预杂凑值签名，返回标准 ASN.1 DER 编码签名。
func SignASN1(priv *SM2PrivateKey, hash []byte) ([]byte, error) {
	return DefaultSM2().SignASN1(priv, hash)
}

// Verify 使用默认 SM2 引擎对预杂凑值验签。
func Verify(pub *SM2PublicKey, hash []byte, r, s *big.Int) error {
	return DefaultSM2().Verify(pub, hash, r, s)
}

// VerifyASN1 使用默认 SM2 引擎对预杂凑值上的 ASN.1 DER 签名进行验签。
func VerifyASN1(pub *SM2PublicKey, hash, sig []byte) bool {
	return DefaultSM2().VerifyASN1(pub, hash, sig)
}

// VerifyMessage 使用默认 SM2 引擎对原始消息直接验签。
func VerifyMessage(pub *SM2PublicKey, msg []byte, r, s *big.Int) bool {
	return DefaultSM2().VerifyMessage(pub, msg, r, s)
}

// Encrypt 使用默认 SM2 引擎对明文进行加密（输出 C1‖C3‖C2 格式）。
func Encrypt(pub *SM2PublicKey, plaintext []byte) ([]byte, error) {
	return DefaultSM2().Encrypt(pub, plaintext)
}

// Decrypt 使用默认 SM2 引擎对 C1‖C3‖C2 格式密文进行解密。
func Decrypt(priv *SM2PrivateKey, ciphertext []byte) ([]byte, error) {
	return DefaultSM2().Decrypt(priv, ciphertext)
}

// ComputeZA 使用默认 SM2 引擎计算用户标识杂凑前缀 ZA。
func ComputeZA(pub *SM2PublicKey, uid []byte) []byte {
	return DefaultSM2().ComputeZA(pub, uid)
}

// ---------------------------------------------------------------------------
// 审计存证 SM2 签名器/验签器包装（三级等保/密评 G-10：审计不可否认性）
// ---------------------------------------------------------------------------

// SM2SignerVerifier 将 SM2 私钥签名与公钥验签封装为单对象。
type SM2SignerVerifier struct {
	priv   *SM2PrivateKey
	pub    *SM2PublicKey
	engine SM2
}

var _ SignerVerifier = (*SM2SignerVerifier)(nil)

// NewSM2SignerVerifier 从已加载的私钥/公钥构建签名器（默认使用全局 DefaultSM2 引擎）。
func NewSM2SignerVerifier(priv *SM2PrivateKey, pub *SM2PublicKey) *SM2SignerVerifier {
	return NewSM2SignerVerifierWithEngine(priv, pub, DefaultSM2())
}

// NewSM2SignerVerifierWithEngine 使用指定的 SM2 引擎构建签名器。
func NewSM2SignerVerifierWithEngine(priv *SM2PrivateKey, pub *SM2PublicKey, engine SM2) *SM2SignerVerifier {
	if engine == nil {
		engine = DefaultSM2()
	}
	sv := &SM2SignerVerifier{priv: priv, pub: pub, engine: engine}
	if priv != nil && pub == nil {
		sv.pub = &priv.PublicKey
	}
	return sv
}

// NewSM2SignerVerifierFromHex 从 hex 字符串加载 SM2 私钥与公钥。
func NewSM2SignerVerifierFromHex(privHex, pubHex string) (*SM2SignerVerifier, error) {
	var priv *SM2PrivateKey
	if strings.TrimSpace(privHex) != "" {
		dBytes, err := hex.DecodeString(strings.TrimSpace(privHex))
		if err != nil {
			return nil, fmt.Errorf("crypto/sm2: invalid private key hex: %w", err)
		}
		if len(dBytes) != SM2KeySize {
			return nil, fmt.Errorf("crypto/sm2: private key must be %d bytes, got %d", SM2KeySize, len(dBytes))
		}
		gmsmPriv, err := sm2.NewPrivateKey(dBytes)
		if err != nil {
			return nil, fmt.Errorf("crypto/sm2: invalid private key: %w", err)
		}
		priv = FromGMSMPrivateKey(gmsmPriv)
	}

	var pub *SM2PublicKey
	if strings.TrimSpace(pubHex) != "" {
		pBytes, err := hex.DecodeString(strings.TrimSpace(pubHex))
		if err != nil {
			return nil, fmt.Errorf("crypto/sm2: invalid public key hex: %w", err)
		}
		switch len(pBytes) {
		case SM2UncompressedLen:
			if pBytes[0] != 0x04 {
				return nil, fmt.Errorf("crypto/sm2: uncompressed public key must start with 0x04")
			}
			ecdsaPub, err := sm2.ParseUncompressedPublicKey(pBytes)
			if err != nil {
				return nil, fmt.Errorf("crypto/sm2: parse public key: %w", err)
			}
			pub = FromGMSMPublicKey(ecdsaPub)
		case SM2KeySize * 2:
			fullBytes := append([]byte{0x04}, pBytes...)
			ecdsaPub, err := sm2.ParseUncompressedPublicKey(fullBytes)
			if err != nil {
				return nil, fmt.Errorf("crypto/sm2: parse 64-byte raw public key: %w", err)
			}
			pub = FromGMSMPublicKey(ecdsaPub)
		default:
			return nil, fmt.Errorf("crypto/sm2: public key must be %d or %d bytes, got %d", SM2UncompressedLen, SM2KeySize*2, len(pBytes))
		}
	}

	if priv == nil && pub == nil {
		return nil, errors.New("crypto/sm2: at least one of private key or public key must be provided")
	}

	return NewSM2SignerVerifier(priv, pub), nil
}

// ParseSM2PrivateKeyFromPEM 从 PEM 字节流解析 SM2 私钥（支持 PKCS#8 与 SEC1 EC PRIVATE KEY 格式）。
func ParseSM2PrivateKeyFromPEM(pemBytes []byte) (*SM2PrivateKey, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, errors.New("crypto/sm2: no valid PEM block found in private key data")
	}
	switch block.Type {
	case "PRIVATE KEY": // PKCS#8 格式
		k, err := smx509.ParsePKCS8PrivateKey(block.Bytes)
		if err != nil {
			return nil, fmt.Errorf("crypto/sm2: parse PKCS#8 private key: %w", err)
		}
		switch priv := k.(type) {
		case *sm2.PrivateKey:
			return FromGMSMPrivateKey(priv), nil
		case *ecdsa.PrivateKey:
			gmsmPriv, err := sm2.NewPrivateKeyFromInt(priv.D)
			if err != nil {
				return nil, fmt.Errorf("crypto/sm2: convert ecdsa private key to sm2: %w", err)
			}
			return FromGMSMPrivateKey(gmsmPriv), nil
		default:
			return nil, fmt.Errorf("crypto/sm2: unsupported PKCS#8 key type %T", k)
		}
	case "EC PRIVATE KEY": // SEC1 格式（如 openssl ecparam -genkey 产出）
		ecKey, err := smx509.ParseECPrivateKey(block.Bytes)
		if err != nil {
			return nil, fmt.Errorf("crypto/sm2: parse SEC1 EC private key: %w", err)
		}
		gmsmPriv, err := sm2.NewPrivateKeyFromInt(ecKey.D)
		if err != nil {
			return nil, fmt.Errorf("crypto/sm2: convert SEC1 private key to sm2: %w", err)
		}
		return FromGMSMPrivateKey(gmsmPriv), nil
	default:
		return nil, fmt.Errorf("crypto/sm2: unsupported PEM block type for private key: %q", block.Type)
	}
}

// ParseSM2PublicKeyFromPEM 从 PEM 字节流解析 SM2 公钥（支持 PKIX PUBLIC KEY 格式）。
func ParseSM2PublicKeyFromPEM(pemBytes []byte) (*SM2PublicKey, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, errors.New("crypto/sm2: no valid PEM block found in public key data")
	}
	pub, err := smx509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("crypto/sm2: parse PKIX public key: %w", err)
	}
	switch p := pub.(type) {
	case *ecdsa.PublicKey:
		return FromGMSMPublicKey(p), nil
	default:
		return nil, fmt.Errorf("crypto/sm2: unsupported PKIX public key type %T", pub)
	}
}

// NewSM2SignerVerifierFromPEM 从 PEM 字节切片构建 SM2 签名/验签器。
func NewSM2SignerVerifierFromPEM(privPEM, pubPEM []byte) (*SM2SignerVerifier, error) {
	var priv *SM2PrivateKey
	var pub *SM2PublicKey
	var err error

	if len(privPEM) > 0 {
		if priv, err = ParseSM2PrivateKeyFromPEM(privPEM); err != nil {
			return nil, err
		}
	}
	if len(pubPEM) > 0 {
		if pub, err = ParseSM2PublicKeyFromPEM(pubPEM); err != nil {
			return nil, err
		}
	}
	if priv == nil && pub == nil {
		return nil, errors.New("crypto/sm2: at least one of private key or public key must be provided in PEM")
	}
	return NewSM2SignerVerifier(priv, pub), nil
}

// LoadSM2SignerVerifier 智能统一加载器：支持传入文件路径、内联 PEM 字符串或十六进制 Hex 编码。
//
// 识别规则：
// 1. 若输入指向一个存在的磁盘文件，自动读取其内容；
// 2. 若内容包含 "-----BEGIN "，按标准 PEM（PKCS#8/SEC1/PKIX）解析；
// 3. 否则按 16 进制 Hex 字符串解析（支持 32B 私钥 hex，64B/65B 公钥 hex）。
func LoadSM2SignerVerifier(privInput, pubInput string) (*SM2SignerVerifier, error) {
	readContent := func(input string) ([]byte, bool) {
		trimmed := strings.TrimSpace(input)
		if trimmed == "" {
			return nil, false
		}
		if data, err := os.ReadFile(trimmed); err == nil {
			return data, true
		}
		return []byte(trimmed), false
	}

	privBytes, _ := readContent(privInput)
	pubBytes, _ := readContent(pubInput)

	isPEM := func(data []byte) bool {
		return strings.Contains(string(data), "-----BEGIN ")
	}

	var priv *SM2PrivateKey
	var pub *SM2PublicKey
	var err error

	if len(privBytes) > 0 {
		if isPEM(privBytes) {
			if priv, err = ParseSM2PrivateKeyFromPEM(privBytes); err != nil {
				return nil, fmt.Errorf("parse private key PEM: %w", err)
			}
		} else {
			// Try hex
			sv, err := NewSM2SignerVerifierFromHex(string(privBytes), "")
			if err != nil {
				return nil, fmt.Errorf("parse private key hex: %w", err)
			}
			priv = sv.priv
		}
	}

	if len(pubBytes) > 0 {
		if isPEM(pubBytes) {
			if pub, err = ParseSM2PublicKeyFromPEM(pubBytes); err != nil {
				return nil, fmt.Errorf("parse public key PEM: %w", err)
			}
		} else {
			// Try hex
			sv, err := NewSM2SignerVerifierFromHex("", string(pubBytes))
			if err != nil {
				return nil, fmt.Errorf("parse public key hex: %w", err)
			}
			pub = sv.pub
		}
	}

	if priv == nil && pub == nil {
		return nil, errors.New("crypto/sm2: neither private key nor public key could be loaded")
	}

	return NewSM2SignerVerifier(priv, pub), nil
}

// Sign 使用 SM2 私钥对 data 签名，返回固定 64 字节签名值（r ‖ s）。
func (sv *SM2SignerVerifier) Sign(data []byte) ([]byte, error) {
	if sv.priv == nil {
		return nil, errors.New("crypto/sm2: signer has no private key")
	}
	r, s, err := sv.engine.SignMessage(sv.priv, data)
	if err != nil {
		return nil, err
	}
	sig := make([]byte, SM2KeySize*2)
	copy(sig[:SM2KeySize], sm2PadScalar(r))
	copy(sig[SM2KeySize:], sm2PadScalar(s))
	return sig, nil
}

// Verify 使用 SM2 公钥验证 data 上的 64 字节签名值（r ‖ s）。
func (sv *SM2SignerVerifier) Verify(data, signature []byte) bool {
	if sv.pub == nil {
		return false
	}
	if len(signature) != SM2KeySize*2 {
		return false
	}
	r := new(big.Int).SetBytes(signature[:SM2KeySize])
	s := new(big.Int).SetBytes(signature[SM2KeySize:])
	return sv.engine.VerifyMessage(sv.pub, data, r, s)
}

// SignASN1 使用 SM2 私钥对 data 进行签名，返回标准 ASN.1 DER 编码签名。
func (sv *SM2SignerVerifier) SignASN1(data []byte) ([]byte, error) {
	if sv.priv == nil {
		return nil, errors.New("crypto/sm2: signer has no private key")
	}
	gmsmPriv := sv.priv.ToGMSM()
	if gmsmPriv == nil {
		return nil, errors.New("crypto/sm2: invalid private key")
	}
	hash, err := sm2.CalculateSM2Hash(&gmsmPriv.PublicKey, data, nil)
	if err != nil {
		return nil, fmt.Errorf("crypto/sm2: calculate sm2 hash: %w", err)
	}
	return sv.engine.SignASN1(sv.priv, hash)
}

// VerifyASN1 使用 SM2 公钥验证 data 上的 ASN.1 DER 编码签名。
func (sv *SM2SignerVerifier) VerifyASN1(data, signature []byte) bool {
	if sv.pub == nil || len(signature) == 0 {
		return false
	}
	gmsmPub := sv.pub.ToECDSA()
	hash, err := sm2.CalculateSM2Hash(gmsmPub, data, nil)
	if err != nil {
		return false
	}
	return sv.engine.VerifyASN1(sv.pub, hash, signature)
}

// sm2PadScalar 将大整数标量左填充至 32 字节。
func sm2PadScalar(v *big.Int) []byte {
	b := v.Bytes()
	if len(b) >= SM2KeySize {
		return b[len(b)-SM2KeySize:]
	}
	pad := make([]byte, SM2KeySize)
	copy(pad[SM2KeySize-len(b):], b)
	return pad
}

// sm2ConstantTimeEqual 以常量时间比较两个等长字节切片。
func sm2ConstantTimeEqual(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}
