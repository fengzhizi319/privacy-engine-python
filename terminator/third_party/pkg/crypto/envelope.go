// Package crypto provides cryptographic utilities for envelope encryption and data protection.
// Package crypto 提供商用密码算法工具库，实现基于国密 SM4-GCM 的信封加密与敏感数据保护。
//
// ==============================================================================
// 【密码学标准与设计规范】
// 本模块严格对标国家密码管理局与国家标准：
//  - **核心密码算法**：GB/T 32907-2016《信息安全技术 SM4 分组密码算法》（128 位分组长度，128 位密钥长度）；
//  - **工作模式**：GCM（Galois/Counter Mode 伽罗瓦/计数器模式，AEAD 认证加密）；
//  - **密钥派生**：RFC 5869 HKDF（Extract-then-Expand），杂凑函数取 GM/T 0004-2012 SM3，
//    每条记录携带独立 16 字节随机 salt，杜绝「短口令直接哈希截断」式弱派生；
//  - **完整性与抗重放保障**：
//    1. 每次加密均由密码学安全随机源（crypto/rand.Reader）生成独占 16 字节 salt 与 12 字节 Nonce；
//    2. GCM 模式自带 16 字节认证标签（Authentication Tag），且 **信封头（enc:<version>:）参与 AAD**，
//       因此剥离/改写前缀或版本号会直接导致认证失败，不存在「去前缀即降级为明文」或跨版本篡改通道；
//  - **统一落盘格式规范**：
//    `enc:<version>:<Base64( 16字节 salt + 12字节 Nonce + SM4 密文 + 16字节 Tag )>`
// ==============================================================================

package crypto

import (
	"crypto/cipher"
	"crypto/hmac"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	// EncryptedPrefix 是统一国密信封密文的标准规范前缀。
	// 统一密文规范格式：enc:<key_version>:<Base64( 16字节 salt + 12字节 Nonce + SM4-GCM 密文 + 16字节 Tag )>
	EncryptedPrefix = "enc:"

	// DefaultKeyVersion 是在未配置多版本密钥注册表时的默认密钥版本号。
	DefaultKeyVersion = "v1"

	// NonceSize 是国密 SM4-GCM 推荐的标准 12 字节随机数长度（96-bit）。
	NonceSize = 12

	// SaltSize 是 HKDF-SM3 密钥派生所需的逐记录独立 16 字节随机盐长度。
	SaltSize = 16

	// hkdfInfo 将派生密钥严格绑定到 PrivShield SM4-GCM 信封加密用途，防止跨用途密钥复用。
	hkdfInfo = "PrivShield SM4-GCM envelope"
)

var (
	// ErrEmptyKey 表示未配置加密密钥：写入路径坚决不降级为明文，直接报错拒绝落盘。
	ErrEmptyKey = errors.New("crypto: encryption key is not configured")

	// ErrUnencryptedValue 表示输入值不带合法的信封密文前缀（enc:），防止剥离前缀导致降级攻击。
	ErrUnencryptedValue = errors.New("crypto: value is not envelope-encrypted (missing enc: prefix)")

	// ErrInvalidEnvelope 表示信封密文结构非法（如缺少版本号、缺少载荷或长度不足）。
	ErrInvalidEnvelope = errors.New("crypto: invalid envelope format")

	// ErrKeyVersionNotFound 表示请求的密钥版本不存在且无可用回退密钥。
	ErrKeyVersionNotFound = errors.New("crypto: key version not found")
)

// CryptoAuditEvent 记录一次密码操作的审计事件（三级等保 G-13）。
type CryptoAuditEvent struct {
	Operation  string // "sm4_encrypt", "sm4_decrypt"
	Timestamp  int64  // Unix nano
	KeyVersion string // 使用的密钥版本
	InputLen   int    // 输入数据长度（不记录明文内容）
	Success    bool   // 操作是否成功
	Error      string // 错误信息（如有）
}

// CryptoAuditLogger 密码操作审计日志接口（G-13）。
type CryptoAuditLogger interface {
	LogCryptoAudit(event CryptoAuditEvent)
}

var cryptoAuditLogger CryptoAuditLogger

// SetCryptoAuditLogger 注册密码操作审计日志记录器（G-13）。
func SetCryptoAuditLogger(logger CryptoAuditLogger) {
	cryptoAuditLogger = logger
}

func auditCrypto(op, keyVersion string, inputLen int, success bool, err error) {
	if cryptoAuditLogger == nil {
		return
	}
	errMsg := ""
	if err != nil {
		errMsg = err.Error()
	}
	cryptoAuditLogger.LogCryptoAudit(CryptoAuditEvent{
		Operation:  op,
		Timestamp:  time.Now().UnixNano(),
		KeyVersion: keyVersion,
		InputLen:   inputLen,
		Success:    success,
		Error:      errMsg,
	})
}

// KeyVersion 表示一个带版本标识的加密密钥（三级等保 G-08 密钥生命周期管理）。
type KeyVersion struct {
	Version   string // 版本标识（如 "v1", "v2", "20260901"）
	Key       []byte // 密钥材料
	Active    bool   // 是否为当前活跃密钥（写入使用）
	CreatedAt int64  // 创建时间（Unix timestamp）
}

// keyRegistry 管理多版本密钥，支持密钥轮换过渡期。
var (
	keyRegistry   []*KeyVersion
	keyRegistryMu sync.RWMutex
)

// RegisterKeyVersion 注册一个新的密钥版本（G-08 密钥轮换）。
// 注册后该版本可用于解密，若 active=true 则同时作为当前写入的活跃版本。
func RegisterKeyVersion(version string, key []byte, active bool) error {
	vLower := strings.ToLower(strings.TrimSpace(version))
	if vLower == "" || len(key) == 0 {
		return errors.New("crypto: invalid key version or empty key material")
	}

	keyRegistryMu.Lock()
	defer keyRegistryMu.Unlock()
	for _, kv := range keyRegistry {
		if kv.Version == version {
			Zeroize(kv.Key)
			kv.Key = key
			kv.Active = active
			return nil
		}
	}
	keyRegistry = append(keyRegistry, &KeyVersion{
		Version:   version,
		Key:       key,
		Active:    active,
		CreatedAt: time.Now().Unix(),
	})
	if active {
		for _, kv := range keyRegistry {
			if kv.Version != version {
				kv.Active = false
			}
		}
	}
	return nil
}

// RegisterKeyVersionsFromEnv 从环境变量注册多版本密钥（G-08 密钥轮换）。
//
// 环境变量约定：
//   - <prefix>KEY_<VERSION>=<key_material>：注册一个密钥版本（VERSION 即版本标识）
//   - <prefix>ACTIVE_VERSION=<version>：指定当前活跃版本（用于加密写入）
//
// 机制与策略完全分离：基础密码包不硬编码任何具体业务环境变量前缀，调用方显式传入前缀参数。
// 若 prefix 为空或未配置，则返回 0。
func RegisterKeyVersionsFromEnv(prefix string) int {
	if prefix == "" {
		return 0
	}
	activeKey := prefix + "ACTIVE_VERSION"
	keyPrefix := prefix + "KEY_"

	activeVersion := os.Getenv(activeKey)

	count := 0
	for _, env := range os.Environ() {
		kv := strings.SplitN(env, "=", 2)
		if len(kv) != 2 {
			continue
		}
		name, material := kv[0], kv[1]
		if !strings.HasPrefix(name, keyPrefix) || name == activeKey {
			continue
		}
		version := strings.TrimPrefix(name, keyPrefix)
		if version == "" || material == "" {
			continue
		}
		active := strings.EqualFold(version, activeVersion)
		if err := RegisterKeyVersion(version, []byte(material), active); err == nil {
			count++
		}
	}
	return count
}

// ActiveKeyVersion 返回当前活跃密钥版本（用于加密写入）。
func ActiveKeyVersion() *KeyVersion {
	keyRegistryMu.RLock()
	defer keyRegistryMu.RUnlock()
	for _, kv := range keyRegistry {
		if kv.Active {
			return kv
		}
	}
	return nil
}

// LookupKeyVersion 根据版本号查找密钥（用于解密读取）。
func LookupKeyVersion(version string) (*KeyVersion, error) {
	keyRegistryMu.RLock()
	defer keyRegistryMu.RUnlock()
	for _, kv := range keyRegistry {
		if kv.Version == version {
			return kv, nil
		}
	}
	return nil, ErrKeyVersionNotFound
}

// DeriveKeyHKDF derives a 16-byte SM4 key from a passphrase and a per-record random
// salt using HKDF (RFC 5869) with SM3 as the underlying hash and default hkdfInfo.
//
// DeriveKeyHKDF 以 HKDF-Extract/Expand 从口令与逐记录随机 salt 派生 16 字节 SM4 密钥，
// 使同一口令在不同记录上产出互不相同的加密密钥，并抵抗短语令的离线暴破。
func DeriveKeyHKDF(secret string, salt []byte) []byte {
	return DeriveKeyHKDFWithInfo(secret, salt, hkdfInfo)
}

// DeriveKeyHKDFWithInfo 使用指定的 HKDF 上下文字符串 info 派生 16 字节 SM4 密钥。
func DeriveKeyHKDFWithInfo(secret string, salt []byte, info string) []byte {
	prk := hkdfExtract(salt, []byte(secret))
	defer Zeroize(prk)
	return hkdfExpand(prk, []byte(info), KeySize)
}

func hkdfExtract(salt, ikm []byte) []byte {
	mac := hmac.New(NewSM3, salt)
	mac.Write(ikm)
	return mac.Sum(nil)
}

func hkdfExpand(prk, info []byte, length int) []byte {
	out := make([]byte, 0, length+SM3Size)
	var t []byte
	var counter byte = 1
	for len(out) < length {
		mac := hmac.New(NewSM3, prk)
		if len(t) > 0 {
			mac.Write(t)
		}
		mac.Write(info)
		mac.Write([]byte{counter})
		t = mac.Sum(nil)
		out = append(out, t...)
		counter++
	}
	return out[:length]
}

// EncryptString encrypts plaintext into a versioned envelope string.
//
// 格式规范：enc:<version>:<Base64(salt || nonce || ciphertext || tag)>
//
// 流程：
//  1. 若 plaintext 为空，直接返回空串（与 DecryptString 保持对称，避免空密文膨胀）；
//  2. 确定密钥与版本：优先使用 ActiveKeyVersion()；若无活跃版本，使用传入的 secret 与 DefaultKeyVersion；若两者皆空则返回 ErrEmptyKey；
//  3. 生成 16 字节独占随机 salt 与 12 字节随机 nonce；
//  4. 通过 HKDF-SM3 派生 16 字节 SM4 密钥；
//  5. AAD 严格绑定完整信封头部（"enc:<version>:"），防止篡改版本号或剥离前缀；
//  6. 执行 SM4-GCM Seal，输出统一信封格式密文字符串。
func EncryptString(plaintext, secret string) (string, error) {
	if plaintext == "" {
		auditCrypto("sm4_encrypt", "", 0, true, nil)
		return "", nil
	}

	var version string
	var keyMaterial string

	if active := ActiveKeyVersion(); active != nil {
		version = active.Version
		keyMaterial = string(active.Key)
	} else if secret != "" {
		version = DefaultKeyVersion
		keyMaterial = secret
	} else {
		auditCrypto("sm4_encrypt", "", len(plaintext), false, ErrEmptyKey)
		return "", ErrEmptyKey
	}

	salt := make([]byte, SaltSize)
	if _, err := io.ReadFull(rand.Reader, salt); err != nil {
		auditCrypto("sm4_encrypt", version, len(plaintext), false, err)
		return "", fmt.Errorf("generate salt: %w", err)
	}

	derivedKey := DeriveKeyHKDF(keyMaterial, salt)
	defer Zeroize(derivedKey)

	gcm, err := newGCM(derivedKey)
	if err != nil {
		auditCrypto("sm4_encrypt", version, len(plaintext), false, err)
		return "", fmt.Errorf("create sm4-gcm: %w", err)
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		auditCrypto("sm4_encrypt", version, len(plaintext), false, err)
		return "", fmt.Errorf("generate nonce: %w", err)
	}

	header := EncryptedPrefix + version + ":"
	sealed := gcm.Seal(nonce, nonce, []byte(plaintext), []byte(header))

	payload := base64.StdEncoding.EncodeToString(append(salt, sealed...))
	envelope := header + payload

	auditCrypto("sm4_encrypt", version, len(plaintext), true, nil)
	return envelope, nil
}

// keyRegistryCount 返回当前注册的密钥版本总数。
func keyRegistryCount() int {
	keyRegistryMu.RLock()
	defer keyRegistryMu.RUnlock()
	return len(keyRegistry)
}

// DecryptString decrypts an envelope-encrypted value produced by this package.
// It accepts only the canonical salted SM4-GCM format with authenticated envelope headers.
//
// 流程：
//  1. 若 ciphertext 为空，直接返回空串；
//  2. 校验是否以 EncryptedPrefix ("enc:") 开头，若不是则拒绝并返回 ErrUnencryptedValue；
//  3. 历史信封格式不提供解密回退；
//  4. 规范格式解析出 <version> 与 Base64 密文载荷；
//  5. 严格版本查找：已配置 registry 时指定版本未命中坚决返回 ErrKeyVersionNotFound（Fail-Closed）；
//  6. Base64 解码载荷并使用 HKDF-SM3 派生密钥，以完整信封头为 AAD 执行 SM4-GCM 解密。
func DecryptString(ciphertext, secret string) (string, error) {
	if ciphertext == "" {
		auditCrypto("sm4_decrypt", "", 0, true, nil)
		return "", nil
	}

	if !strings.HasPrefix(ciphertext, EncryptedPrefix) {
		auditCrypto("sm4_decrypt", "", len(ciphertext), false, ErrUnencryptedValue)
		return "", ErrUnencryptedValue
	}

	trimmed := strings.TrimPrefix(ciphertext, EncryptedPrefix)
	idx := strings.Index(trimmed, ":")
	if idx <= 0 || idx >= len(trimmed)-1 {
		auditCrypto("sm4_decrypt", "", len(ciphertext), false, ErrInvalidEnvelope)
		return "", ErrInvalidEnvelope
	}

	version := trimmed[:idx]
	encodedPayload := trimmed[idx+1:]

	var keyMaterial string
	if kv, err := LookupKeyVersion(version); err == nil {
		keyMaterial = string(kv.Key)
	} else if keyRegistryCount() == 0 && version == DefaultKeyVersion && secret != "" {
		// 未启用多版本注册表时，回退到传入的 secret（单密钥运行态）
		keyMaterial = secret
	} else {
		// 已配置注册表但请求的版本未命中，严格拒绝（Fail-Closed，防止版本漂移绕过）
		auditCrypto("sm4_decrypt", version, len(ciphertext), false, ErrKeyVersionNotFound)
		return "", ErrKeyVersionNotFound
	}

	data, err := base64.StdEncoding.DecodeString(encodedPayload)
	if err != nil {
		auditCrypto("sm4_decrypt", version, len(ciphertext), false, err)
		return "", fmt.Errorf("base64 decode envelope: %w", err)
	}

	// 标准规范载荷必须包含: SaltSize(16) + NonceSize(12) + TagSize(16) = 44 字节
	const minPayloadLen = SaltSize + NonceSize + 16
	if len(data) >= minPayloadLen {
		salt := data[:SaltSize]
		rest := data[SaltSize:]
		nonce := rest[:NonceSize]
		ctAndTag := rest[NonceSize:]

		derivedKey := DeriveKeyHKDF(keyMaterial, salt)
		gcm, err := newGCM(derivedKey)
		Zeroize(derivedKey)
		if err == nil {
			header := EncryptedPrefix + version + ":"
			plaintext, err := gcm.Open(nil, nonce, ctAndTag, []byte(header))
			if err == nil {
				auditCrypto("sm4_decrypt", version, len(plaintext), true, nil)
				return string(plaintext), nil
			}
		}
	}

	if len(data) < minPayloadLen {
		auditCrypto("sm4_decrypt", version, len(ciphertext), false, ErrInvalidEnvelope)
		return "", fmt.Errorf("%w: ciphertext too short for canonical envelope (%d < %d)", ErrInvalidEnvelope, len(data), minPayloadLen)
	}

	auditCrypto("sm4_decrypt", version, len(ciphertext), false, errors.New("sm4-gcm decrypt failed"))
	return "", fmt.Errorf("sm4-gcm decrypt failed (invalid key or tampered data)")
}

func newGCM(key []byte) (cipher.AEAD, error) {
	block, err := NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("create sm4 cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("create sm4-gcm: %w", err)
	}
	return gcm, nil
}

// IsEncrypted returns true if the value carries a recognized envelope prefix.
// It checks the canonical envelope structure; authenticated decoding is done by DecryptString.
func IsEncrypted(value string) bool {
	if !strings.HasPrefix(value, EncryptedPrefix) {
		return false
	}
	rest := strings.TrimPrefix(value, EncryptedPrefix)
	idx := strings.Index(rest, ":")
	return idx > 0 && len(rest) > idx+1
}
