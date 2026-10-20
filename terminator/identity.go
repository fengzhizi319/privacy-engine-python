// Package main — 终结器身份头签名与剥离。
//
// 终结器是 Python 引擎唯一的 TLS 边界：mTLS 在终结器完成，回环链路是明文，
// 因此客户端证书身份必须以**带签名的应用层头**跨过这一跳。Python 侧只接受
// 来自回环且 HMAC 校验通过的身份头；终结器必须先剥离客户端伪造的同名头，
// 再注入自己的签名头，防止外部请求直接把身份"写"进回环链路。
package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/emmansun/gmsm/sm3"
	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
	"google.golang.org/grpc/metadata"
)

const (
	// identityCanonicalVersion 是签名规范串的版本前缀；改动字段顺序或含义必须升版。
	identityCanonicalVersion = "privshield-identity-v1"

	headerClientCN          = "X-PrivShield-Client-CN"
	headerClientCertSHA256  = "X-PrivShield-Client-Cert-SHA256"
	headerClientSPKI        = "X-PrivShield-Client-SPKI"
	headerClientSPKIDER     = "X-PrivShield-Client-SPKI-DER"
	headerIdentityTimestamp = "X-PrivShield-Identity-Timestamp"
	headerIdentitySig       = "X-PrivShield-Identity-Sig"

	mdClientCN          = "x-privshield-client-cn"
	mdClientCertSHA256  = "x-privshield-client-cert-sha256"
	mdClientSPKI        = "x-privshield-client-spki"
	mdClientSPKIDER     = "x-privshield-client-spki-der"
	mdIdentityTimestamp = "x-privshield-identity-timestamp"
	mdIdentitySig       = "x-privshield-identity-sig"
)

// identitySigner 按选定算法对身份规范串做 HMAC。
type identitySigner struct {
	key  []byte
	algo string // "sm3" | "sha256"
}

func newIdentitySigner(key []byte, algo string) (*identitySigner, error) {
	algo = strings.ToLower(strings.TrimSpace(algo))
	switch algo {
	case "sm3", "sha256":
	default:
		return nil, fmt.Errorf("unsupported identity HMAC algorithm %q (want sm3 or sha256)", algo)
	}
	if len(key) < 16 {
		return nil, fmt.Errorf("identity HMAC key must be at least 16 bytes (got %d)", len(key))
	}
	return &identitySigner{key: append([]byte(nil), key...), algo: algo}, nil
}

// canonicalIdentity 是跨语言签名规范串：
//
//	privshield-identity-v1\n
//	<METHOD>\n
//	<PATH>\n
//	<CN>\n
//	<CERT_SHA256>\n
//	<SPKI_SHA256>\n
//	<TIMESTAMP>
//
// 字段固定、无转义歧义；gRPC 侧 METHOD 固定为 POST、PATH 为完整方法名
// （如 /PrivacyService/Process），与 HTTP 的 method/path 语义对齐。
func canonicalIdentity(method, path, cn, certSHA256, spkiSHA256 string, timestamp int64) string {
	return identityCanonicalVersion + "\n" +
		strings.ToUpper(method) + "\n" +
		path + "\n" +
		cn + "\n" +
		certSHA256 + "\n" +
		spkiSHA256 + "\n" +
		strconv.FormatInt(timestamp, 10)
}

func (s *identitySigner) sign(canonical string) string {
	var mac hashWriter
	if s.algo == "sha256" {
		mac = hmac.New(sha256.New, s.key)
	} else {
		mac = hmac.New(sm3.New, s.key)
	}
	_, _ = mac.Write([]byte(canonical))
	return hex.EncodeToString(mac.Sum(nil))
}

type hashWriter interface {
	Write([]byte) (int, error)
	Sum([]byte) []byte
}

// certFingerprint 返回叶证书 DER 的小写 hex SHA-256。
func certFingerprint(peer *tlsutil.PeerIdentity) string {
	if peer == nil || len(peer.CertDER) == 0 {
		return ""
	}
	sum := sha256.Sum256(peer.CertDER)
	return hex.EncodeToString(sum[:])
}

// spkiFingerprint 返回 SPKI DER 的小写 hex SHA-256。
func spkiFingerprint(peer *tlsutil.PeerIdentity) string {
	if peer == nil || len(peer.SPKIDER) == 0 {
		return ""
	}
	sum := sha256.Sum256(peer.SPKIDER)
	return hex.EncodeToString(sum[:])
}

// restHeaders 生成注入到回环 REST 请求的身份头；无对端证书时返回 nil。
//
// 除摘要外还注入 base64(SPKI DER)：Python 侧服务账号激活需要 DER 本体
// （与 Go 侧 expected_public_key_sha256 比对），仅凭摘要无法完成。
// DER 本身不进签名规范串，其完整性由「sha256(DER) == 已签名的 SPKI 摘要」
// 在接收端校验，因此规范串版本无需变更。
func (s *identitySigner) restHeaders(method, path string, peer *tlsutil.PeerIdentity) map[string]string {
	if s == nil || peer == nil || peer.Empty() {
		return nil
	}
	cn := peer.CommonName
	certFP := certFingerprint(peer)
	spkiFP := spkiFingerprint(peer)
	ts := time.Now().Unix()
	sig := s.sign(canonicalIdentity(method, path, cn, certFP, spkiFP, ts))
	return map[string]string{
		headerClientCN:          cn,
		headerClientCertSHA256:  certFP,
		headerClientSPKI:        spkiFP,
		headerClientSPKIDER:     base64.StdEncoding.EncodeToString(peer.SPKIDER),
		headerIdentityTimestamp: strconv.FormatInt(ts, 10),
		headerIdentitySig:       sig,
	}
}

// grpcMetadata 生成注入到回环 gRPC 请求的身份 metadata，键名一律小写。
func (s *identitySigner) grpcMetadata(method string, peer *tlsutil.PeerIdentity) metadata.MD {
	if s == nil || peer == nil || peer.Empty() {
		return nil
	}
	cn := peer.CommonName
	certFP := certFingerprint(peer)
	spkiFP := spkiFingerprint(peer)
	ts := time.Now().Unix()
	sig := s.sign(canonicalIdentity(http.MethodPost, method, cn, certFP, spkiFP, ts))
	return metadata.Pairs(
		mdClientCN, cn,
		mdClientCertSHA256, certFP,
		mdClientSPKI, spkiFP,
		mdClientSPKIDER, base64.StdEncoding.EncodeToString(peer.SPKIDER),
		mdIdentityTimestamp, strconv.FormatInt(ts, 10),
		mdIdentitySig, sig,
	)
}

// stripRESTIdentityHeaders 移除客户端可能自带的身份头（含大小写变体）。
func stripRESTIdentityHeaders(h http.Header) {
	for _, name := range []string{
		headerClientCN,
		headerClientCertSHA256,
		headerClientSPKI,
		headerClientSPKIDER,
		headerIdentityTimestamp,
		headerIdentitySig,
	} {
		h.Del(name)
	}
}

// stripGRPCIdentityMetadata 移除客户端可能自带的身份 metadata。
func stripGRPCIdentityMetadata(md metadata.MD) {
	for _, key := range []string{
		mdClientCN,
		mdClientCertSHA256,
		mdClientSPKI,
		mdClientSPKIDER,
		mdIdentityTimestamp,
		mdIdentitySig,
	} {
		md.Delete(key)
	}
}

// peerIdentities 返回白名单匹配身份，顺序与 Go gRPC 拦截器一致：
// SAN URI → SAN DNS → Subject CN。
func peerIdentities(peer *tlsutil.PeerIdentity) []string {
	if peer == nil || peer.Empty() {
		return nil
	}
	out := make([]string, 0, len(peer.URIs)+len(peer.DNSNames)+1)
	for _, uri := range peer.URIs {
		if uri != "" {
			out = append(out, uri)
		}
	}
	for _, dns := range peer.DNSNames {
		if dns != "" {
			out = append(out, dns)
		}
	}
	if peer.CommonName != "" {
		out = append(out, peer.CommonName)
	}
	return out
}

// authorizePeer 在白名单中校验对端身份并执行 cert_fingerprints 钉扎。
// 返回命中的身份与是否通过；REST 侧的 scope 裁决仍由 Python 引擎完成。
func authorizePeer(dw *tlsutil.DynamicWhitelist, peer *tlsutil.PeerIdentity) (string, bool) {
	if dw == nil || peer == nil || peer.Empty() {
		return "", false
	}
	peerFP := certFingerprint(peer)
	for _, identity := range peerIdentities(peer) {
		if _, ok := dw.GetScopes(identity); !ok {
			continue
		}
		if pinned, enabled := dw.GetCertFingerprints(identity); enabled {
			matched := false
			for _, want := range pinned {
				if hmac.Equal([]byte(peerFP), []byte(want)) {
					matched = true
					break
				}
			}
			if !matched {
				return identity, false
			}
		}
		return identity, true
	}
	return "", false
}
