package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	pkgcrypto "github.com/fengzhizi319/PrivShield-go/pkg/crypto"
	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
	"google.golang.org/grpc/metadata"
)

const testIdentityKey = "0123456789abcdef0123456789abcdef"

func testSigner(t *testing.T) *identitySigner {
	t.Helper()
	signer, err := newIdentitySigner([]byte(testIdentityKey), "sm3")
	if err != nil {
		t.Fatalf("newIdentitySigner: %v", err)
	}
	return signer
}

func testPeer() *tlsutil.PeerIdentity {
	return &tlsutil.PeerIdentity{
		CommonName: "privshield-test-client",
		CertDER:    []byte("test-cert-der"),
		SPKIDER:    []byte("test-spki-der"),
		DNSNames:   []string{"client.privshield.internal"},
		Verified:   true,
	}
}

func TestCanonicalIdentityExact(t *testing.T) {
	got := canonicalIdentity("post", "/v1/privacy/mask", "svc-a", "certfp", "spkifp", 1700000000)
	want := "privshield-identity-v1\nPOST\n/v1/privacy/mask\nsvc-a\ncertfp\nspkifp\n1700000000"
	if got != want {
		t.Fatalf("canonicalIdentity mismatch:\n got: %q\nwant: %q", got, want)
	}
}

func TestSignMatchesReferenceHMACSM3(t *testing.T) {
	signer := testSigner(t)
	canonical := canonicalIdentity("POST", "/PrivacyService/Process", "svc-a", "cert", "spki", 1700000000)
	got := signer.sign(canonical)
	want := pkgcrypto.HMACSM3Hex([]byte(testIdentityKey), []byte(canonical))
	if got != want {
		t.Fatalf("sign mismatch: got %s want %s", got, want)
	}
	if len(got) != 64 {
		t.Fatalf("SM3 HMAC hex length = %d, want 64", len(got))
	}
}

func TestSignSHA256(t *testing.T) {
	signer, err := newIdentitySigner([]byte(testIdentityKey), "sha256")
	if err != nil {
		t.Fatalf("newIdentitySigner: %v", err)
	}
	got := signer.sign("payload")
	mac := hmac.New(sha256.New, []byte(testIdentityKey))
	_, _ = mac.Write([]byte("payload"))
	want := hex.EncodeToString(mac.Sum(nil))
	if got != want {
		t.Fatalf("sha256 HMAC mismatch: got %s want %s", got, want)
	}
}

func TestNewIdentitySignerRejectsBadConfig(t *testing.T) {
	if _, err := newIdentitySigner([]byte("short"), "sm3"); err == nil {
		t.Fatal("short key must be rejected")
	}
	if _, err := newIdentitySigner([]byte(testIdentityKey), "md5"); err == nil {
		t.Fatal("unsupported algorithm must be rejected")
	}
}

func TestRestHeadersSignAndStripForgery(t *testing.T) {
	signer := testSigner(t)
	peer := testPeer()
	headers := signer.restHeaders(http.MethodPost, "/v1/privacy/mask", peer)
	if headers[headerClientCN] != peer.CommonName {
		t.Fatalf("CN header = %q", headers[headerClientCN])
	}
	certSum := sha256.Sum256(peer.CertDER)
	if headers[headerClientCertSHA256] != hex.EncodeToString(certSum[:]) {
		t.Fatalf("cert fingerprint = %q", headers[headerClientCertSHA256])
	}
	if headers[headerClientSPKIDER] != base64.StdEncoding.EncodeToString(peer.SPKIDER) {
		t.Fatalf("SPKI DER header = %q", headers[headerClientSPKIDER])
	}
	der, err := base64.StdEncoding.DecodeString(headers[headerClientSPKIDER])
	if err != nil {
		t.Fatalf("decode SPKI DER: %v", err)
	}
	derSum := sha256.Sum256(der)
	if hex.EncodeToString(derSum[:]) != headers[headerClientSPKI] {
		t.Fatal("SPKI DER must hash to the signed SPKI fingerprint")
	}
	sig := signer.sign(canonicalIdentity(
		http.MethodPost,
		"/v1/privacy/mask",
		peer.CommonName,
		headers[headerClientCertSHA256],
		headers[headerClientSPKI],
		1700000000,
	))
	if len(headers[headerIdentitySig]) != len(sig) {
		t.Fatalf("signature length = %d, want %d", len(headers[headerIdentitySig]), len(sig))
	}

	forged := http.Header{}
	forged.Set(strings.ToLower(headerClientCN), "attacker")
	forged.Set(headerClientCertSHA256, "deadbeef")
	forged.Set(headerClientSPKIDER, "Zm9yZ2Vk")
	forged.Set(headerIdentitySig, "forged")
	forged.Set("X-Other", "keep")
	stripRESTIdentityHeaders(forged)
	for _, name := range []string{headerClientCN, headerClientCertSHA256, headerClientSPKIDER, headerIdentitySig} {
		if got := forged.Get(name); got != "" {
			t.Fatalf("forged header %s was not stripped: %q", name, got)
		}
	}
	if forged.Get("X-Other") != "keep" {
		t.Fatal("unrelated header must be preserved")
	}
}

func TestGRPCMetadataSignAndStripForgery(t *testing.T) {
	signer := testSigner(t)
	md := signer.grpcMetadata("/PrivacyService/Process", testPeer())
	if got := md.Get(mdClientCN); len(got) != 1 || got[0] != "privshield-test-client" {
		t.Fatalf("CN metadata = %v", got)
	}
	if got := md.Get(mdIdentitySig); len(got) != 1 || len(got[0]) != 64 {
		t.Fatalf("signature metadata = %v", got)
	}
	if got := md.Get(mdClientSPKIDER); len(got) != 1 || got[0] != base64.StdEncoding.EncodeToString(testPeer().SPKIDER) {
		t.Fatalf("SPKI DER metadata = %v", got)
	}

	incoming := metadata.Pairs(mdClientCN, "attacker", mdClientSPKIDER, "Zm9yZ2Vk", mdIdentitySig, "forged", "x-request-id", "req-1")
	stripGRPCIdentityMetadata(incoming)
	if incoming.Get(mdClientCN) != nil || incoming.Get(mdClientSPKIDER) != nil || incoming.Get(mdIdentitySig) != nil {
		t.Fatalf("forged metadata survived: %v", incoming)
	}
	if got := incoming.Get("x-request-id"); len(got) != 1 || got[0] != "req-1" {
		t.Fatalf("request id metadata lost: %v", got)
	}
}

func writeWhitelistFile(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mtls-whitelist.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write whitelist: %v", err)
	}
	return path
}

func TestAuthorizePeerGoWhitelistSchema(t *testing.T) {
	peer := testPeer()
	sum := sha256.Sum256(peer.CertDER)
	fp := hex.EncodeToString(sum[:])

	path := writeWhitelistFile(t, `version: "1.0"
clients:
  - cn: privshield-test-client
    allowed_scopes: ["*"]
    cert_fingerprints: ["`+fp+`"]
    enabled: true
`)
	dw, err := tlsutil.NewDynamicWhitelist(path)
	if err != nil {
		t.Fatalf("NewDynamicWhitelist: %v", err)
	}
	defer dw.Close()

	identity, ok := authorizePeer(dw, peer)
	if !ok || identity != peer.CommonName {
		t.Fatalf("authorizePeer = (%q,%v), want (%q,true)", identity, ok, peer.CommonName)
	}

	mismatch := *peer
	mismatch.CertDER = []byte("different-cert")
	if _, ok := authorizePeer(dw, &mismatch); ok {
		t.Fatal("certificate fingerprint mismatch must be rejected")
	}

	unknown := &tlsutil.PeerIdentity{CommonName: "someone-else", CertDER: []byte("x"), Verified: true}
	if _, ok := authorizePeer(dw, unknown); ok {
		t.Fatal("unknown CN must be rejected")
	}
}

func TestAuthorizePeerNoWhitelistFailsClosed(t *testing.T) {
	if _, ok := authorizePeer(nil, testPeer()); ok {
		t.Fatal("nil whitelist must not authorize")
	}
}
