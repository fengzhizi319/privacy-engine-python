"""测试专用自签名证书生成工具 / Ephemeral Certificate Generation Helpers for Security Tests.

中文说明：
为 TLS/mTLS 安全测试动态生成临时 CA/服务端/客户端证书链，避免在仓库中提交真实证书。
生成的证书包括：
- 受信 CA 证书（用于签发服务端/客户端证书）。
- 服务端证书（含 localhost/127.0.0.1 SAN）。
- 客户端证书（用于 mTLS 双向认证）。
- 不受信 CA 及其签发的服务端证书（用于负面测试）。

English Description:
Helpers to generate ephemeral CA/server/client certificates for security tests.
Avoids committing real certificates to the repository. Generated certs include:
- Trusted CA certificate (for signing server/client certs).
- Server certificate (with localhost/127.0.0.1 SAN).
- Client certificate (for mTLS mutual authentication).
- Untrusted CA and its server certificate (for negative tests).
"""

from __future__ import annotations

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def _generate_key() -> rsa.RSAPrivateKey:
    """生成 2048 位 RSA 私钥 / Generate 2048-bit RSA Private Key."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _write_pem(path: Path, data: bytes) -> None:
    """将 PEM 编码数据写入文件 / Write PEM-encoded Data to File."""
    path.write_bytes(data)


def _make_ca(name: str, key: rsa.RSAPrivateKey) -> x509.Certificate:
    """创建自签名 CA 证书 / Create Self-Signed CA Certificate.

    中文说明：生成 BasicConstraints(ca=True) 的根 CA 证书，
    有效期 1 天，用于签发测试用服务端/客户端证书。

    English Description: Generates a root CA certificate with BasicConstraints(ca=True),
    valid for 1 day, used to sign test server/client certificates.

    Args:
        name: CA 证书的 Common Name / Common Name for the CA certificate.
        key: CA 私钥 / CA private key.

    Returns:
        自签名 CA 证书 / Self-signed CA certificate.
    """
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                data_encipherment=False,
                decipher_only=False,
                encipher_only=False,
                key_agreement=False,
                key_encipherment=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return cert


def _make_end_entity(
    cn: str,
    subject_key: rsa.RSAPrivateKey,
    issuer_cert: x509.Certificate,
    issuer_key: rsa.RSAPrivateKey,
    *,
    san: list[x509.GeneralName] | None = None,
    extended_key_usage: list[x509.ObjectIdentifier] | None = None,
) -> x509.Certificate:
    """创建由 CA 签发的终端实体证书 / Create CA-Signed End-Entity Certificate.

    中文说明：生成服务端或客户端证书，支持可选的 SAN 和扩展密钥用途。
    有效期 1 天，仅用于测试。

    English Description: Generates a server or client certificate signed by the CA,
    with optional SAN and Extended Key Usage. Valid for 1 day, test-only.

    Args:
        cn: 证书 Common Name / Certificate Common Name.
        subject_key: 终端实体私钥 / End-entity private key.
        issuer_cert: 签发者 CA 证书 / Issuer CA certificate.
        issuer_key: 签发者 CA 私钥 / Issuer CA private key.
        san: 主题备用名称列表 / Subject Alternative Names (optional).
        extended_key_usage: 扩展密钥用途 / Extended Key Usage OIDs (optional).

    Returns:
        CA 签发的终端实体证书 / CA-signed end-entity certificate.
    """
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_cert.subject)
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
    )
    # 添加主题备用名称（如 localhost, 127.0.0.1）
    if san:
        builder = builder.add_extension(x509.SubjectAlternativeName(san), critical=False)
    # 添加扩展密钥用途（SERVER_AUTH 或 CLIENT_AUTH）
    if extended_key_usage:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage(extended_key_usage), critical=False
        )
    builder = builder.add_extension(
        x509.SubjectKeyIdentifier.from_public_key(subject_key.public_key()),
        critical=False,
    )
    builder = builder.add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()),
        critical=False,
    )
    return builder.sign(issuer_key, hashes.SHA256())


def generate_test_certs(tmp_dir: Path) -> dict[str, Path]:
    """生成完整的测试证书链 / Generate Complete Test Certificate Chain.

    中文说明：
    生成受信 CA、服务端证书、客户端证书以及不受信 CA/服务端证书（用于负面测试）。
    所有证书以 PEM 格式写入临时目录。

    English Description:
    Generates a trusted CA, server cert, client cert, and an untrusted CA/server cert
    (for negative tests). All certificates are written in PEM format to a temp directory.

    Args:
        tmp_dir: 证书输出目录 / Certificate output directory.

    Returns:
        包含各证书/密钥文件路径的字典 / Dictionary of PEM file paths.
        Keys: ca_cert, server_cert, server_key, client_cert, client_key,
              bad_ca_cert, bad_server_cert, bad_server_key.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 受信 CA（用于签发服务端/客户端证书）
    ca_key = _generate_key()
    ca_cert = _make_ca("PrivShield-test-ca", ca_key)

    # 服务端证书（含 localhost / 127.0.0.1 SAN，用于 TLS 服务端认证）
    server_key = _generate_key()
    server_cert = _make_end_entity(
        "localhost",
        server_key,
        ca_cert,
        ca_key,
        san=[
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ],
        extended_key_usage=[ExtendedKeyUsageOID.SERVER_AUTH],
    )

    # 客户端证书（用于 mTLS 双向认证）
    client_key = _generate_key()
    client_cert = _make_end_entity(
        "internal-client",
        client_key,
        ca_cert,
        ca_key,
        extended_key_usage=[ExtendedKeyUsageOID.CLIENT_AUTH],
    )

    # 不受信 CA + 服务端证书（用于负面测试：验证不受信 CA 连接被拒绝）
    bad_ca_key = _generate_key()
    bad_ca_cert = _make_ca("untrusted-test-ca", bad_ca_key)
    bad_server_key = _generate_key()
    bad_server_cert = _make_end_entity(
        "localhost",
        bad_server_key,
        bad_ca_cert,
        bad_ca_key,
        san=[x509.DNSName("localhost")],
        extended_key_usage=[ExtendedKeyUsageOID.SERVER_AUTH],
    )

    # 定义输出文件路径
    paths: dict[str, Path] = {
        "ca_cert": tmp_dir / "ca.crt",
        "server_cert": tmp_dir / "server.crt",
        "server_key": tmp_dir / "server.key",
        "client_cert": tmp_dir / "client.crt",
        "client_key": tmp_dir / "client.key",
        "bad_ca_cert": tmp_dir / "bad_ca.crt",
        "bad_server_cert": tmp_dir / "bad_server.crt",
        "bad_server_key": tmp_dir / "bad_server.key",
    }

    # 写入所有证书和密钥文件（PEM 格式）
    _write_pem(paths["ca_cert"], ca_cert.public_bytes(serialization.Encoding.PEM))
    _write_pem(
        paths["server_cert"], server_cert.public_bytes(serialization.Encoding.PEM)
    )
    _write_pem(
        paths["server_key"],
        server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )
    _write_pem(
        paths["client_cert"], client_cert.public_bytes(serialization.Encoding.PEM)
    )
    _write_pem(
        paths["client_key"],
        client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )
    _write_pem(
        paths["bad_ca_cert"], bad_ca_cert.public_bytes(serialization.Encoding.PEM)
    )
    _write_pem(
        paths["bad_server_cert"],
        bad_server_cert.public_bytes(serialization.Encoding.PEM),
    )
    _write_pem(
        paths["bad_server_key"],
        bad_server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )

    return paths
