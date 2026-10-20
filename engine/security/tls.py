"""TLS helpers for REST (uvicorn) and gRPC.

为 Uvicorn 与 gRPC server 构造 TLS 参数。支持仅服务端 TLS 与可选/强制 mTLS。
Provides helpers to build TLS arguments for Uvicorn and gRPC, including optional and
required mutual TLS.
"""

from __future__ import annotations

import ssl
from typing import TYPE_CHECKING, Any

import grpc

if TYPE_CHECKING:
    from .config import SecuritySettings


def _map_client_auth(mode: str) -> int:
    """Map our textual client-auth mode to the ssl module's certificate requirement."""
    return {
        "none": ssl.CERT_NONE,
        "optional": ssl.CERT_OPTIONAL,
        "require": ssl.CERT_REQUIRED,
    }[mode]


def uvicorn_ssl_kwargs(settings: SecuritySettings) -> dict[str, Any]:
    """Build the keyword arguments for ``uvicorn.run`` when TLS is enabled.

    Returns a dictionary that can be spread into ``uvicorn.run(..., **ssl_kwargs)``.
    If TLS is disabled the returned dictionary is empty.
    """
    if not settings.tls_enabled:
        return {}

    kwargs: dict[str, Any] = {
        "ssl_keyfile": str(settings.tls_key_file),
        "ssl_certfile": str(settings.tls_cert_file),
        "ssl_cert_reqs": _map_client_auth(settings.tls_client_auth),
        # Use a server-side SSL context so that client-auth settings are enforced.
        "ssl_version": ssl.PROTOCOL_TLS_SERVER,
    }
    if settings.tls_key_password is not None:
        kwargs["ssl_keyfile_password"] = settings.tls_key_password
    if settings.tls_ca_file is not None:
        kwargs["ssl_ca_certs"] = str(settings.tls_ca_file)
    return kwargs


def grpc_server_credentials(settings: SecuritySettings) -> grpc.ServerCredentials:
    """Build gRPC server credentials from the security settings.

    - ``tls_client_auth == "none"``: standard server-side TLS.
    - ``tls_client_auth == "optional"``: server TLS + request but do not require client cert.
    - ``tls_client_auth == "require"``: mutual TLS; client must present a trusted cert.
    """
    if not settings.tls_enabled:
        raise RuntimeError("grpc_server_credentials called with TLS disabled.")

    private_key = settings.tls_key_file.read_bytes()  # type: ignore[union-attr]
    certificate_chain = settings.tls_cert_file.read_bytes()  # type: ignore[union-attr]

    if settings.tls_client_auth in ("optional", "require"):
        root_certificates = settings.tls_ca_file.read_bytes()  # type: ignore[union-attr]
        return grpc.ssl_server_credentials(
            ((private_key, certificate_chain),),
            root_certificates=root_certificates,
            require_client_auth=(settings.tls_client_auth == "require"),
        )

    return grpc.ssl_server_credentials(((private_key, certificate_chain),))
