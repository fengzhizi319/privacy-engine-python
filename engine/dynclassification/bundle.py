"""策略清单（Policy Bundle）加载与完整性校验模块 / Policy Bundle Manifest Loader & Verifier.

中文说明：
对齐 Go 版本的 privshield-policy-bundle/v1 清单规范（services/privacy-engine/internal/configinput/bundle.go）。
以 JSON 清单作为策略、领域规则、分级词表与标准组合的权威文件输入。
加载时校验架构版本、发布标识、相对路径安全性、预算约束（文件大小与总数），
并通过 SHA-256 摘要常量时间比对确保所有配置文件完整性与不可篡改性。

English Description:
Implements policy bundle loading and integrity verification matching Go's
privshield-policy-bundle/v1 specification. Loads and verifies all policy, domain,
taxonomy, and standard inputs defined in a manifest.json file with strict security
guards (path traversal rejection, nested symlink denial, size budgets, and constant-time SHA256 checks).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..observability.logging_config import get_logger

logger = get_logger(__name__)

# 预算上限（与 Go 对齐）
MAX_DIRECTORY_ENTRIES = 1024
MAX_SINGLE_FILE_BYTES = 4 * 1024 * 1024  # 4 MiB
MAX_BUNDLE_TOTAL_BYTES = 16 * 1024 * 1024  # 16 MiB
MANIFEST_SCHEMA = "privshield-policy-bundle/v1"


class BundleValidationError(ValueError):
    """策略清单校验失败异常。"""


@dataclass(frozen=True)
class BundleFile:
    """清单引用的单一文件条目。"""

    path: str
    sha256: str


@dataclass(frozen=True)
class BundleManifest:
    """策略清单定义 (privshield-policy-bundle/v1)。"""

    schema: str
    release: str
    policy: BundleFile
    domains: list[BundleFile] = field(default_factory=list)
    taxonomies: list[BundleFile] = field(default_factory=list)
    standards: list[BundleFile] = field(default_factory=list)


@dataclass
class Bundle:
    """已加载并验真的策略包。"""

    root: str
    release: str
    policy_path: str
    policy: bytes
    domains: dict[str, bytes] = field(default_factory=dict)
    taxonomies: dict[str, bytes] = field(default_factory=dict)
    standards: dict[str, bytes] = field(default_factory=dict)


def _validate_release(release: str) -> None:
    """校验 release 标识符：1..128 字节可见 ASCII 字符，首尾无空白。"""
    if not release or len(release) > 128 or release.strip() != release:
        raise BundleValidationError(
            "manifest release must contain 1..128 bytes without surrounding whitespace"
        )
    for c in release:
        code = ord(c)
        if code < 33 or code > 126:
            raise BundleValidationError(
                f"manifest release must be visible ASCII, got char {c!r} (code {code})"
            )


def _parse_bundle_file(data: Any, context: str) -> BundleFile:
    """解析并校验单个文件条目对象。"""
    if not isinstance(data, dict):
        raise BundleValidationError(f"{context} must be an object")
    path = data.get("path")
    digest = data.get("sha256")
    if not isinstance(path, str) or not path:
        raise BundleValidationError(f"{context}.path must be a non-empty string")
    if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != digest:
        raise BundleValidationError(
            f"{context}.sha256 must be a 64-character lowercase hex string"
        )
    try:
        bytes.fromhex(digest)
    except ValueError as e:
        raise BundleValidationError(f"invalid sha256 hex in {context}: {e}") from e
    return BundleFile(path=path, sha256=digest)


def parse_manifest_dict(raw: dict[str, Any]) -> BundleManifest:
    """解析并校验清单字典数据。"""
    schema = raw.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise BundleValidationError(
            f"unsupported policy manifest schema: {schema!r}, expected {MANIFEST_SCHEMA!r}"
        )

    release = raw.get("release")
    if not isinstance(release, str):
        raise BundleValidationError("manifest release must be a string")
    _validate_release(release)

    policy_raw = raw.get("policy")
    if not isinstance(policy_raw, dict):
        raise BundleValidationError("manifest policy must be an object")
    policy_file = _parse_bundle_file(policy_raw, "policy")
    if policy_file.path != "privacy.yaml":
        raise BundleValidationError(
            f"manifest policy path must be 'privacy.yaml', got {policy_file.path!r}"
        )

    def parse_group(group_name: str) -> list[BundleFile]:
        group_raw = raw.get(group_name, [])
        if not isinstance(group_raw, list):
            raise BundleValidationError(f"manifest {group_name} must be a list")
        files: list[BundleFile] = []
        for i, item in enumerate(group_raw):
            bf = _parse_bundle_file(item, f"{group_name}[{i}]")
            # 必须为直接子文件，例如 domains/medical.yaml
            expected_parent = group_name
            p = Path(bf.path)
            if str(p.parent) != expected_parent:
                raise BundleValidationError(
                    f"manifest {group_name} files must be direct children of {expected_parent}, got {bf.path!r}"
                )
            files.append(bf)
        return files

    domains = parse_group("domains")
    taxonomies = parse_group("taxonomies")
    standards = parse_group("standards")

    total_files = 1 + len(domains) + len(taxonomies) + len(standards)
    if total_files > MAX_DIRECTORY_ENTRIES:
        raise BundleValidationError(
            f"manifest exceeds {MAX_DIRECTORY_ENTRIES} files (got {total_files})"
        )

    return BundleManifest(
        schema=schema,
        release=release,
        policy=policy_file,
        domains=domains,
        taxonomies=taxonomies,
        standards=standards,
    )


def load_bundle(manifest_path: str | Path) -> Bundle:
    """加载并验真策略包。

    Args:
        manifest_path: 指向 manifest.json 的文件路径（可为符号链接）。

    Returns:
        Bundle: 已完成验真并加载进内存的 Bundle 实体。

    Raises:
        BundleValidationError: 当清单格式不合法、存在符号链接穿越或摘要比对不符时抛出。
        FileNotFoundError: 当清单或引用文件不存在时抛出。
    """
    path_obj = Path(manifest_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")

    # 解析符号链接指针（只解析一次，确定发布目录 root）
    pinned_manifest = path_obj.resolve()
    bundle_root = pinned_manifest.parent

    # 读取清单文件
    manifest_raw_bytes = pinned_manifest.read_bytes()
    if len(manifest_raw_bytes) > MAX_SINGLE_FILE_BYTES:
        raise BundleValidationError("manifest file exceeds maximum size limit (4 MiB)")

    try:
        manifest_dict = json.loads(manifest_raw_bytes.decode("utf-8"))
    except Exception as e:
        raise BundleValidationError(f"failed to parse manifest json: {e}") from e

    manifest = parse_manifest_dict(manifest_dict)

    seen_paths: set[str] = set()
    total_bytes = len(manifest_raw_bytes)

    bundle = Bundle(
        root=str(bundle_root),
        release=manifest.release,
        policy_path="",
        policy=b"",
        domains={},
        taxonomies={},
        standards={},
    )

    def read_and_verify(bf: BundleFile) -> tuple[str, bytes]:
        nonlocal total_bytes
        # 路径安全性校验
        rel_path = bf.path
        if not rel_path or "\\" in rel_path or os.path.isabs(rel_path):
            raise BundleValidationError(f"invalid bundle path {rel_path!r}")
        if os.path.normpath(rel_path) != rel_path or rel_path == "." or rel_path.startswith(".."):
            raise BundleValidationError(f"invalid normalized bundle path {rel_path!r}")
        if rel_path in seen_paths:
            raise BundleValidationError(f"duplicate bundle path {rel_path!r}")
        seen_paths.add(rel_path)

        ext = Path(rel_path).suffix.lower()
        if ext not in (".yaml", ".yml"):
            raise BundleValidationError(f"bundle file {rel_path!r} must be YAML (.yaml or .yml)")

        # 检查路径中不能存在嵌套符号链接
        curr = bundle_root
        for part in Path(rel_path).parts:
            curr = curr / part
            if not curr.exists():
                raise FileNotFoundError(f"bundle file {rel_path!r} not found at {curr}")
            if curr.is_symlink():
                raise BundleValidationError(
                    f"bundle path {rel_path!r} contains a symlink at {curr}"
                )

        target_file = curr
        file_bytes = target_file.read_bytes()
        file_len = len(file_bytes)
        if file_len > MAX_SINGLE_FILE_BYTES:
            raise BundleValidationError(
                f"bundle file {rel_path!r} exceeds 4 MiB limit ({file_len} bytes)"
            )
        total_bytes += file_len
        if total_bytes > MAX_BUNDLE_TOTAL_BYTES:
            raise BundleValidationError(
                f"bundle total size exceeds 16 MiB limit ({total_bytes} bytes)"
            )

        # 常量时间 SHA-256 摘要比对
        actual_sha256 = hashlib.sha256(file_bytes).hexdigest().lower()
        if not hmac.compare_digest(actual_sha256, bf.sha256.lower()):
            raise BundleValidationError(
                f"bundle digest mismatch for {rel_path!r}: expected {bf.sha256}, got {actual_sha256}"
            )

        return str(target_file), file_bytes

    policy_full_path, policy_bytes = read_and_verify(manifest.policy)
    bundle.policy_path = policy_full_path
    bundle.policy = policy_bytes

    for item in manifest.domains:
        p, b = read_and_verify(item)
        bundle.domains[p] = b

    for item in manifest.taxonomies:
        p, b = read_and_verify(item)
        bundle.taxonomies[p] = b

    for item in manifest.standards:
        p, b = read_and_verify(item)
        bundle.standards[p] = b

    logger.info(
        "policy_bundle_loaded_successfully",
        extra={
            "release": bundle.release,
            "root": bundle.root,
            "domains_count": len(bundle.domains),
            "taxonomies_count": len(bundle.taxonomies),
            "standards_count": len(bundle.standards),
            "total_bytes": total_bytes,
        },
    )
    return bundle
