"""策略清单（Policy Bundle）加载与完整性校验测试套件。

对齐 Go internal/configinput/bundle.go 与 services/privacy-engine/internal/service/manifest_test.go.
"""

import hashlib
import json
from pathlib import Path

import pytest

from engine.dynclassification.bundle import (
    BundleValidationError,
    load_bundle,
    parse_manifest_dict,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_load_bundle_success(tmp_path: Path):
    bundle_dir = tmp_path / "release-001"
    bundle_dir.mkdir()
    (bundle_dir / "domains").mkdir()
    (bundle_dir / "taxonomies").mkdir()
    (bundle_dir / "standards").mkdir()

    policy_bytes = b"version: 1\nenabled: true\n"
    domain_bytes = b"name: medical\nrules: []\n"
    tax_bytes = b"name: default\nlevels: []\n"
    std_bytes = b"id: std_01\nname: test\n"

    (bundle_dir / "privacy.yaml").write_bytes(policy_bytes)
    (bundle_dir / "domains" / "medical.yaml").write_bytes(domain_bytes)
    (bundle_dir / "taxonomies" / "default.yaml").write_bytes(tax_bytes)
    (bundle_dir / "standards" / "std.yaml").write_bytes(std_bytes)

    manifest_data = {
        "schema": "privshield-policy-bundle/v1",
        "release": "release-001",
        "policy": {"path": "privacy.yaml", "sha256": _sha256(policy_bytes)},
        "domains": [{"path": "domains/medical.yaml", "sha256": _sha256(domain_bytes)}],
        "taxonomies": [{"path": "taxonomies/default.yaml", "sha256": _sha256(tax_bytes)}],
        "standards": [{"path": "standards/std.yaml", "sha256": _sha256(std_bytes)}],
    }
    manifest_file = bundle_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    bundle = load_bundle(manifest_file)
    assert bundle.release == "release-001"
    assert bundle.root == str(bundle_dir)
    assert bundle.policy == policy_bytes
    assert len(bundle.domains) == 1
    assert len(bundle.taxonomies) == 1
    assert len(bundle.standards) == 1


def test_manifest_schema_rejection():
    raw = {
        "schema": "invalid-schema/v2",
        "release": "rel-1",
        "policy": {"path": "privacy.yaml", "sha256": "0" * 64},
    }
    with pytest.raises(BundleValidationError, match="unsupported policy manifest schema"):
        parse_manifest_dict(raw)


def test_manifest_release_rejection():
    # 首尾有空格
    raw = {
        "schema": "privshield-policy-bundle/v1",
        "release": "  rel-1  ",
        "policy": {"path": "privacy.yaml", "sha256": "0" * 64},
    }
    with pytest.raises(BundleValidationError, match="surrounding whitespace"):
        parse_manifest_dict(raw)

    # 包含控制字符/非可见ASCII
    raw2 = {
        "schema": "privshield-policy-bundle/v1",
        "release": "rel\x00bad",
        "policy": {"path": "privacy.yaml", "sha256": "0" * 64},
    }
    with pytest.raises(BundleValidationError, match="visible ASCII"):
        parse_manifest_dict(raw2)


def test_manifest_policy_path_must_be_privacy_yaml():
    raw = {
        "schema": "privshield-policy-bundle/v1",
        "release": "rel-1",
        "policy": {"path": "other.yaml", "sha256": "0" * 64},
    }
    with pytest.raises(BundleValidationError, match="must be 'privacy.yaml'"):
        parse_manifest_dict(raw)


def test_manifest_group_direct_child_enforcement():
    raw = {
        "schema": "privshield-policy-bundle/v1",
        "release": "rel-1",
        "policy": {"path": "privacy.yaml", "sha256": "0" * 64},
        "domains": [{"path": "domains/sub/deep.yaml", "sha256": "0" * 64}],
    }
    with pytest.raises(BundleValidationError, match="direct children"):
        parse_manifest_dict(raw)


def test_digest_mismatch_rejection(tmp_path: Path):
    bundle_dir = tmp_path / "release-tampered"
    bundle_dir.mkdir()
    policy_bytes = b"content: authentic\n"
    (bundle_dir / "privacy.yaml").write_bytes(policy_bytes)

    # 故意给出错误的 sha256 摘要
    wrong_sha256 = "a" * 64
    manifest_data = {
        "schema": "privshield-policy-bundle/v1",
        "release": "release-tampered",
        "policy": {"path": "privacy.yaml", "sha256": wrong_sha256},
        "domains": [],
        "taxonomies": [],
        "standards": [],
    }
    manifest_file = bundle_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="bundle digest mismatch"):
        load_bundle(manifest_file)


def test_symlink_inside_bundle_rejected(tmp_path: Path):
    bundle_dir = tmp_path / "release-symlink"
    bundle_dir.mkdir()
    (bundle_dir / "domains").mkdir()

    real_file = tmp_path / "outside.yaml"
    outside_bytes = b"name: outside\n"
    real_file.write_bytes(outside_bytes)

    # 在 domains 目录创建指向外部文件的软链接
    symlink_file = bundle_dir / "domains" / "link.yaml"
    symlink_file.symlink_to(real_file)

    manifest_data = {
        "schema": "privshield-policy-bundle/v1",
        "release": "release-symlink",
        "policy": {"path": "privacy.yaml", "sha256": _sha256(b"privacy: true\n")},
        "domains": [{"path": "domains/link.yaml", "sha256": _sha256(outside_bytes)}],
        "taxonomies": [],
        "standards": [],
    }
    (bundle_dir / "privacy.yaml").write_bytes(b"privacy: true\n")
    manifest_file = bundle_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="contains a symlink"):
        load_bundle(manifest_file)
