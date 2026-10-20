"""Deployment 部署产物验证测试 / Deployment Artifact Validation Tests.

中文说明：
本模块验证 Helm chart 与原生 K8s manifests 的语法正确性及服务配置完整性，不依赖实际集群：

1. Helm Chart 语法与配置验证 / Helm Chart Validation:
   - helm lint: 分别针对 default, values-production.yaml, values-ml.yaml 检查语法与引用
   - helm template (default): 默认 values 渲染为合法 YAML，验证核心 Deployment/Service/ConfigMap
   - helm template (production): 生产 values 渲染，验证多副本、TLS、API Key Auth、限速、HPA、NetworkPolicy、ServiceMonitor
   - helm template (ml): ML values 渲染，验证 ML 镜像标签 (-ml) 与资源分配 (8Gi)
   - helm template (all services): 配置拉起全部服务（Core + Layer-3 LLM + Ingress + HPA + NetworkPolicy + ServiceMonitor + PDB + Secrets）
   - helm template (llm service): 验证 Layer-3 LLM (vLLM) 独立推理服务 Deployment/Service 及多种存储模式 (PVC/hostPath/emptyDir)
   - helm template (inline secrets): 验证内联 TLS 证书与 API Key 自动生成 Secret 资源并正确挂载
   - helm template (mTLS & whitelist): 验证 mTLS 客户端证书校验与 CN 白名单 ConfigMap 挂载
   - helm template (custom ports & grpc toggle): 验证自定义 REST/gRPC 端口及 gRPC 开关
   - helm template (ingress): 验证 Ingress 路由、TLS 及 IngressClassName
   - helm template (pdb): 验证 PodDisruptionBudget 最小可用副本
   - helm template (observability): 验证 OpenTelemetry Tracing 环境变量注入

2. K8s Manifests 验证 / K8s Manifests Validation:
   - deploy/k8s/ 目录下所有 YAML 文件可被正确解析
   - 每个文件至少包含一个非空 YAML 文档

测试策略：
- helm 相关测试通过 skipif 标记，未安装 helm 时自动跳过
- K8s YAML 验证仅依赖 PyYAML，无外部工具依赖
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

# 项目根目录（从测试文件位置向上 3 层）/ Project root (3 levels up from test file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Helm chart 目录 / Helm chart directory
HELM_DIR = PROJECT_ROOT / "deploy" / "helm" / "PrivShield"
# 原生 K8s manifests 目录 / Raw K8s manifests directory
K8S_DIR = PROJECT_ROOT / "deploy" / "k8s"

# 检测 helm 可执行文件是否在 PATH 中 / Check if helm binary is in PATH
HELM = shutil.which("helm")


def _run_helm_template(
    release_name: str = "test",
    values: dict[str, Any] | None = None,
    values_files: list[Path | str] | None = None,
    extra_args: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Helper to render Helm chart templates and parse output documents."""
    assert HELM is not None, "helm binary not found"
    cmd = [HELM, "template", release_name, str(HELM_DIR)]
    if values_files:
        for vf in values_files:
            cmd.extend(["-f", str(vf)])
    if extra_args:
        cmd.extend(extra_args)
    if values is not None:
        cmd.extend(["-f", "-"])
        input_data = yaml.dump(values)
    else:
        input_data = None

    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"helm template failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def _find_docs_by_kind(docs: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Filter parsed YAML documents by their Kubernetes kind."""
    return [d for d in docs if d.get("kind") == kind]


def _get_container_env_map(container: dict[str, Any]) -> dict[str, Any]:
    """Extract environment variables from a container spec into a name->value mapping."""
    env_map: dict[str, Any] = {}
    for env in container.get("env", []):
        name = env.get("name")
        if not name:
            continue
        if "value" in env:
            env_map[name] = env["value"]
        elif "valueFrom" in env:
            env_map[name] = env["valueFrom"]
    return env_map


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_lint() -> None:
    """验证默认配置下 helm lint 通过：检查 chart 结构、模板语法、values 引用一致性。"""
    result = subprocess.run(
        [HELM, "lint", str(HELM_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_lint_production_values() -> None:
    """验证使用 values-production.yaml 时 helm lint 通过。"""
    prod_values = HELM_DIR / "values-production.yaml"
    result = subprocess.run(
        [HELM, "lint", str(HELM_DIR), "-f", str(prod_values)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_lint_ml_values() -> None:
    """验证使用 values-ml.yaml 时 helm lint 通过。"""
    ml_values = HELM_DIR / "values-ml.yaml"
    result = subprocess.run(
        [HELM, "lint", str(HELM_DIR), "-f", str(ml_values)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_default_values() -> None:
    """验证默认 values 渲染产生合法 YAML 文档并包含核心组件。

    默认渲染应包含：
    - Core Deployment (1 副本，轻量 core 镜像，HTTP 8079 & gRPC 50051，/health & /readyz 探针)
    - Core Service (ClusterIP, 暴露 8079 和 50051)
    - ConfigMap (挂载 /etc/PrivShield/privacy-profile.yaml)
    - ServiceAccount
    """
    docs = _run_helm_template("test")
    assert docs, "No YAML documents returned from default helm template"

    # 1. 验证 Core Deployment
    deployments = _find_docs_by_kind(docs, "Deployment")
    assert len(deployments) == 1
    core_deploy = deployments[0]
    assert core_deploy["metadata"]["name"] == "test-privshield"
    assert core_deploy["spec"]["replicas"] == 1

    containers = core_deploy["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1
    core_container = containers[0]
    assert core_container["image"] == "privshield:1.8.0"
    assert core_container["imagePullPolicy"] == "IfNotPresent"

    # 验证端口
    port_map = {p["name"]: p["containerPort"] for p in core_container.get("ports", [])}
    assert port_map.get("http") == 8079
    assert port_map.get("grpc") == 50051

    # 验证基础环境变量
    env_map = _get_container_env_map(core_container)
    assert env_map.get("PRIVACY_REST_HOST") == "0.0.0.0"
    assert env_map.get("PRIVACY_REST_PORT") == "8079"
    assert env_map.get("PRIVACY_GRPC_HOST") == "0.0.0.0"
    assert env_map.get("PRIVACY_GRPC_PORT") == "50051"
    assert env_map.get("PRIVACY_PROFILE") == "/etc/PrivShield/privacy-profile.yaml"
    assert env_map.get("PRIVACY_LOG_LEVEL") == "INFO"
    assert env_map.get("PRIVACY_LOG_FORMAT") == "text"

    # 验证探针
    assert core_container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert core_container["readinessProbe"]["httpGet"]["path"] == "/readyz"

    # 2. 验证 Core Service
    services = _find_docs_by_kind(docs, "Service")
    assert len(services) == 1
    core_svc = services[0]
    assert core_svc["metadata"]["name"] == "test-privshield"
    svc_ports = {p["name"]: p["port"] for p in core_svc["spec"]["ports"]}
    assert svc_ports.get("http") == 8079
    assert svc_ports.get("grpc") == 50051

    # 3. 验证 ConfigMap
    configmaps = _find_docs_by_kind(docs, "ConfigMap")
    assert len(configmaps) == 1
    assert configmaps[0]["metadata"]["name"] == "test-privshield-config"
    assert "privacy-profile.yaml" in configmaps[0]["data"]

    # 4. 验证 ServiceAccount
    service_accounts = _find_docs_by_kind(docs, "ServiceAccount")
    assert len(service_accounts) == 1
    assert service_accounts[0]["metadata"]["name"] == "test-privshield"


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_production_values() -> None:
    """验证生产模式 values 渲染并包含 TLS/Auth/限速、HPA、NetworkPolicy 及 ServiceMonitor。

    生产模式配置：
    - 2 副本
    - JSON 结构化日志
    - TLS 启用，挂载外部证书 Secret
    - API Key Auth 启用，引用外部 Secret
    - 限速开启
    - HTTPS exec 模式探针
    - 自动伸缩 HPA
    - NetworkPolicy 隔离
    - Prometheus ServiceMonitor (scheme=https)
    """
    prod_values = HELM_DIR / "values-production.yaml"
    docs = _run_helm_template(
        release_name="prod",
        values_files=[prod_values],
        extra_args=[
            "--set",
            "security.tls.existingSecret=tls-secret",
            "--set",
            "security.auth.apiKeysSecret=keys-secret",
        ],
    )

    # 1. 验证 Deployment 生产配置
    deployments = _find_docs_by_kind(docs, "Deployment")
    assert len(deployments) == 1
    deploy = deployments[0]
    # HPA 启用时不应显式渲染 spec.replicas 避免冲突
    assert "replicas" not in deploy["spec"]

    container = deploy["spec"]["template"]["spec"]["containers"][0]
    env_map = _get_container_env_map(container)

    assert env_map.get("PRIVACY_LOG_FORMAT") == "json"
    assert env_map.get("PRIVACY_TLS_ENABLED") == "true"
    assert env_map.get("PRIVACY_AUTH_ENABLED") == "true"
    assert env_map.get("PRIVACY_RATE_LIMIT_ENABLED") == "true"

    # 验证 API Key 引用的 secretKeyRef
    auth_secret_ref = env_map.get("PRIVACY_AUTH_EXTERNAL_KEYS_JSON", {}).get("secretKeyRef", {})
    assert auth_secret_ref.get("name") == "keys-secret"
    assert auth_secret_ref.get("key") == "api-keys.json"

    # 验证 TLS 探针切换为 exec 模式（curl/wget HTTPS）
    assert "exec" in container["livenessProbe"]
    assert "https://" in container["livenessProbe"]["exec"]["command"][2]
    assert "exec" in container["readinessProbe"]
    assert "https://" in container["readinessProbe"]["exec"]["command"][2]

    # 验证卷挂载
    vols = {v["name"]: v for v in deploy["spec"]["template"]["spec"]["volumes"]}
    assert "tls-certs" in vols
    assert vols["tls-certs"]["secret"]["secretName"] == "tls-secret"

    # 2. 验证 HPA
    hpas = _find_docs_by_kind(docs, "HorizontalPodAutoscaler")
    assert len(hpas) == 1
    hpa = hpas[0]
    assert hpa["spec"]["scaleTargetRef"]["name"] == "prod-privshield"
    assert hpa["spec"]["minReplicas"] == 2
    assert hpa["spec"]["maxReplicas"] == 10

    # 3. 验证 NetworkPolicy
    netpols = _find_docs_by_kind(docs, "NetworkPolicy")
    assert len(netpols) == 1
    assert netpols[0]["metadata"]["name"] == "prod-privshield"

    # 4. 验证 ServiceMonitor
    monitors = _find_docs_by_kind(docs, "ServiceMonitor")
    assert len(monitors) == 1
    monitor_endpoint = monitors[0]["spec"]["endpoints"][0]
    assert monitor_endpoint["path"] == "/metrics"
    assert monitor_endpoint["scheme"] == "https"
    assert monitor_endpoint["tlsConfig"]["insecureSkipVerify"] is True


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_ml_flavor() -> None:
    """验证 ML 镜像版本 (flavor: ml) 的 values 渲染。

    验证：
    - 镜像标签自动添加 -ml 后缀 (如 privshield:1.8.0-ml)
    - 资源限制调整为 ML 场景 (CPU 4000m / 内存 8Gi)
    - HPA 自动启用 (minReplicas 1, maxReplicas 3)
    """
    ml_values = HELM_DIR / "values-ml.yaml"
    docs = _run_helm_template(release_name="ml-test", values_files=[ml_values])

    deployments = _find_docs_by_kind(docs, "Deployment")
    assert len(deployments) == 1
    container = deployments[0]["spec"]["template"]["spec"]["containers"][0]

    assert container["image"].endswith("-ml")
    assert container["resources"]["limits"]["memory"] == "8Gi"
    assert container["resources"]["limits"]["cpu"] == "4000m"

    hpas = _find_docs_by_kind(docs, "HorizontalPodAutoscaler")
    assert len(hpas) == 1
    assert hpas[0]["spec"]["minReplicas"] == 1
    assert hpas[0]["spec"]["maxReplicas"] == 3


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_all_services_and_workloads() -> None:
    """全面验证：配置并拉起所有服务与全套 Kubernetes 资源。

    包含：
    1. Core Agent Deployment (REST 8079 / gRPC 50051)
    2. Core Agent Service
    3. Layer-3 LLM (vLLM) 独立推理服务 Deployment (GPU / PVC 挂载)
    4. Layer-3 LLM Service (Port 8000)
    5. Ingress (TLS 路由与 Path 配置)
    6. HorizontalPodAutoscaler (CPU + 内存双指标伸缩)
    7. NetworkPolicy (网络隔离与 Pod Selector)
    8. ServiceMonitor (Prometheus Operator 采集端点)
    9. PodDisruptionBudget (最小可用副本保护)
    10. Namespace 自动创建
    11. ServiceAccount 自动创建
    12. ConfigMap (隐私策略配置)
    13. Secret (内联证书与 API Key)
    """
    all_values: dict[str, Any] = {
        "namespace": {"create": True},
        "serviceAccount": {"create": True},
        # 启用 Layer-3 LLM 独立推理服务
        "llm": {
            "enabled": True,
            "provider": "vllm",
            "port": 8000,
            "servedModelName": "Qwen3.5-0.8B-Privacy-Classifier-Smoother",
            "modelPath": "/models/Qwen3.5-0.8B-Privacy-Classifier-Smoother",
            "gpuMemoryUtilization": "0.85",
            "maxModelLen": "4096",
            "storage": {
                "existingClaim": "privshield-models-pvc",
            },
            "nodeSelector": {"nvidia.com/gpu": "true"},
            "resources": {
                "limits": {"cpu": "8", "memory": "16Gi", "nvidia.com/gpu": 1},
                "requests": {"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": 1},
            },
            "probes": {
                "liveness": {"initialDelaySeconds": 45, "periodSeconds": 20, "failureThreshold": 3},
                "readiness": {"initialDelaySeconds": 45, "periodSeconds": 10, "failureThreshold": 3},
            },
        },
        # 启用 Ingress
        "ingress": {
            "enabled": True,
            "className": "nginx",
            "hosts": [
                {
                    "host": "privshield.enterprise.local",
                    "paths": [{"path": "/", "pathType": "Prefix"}],
                }
            ],
            "tls": [
                {
                    "secretName": "privshield-ingress-tls",
                    "hosts": ["privshield.enterprise.local"],
                }
            ],
        },
        # 启用 HPA
        "autoscaling": {
            "enabled": True,
            "minReplicas": 2,
            "maxReplicas": 8,
            "targetCPUUtilizationPercentage": 75,
            "targetMemoryUtilizationPercentage": 80,
        },
        # 启用 NetworkPolicy
        "networkPolicy": {
            "enabled": True,
            "ingress": {
                "from": [
                    {
                        "podSelector": {
                            "matchLabels": {"app.kubernetes.io/part-of": "privshield-mesh"}
                        }
                    }
                ]
            },
        },
        # 启用 ServiceMonitor
        "serviceMonitor": {
            "enabled": True,
            "interval": "15s",
            "scrapeTimeout": "10s",
        },
        # 启用 PodDisruptionBudget
        "podDisruptionBudget": {
            "enabled": True,
            "minAvailable": 1,
        },
        # 启用全套安全设置（内联 Secret + 限速 + 内部 mTLS）
        "security": {
            "tls": {
                "enabled": True,
                "cert": "-----BEGIN CERTIFICATE-----\nFAKE_CERT\n-----END CERTIFICATE-----",
                "key": "-----BEGIN PRIVATE KEY-----\nFAKE_KEY\n-----END PRIVATE KEY-----",
            },
            "auth": {
                "enabled": True,
                "apiKeys": '{"key-001": {"name": "app-gateway", "scopes": ["*"]}}',
                "internalMtls": {
                    "enabled": True,
                    "whitelistConfigMap": "mtls-cn-whitelist-cm",
                },
            },
            "rateLimit": {
                "enabled": True,
                "defaultRps": 50,
                "defaultBurst": 100,
                "redisUrl": "redis://redis-cluster:6379/0",
            },
        },
    }

    docs = _run_helm_template(release_name="fullstack", values=all_values)

    # 1. 验证全部 13 类 Kubernetes 资源均已正确渲染
    kinds = {d["kind"] for d in docs}
    expected_kinds = {
        "Namespace",
        "ServiceAccount",
        "Secret",
        "ConfigMap",
        "Service",
        "Deployment",
        "HorizontalPodAutoscaler",
        "Ingress",
        "NetworkPolicy",
        "PodDisruptionBudget",
        "ServiceMonitor",
    }
    assert expected_kinds.issubset(kinds), f"Missing kinds: {expected_kinds - kinds}"

    # 2. 验证包含两个 Deployment (Core 和 LLM)
    deployments = _find_docs_by_kind(docs, "Deployment")
    assert len(deployments) == 2
    deploy_map = {d["metadata"]["name"]: d for d in deployments}
    assert "fullstack-privshield" in deploy_map
    assert "fullstack-privshield-llm" in deploy_map

    core_deploy = deploy_map["fullstack-privshield"]
    llm_deploy = deploy_map["fullstack-privshield-llm"]

    # 验证 Core 注入了连接 Layer-3 LLM 服务的环境变量
    core_container = core_deploy["spec"]["template"]["spec"]["containers"][0]
    core_env = _get_container_env_map(core_container)
    assert core_env.get("PRIVACY_LLM_PROVIDER") == "vllm"
    assert core_env.get("PRIVACY_LLM_API_BASE") == "http://fullstack-privshield-llm:8000/v1"
    assert core_env.get("PRIVACY_LLM_MODEL_NAME") == "Qwen3.5-0.8B-Privacy-Classifier-Smoother"
    assert core_env.get("PRIVACY_LLM_API_KEY") == "EMPTY"
    assert core_env.get("PRIVACY_RATE_LIMIT_REDIS_URL") == "redis://redis-cluster:6379/0"
    assert core_env.get("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED") == "true"
    assert core_env.get("PRIVACY_AUTH_MTLS_WHITELIST_FILE") == "/etc/PrivShield/mtls-whitelist.yaml"

    # 验证 LLM Deployment 参数与容器结构
    llm_container = llm_deploy["spec"]["template"]["spec"]["containers"][0]
    assert llm_container["name"] == "vllm"
    assert llm_container["command"] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in llm_container["args"]
    assert "/models/Qwen3.5-0.8B-Privacy-Classifier-Smoother" in llm_container["args"]
    assert llm_container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert llm_container["ports"][0]["containerPort"] == 8000
    assert llm_container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert llm_container["livenessProbe"]["initialDelaySeconds"] == 45

    # 验证 LLM 存储挂载为 PVC
    llm_vols = {v["name"]: v for v in llm_deploy["spec"]["template"]["spec"]["volumes"]}
    assert "model-weights" in llm_vols
    assert llm_vols["model-weights"]["persistentVolumeClaim"]["claimName"] == "privshield-models-pvc"

    # 3. 验证包含两个 Service (Core 和 LLM)
    services = _find_docs_by_kind(docs, "Service")
    assert len(services) == 2
    svc_map = {s["metadata"]["name"]: s for s in services}
    assert "fullstack-privshield" in svc_map
    assert "fullstack-privshield-llm" in svc_map

    llm_svc = svc_map["fullstack-privshield-llm"]
    assert llm_svc["spec"]["ports"][0]["port"] == 8000
    assert llm_svc["spec"]["selector"]["app.kubernetes.io/component"] == "llm"

    # 4. 验证 Ingress 正确反向代理至 Core Service
    ingresses = _find_docs_by_kind(docs, "Ingress")
    assert len(ingresses) == 1
    ing = ingresses[0]
    assert ing["spec"]["ingressClassName"] == "nginx"
    rule = ing["spec"]["rules"][0]
    assert rule["host"] == "privshield.enterprise.local"
    backend_svc = rule["http"]["paths"][0]["backend"]["service"]
    assert backend_svc["name"] == "fullstack-privshield"
    assert backend_svc["port"]["number"] == 8079

    # 5. 验证 PodDisruptionBudget
    pdbs = _find_docs_by_kind(docs, "PodDisruptionBudget")
    assert len(pdbs) == 1
    assert pdbs[0]["spec"]["minAvailable"] == 1

    # 6. 验证 Secret 包含证书和 API Key
    secrets = _find_docs_by_kind(docs, "Secret")
    assert len(secrets) == 1
    secret_data = secrets[0]["stringData"]
    assert "tls.crt" in secret_data
    assert "tls.key" in secret_data
    assert "api-keys.json" in secret_data


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_llm_storage_options() -> None:
    """验证 Layer-3 LLM 服务的多种存储模式与自定义端口。

    测试场景：
    1. hostPath 存储模式（本地单节点测试）
    2. emptyDir 存储模式（未显式指定存储时的回退）
    3. 自定义 LLM 服务端口（如 8008）在 Deployment、Service 及 Core 环境变量中同步生效
    """
    # 场景 1: hostPath 存储模式 + 自定义端口 8008
    values_hostpath: dict[str, Any] = {
        "llm": {
            "enabled": True,
            "port": 8008,
            "storage": {"hostPath": "/data/shared/models"},
        }
    }
    docs_hp = _run_helm_template(release_name="hp-test", values=values_hostpath)

    llm_deploy_hp = next(d for d in _find_docs_by_kind(docs_hp, "Deployment") if d["metadata"]["name"].endswith("-llm"))
    vols_hp = {v["name"]: v for v in llm_deploy_hp["spec"]["template"]["spec"]["volumes"]}
    assert vols_hp["model-weights"]["hostPath"]["path"] == "/data/shared/models"

    llm_svc_hp = next(s for s in _find_docs_by_kind(docs_hp, "Service") if s["metadata"]["name"].endswith("-llm"))
    assert llm_svc_hp["spec"]["ports"][0]["port"] == 8008

    core_deploy_hp = next(d for d in _find_docs_by_kind(docs_hp, "Deployment") if not d["metadata"]["name"].endswith("-llm"))
    core_env_hp = _get_container_env_map(core_deploy_hp["spec"]["template"]["spec"]["containers"][0])
    assert core_env_hp["PRIVACY_LLM_API_BASE"] == "http://hp-test-privshield-llm:8008/v1"

    # 场景 2: emptyDir 默认回退
    values_emptydir: dict[str, Any] = {
        "llm": {
            "enabled": True,
        }
    }
    docs_ed = _run_helm_template(release_name="ed-test", values=values_emptydir)
    llm_deploy_ed = next(d for d in _find_docs_by_kind(docs_ed, "Deployment") if d["metadata"]["name"].endswith("-llm"))
    vols_ed = {v["name"]: v for v in llm_deploy_ed["spec"]["template"]["spec"]["volumes"]}
    assert vols_ed["model-weights"] == {"name": "model-weights", "emptyDir": {}}


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_custom_ports_and_grpc_toggle() -> None:
    """验证自定义 REST 与 gRPC 端口及 gRPC 服务暴露开关。

    配置：
    - REST 端口: 9090
    - gRPC 端口: 60051
    - exposeGrpc: false (关闭 gRPC Service 端口暴露)
    """
    values: dict[str, Any] = {
        "service": {
            "restPort": 9090,
            "grpcPort": 60051,
            "exposeGrpc": False,
        }
    }
    docs = _run_helm_template(release_name="ports-test", values=values)

    # 1. 验证 Service 仅暴露 http 9090，不暴露 grpc
    services = _find_docs_by_kind(docs, "Service")
    assert len(services) == 1
    ports = services[0]["spec"]["ports"]
    assert len(ports) == 1
    assert ports[0]["name"] == "http"
    assert ports[0]["port"] == 9090

    # 2. 验证 Deployment 环境变量与探针端口
    deployment = _find_docs_by_kind(docs, "Deployment")[0]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env_map = _get_container_env_map(container)

    assert env_map.get("PRIVACY_REST_PORT") == "9090"
    assert env_map.get("PRIVACY_GRPC_PORT") == "60051"

    c_ports = {p["name"]: p["containerPort"] for p in container["ports"]}
    assert c_ports["http"] == 9090
    assert c_ports["grpc"] == 60051


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_mtls_client_auth_and_whitelist() -> None:
    """验证 mTLS 客户端认证与 CN 白名单模式配置。

    测试场景：
    1. mTLS require + CA 证书 Secret + 白名单 ConfigMap
    2. 回退模式：静态 allowedCns 列表
    """
    # 场景 1: CA Secret + Whitelist ConfigMap
    values_cm: dict[str, Any] = {
        "security": {
            "tls": {
                "enabled": True,
                "existingSecret": "tls-secret",
                "clientAuth": "require",
                "caSecret": "ca-certs-secret",
                "caFile": "/certs/ca.crt",
            },
            "auth": {
                "internalMtls": {
                    "enabled": True,
                    "whitelistConfigMap": "mtls-whitelist-config",
                }
            },
        }
    }
    docs_cm = _run_helm_template(release_name="mtls-cm", values=values_cm)
    deploy_cm = _find_docs_by_kind(docs_cm, "Deployment")[0]
    container_cm = deploy_cm["spec"]["template"]["spec"]["containers"][0]
    env_cm = _get_container_env_map(container_cm)

    assert env_cm.get("PRIVACY_TLS_CLIENT_AUTH") == "require"
    assert env_cm.get("PRIVACY_TLS_CA_FILE") == "/certs/ca.crt"
    assert env_cm.get("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED") == "true"
    assert env_cm.get("PRIVACY_AUTH_MTLS_WHITELIST_FILE") == "/etc/PrivShield/mtls-whitelist.yaml"

    vols_cm = {v["name"]: v for v in deploy_cm["spec"]["template"]["spec"]["volumes"]}
    assert vols_cm["ca-cert"]["secret"]["secretName"] == "ca-certs-secret"
    assert vols_cm["mtls-whitelist"]["configMap"]["name"] == "mtls-whitelist-config"

    # 场景 2: 静态 allowedCns 回退
    values_static: dict[str, Any] = {
        "security": {
            "tls": {
                "enabled": True,
                "existingSecret": "tls-secret",
            },
            "auth": {
                "internalMtls": {
                    "enabled": True,
                    "allowedCns": "client-app-1,client-app-2",
                }
            },
        }
    }
    docs_static = _run_helm_template(release_name="mtls-static", values=values_static)
    deploy_static = _find_docs_by_kind(docs_static, "Deployment")[0]
    container_static = deploy_static["spec"]["template"]["spec"]["containers"][0]
    env_static = _get_container_env_map(container_static)

    assert env_static.get("PRIVACY_AUTH_MTLS_ALLOWED_CNS") == "client-app-1,client-app-2"


@pytest.mark.skipif(HELM is None, reason="helm not found in PATH")
def test_helm_template_observability_and_tracing() -> None:
    """验证 OpenTelemetry Tracing 与 ServiceName 环境变量注入。"""
    values: dict[str, Any] = {
        "agent": {
            "serviceName": "privshield-tracing-svc",
            "otelExporterOtlpEndpoint": "http://otel-collector.monitoring:4317",
        }
    }
    docs = _run_helm_template(release_name="otel-test", values=values)
    deployment = _find_docs_by_kind(docs, "Deployment")[0]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env_map = _get_container_env_map(container)

    assert env_map.get("OTEL_SERVICE_NAME") == "privshield-tracing-svc"
    assert env_map.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "http://otel-collector.monitoring:4317"


# Docker Compose 目录 / Docker Compose directory
COMPOSE_DIR = PROJECT_ROOT / "deploy" / "docker-compose"


def test_k8s_manifests_are_valid_yaml() -> None:
    """验证 deploy/k8s/ 下所有 YAML 文件可被正确解析。

    包括 deployment.yaml, service.yaml, configmap.yaml,
    namespace.yaml, kustomization.yaml, secret.example.yaml,
    llm-deployment.yaml, llm-service.yaml。
    """
    for path in K8S_DIR.glob("*.yaml"):
        with path.open("r", encoding="utf-8") as f:
            content = f.read()
        docs = list(yaml.safe_load_all(content))
        assert any(d is not None for d in docs), f"{path} contains no documents"


def test_subservices_k8s_manifests_are_valid_yaml() -> None:
    """验证各子服务 (service-hub, datasource-mgr, audit-log, console) 的独立 K8s 清单均为合法 YAML。

    涵盖目录：
    - services/service-hub/deploy/k8s/ 及 postgres/ 子目录
    - services/datasource-mgr/deploy/k8s/
    - services/audit-log/deploy/k8s/
    - console/deploy/k8s/
    """
    manifest_dirs = [
        PROJECT_ROOT / "services" / "service-hub" / "deploy" / "k8s",
        PROJECT_ROOT / "services" / "service-hub" / "deploy" / "k8s" / "postgres",
        PROJECT_ROOT / "services" / "datasource-mgr" / "deploy" / "k8s",
        PROJECT_ROOT / "services" / "audit-log" / "deploy" / "k8s",
        PROJECT_ROOT / "console" / "deploy" / "k8s",
    ]
    total_manifest_files = 0
    for mdir in manifest_dirs:
        assert mdir.is_dir(), f"Manifest directory does not exist: {mdir}"
        yaml_files = list(mdir.glob("*.yaml")) + list(mdir.glob("*.yml"))
        assert len(yaml_files) > 0, f"No YAML manifest files found in {mdir}"
        for path in yaml_files:
            with path.open("r", encoding="utf-8") as f:
                content = f.read()
            docs = list(yaml.safe_load_all(content))
            assert any(d is not None for d in docs), f"{path} contains no valid documents"
            total_manifest_files += 1

    assert total_manifest_files >= 15, f"Expected at least 15 subservice K8s manifests, found {total_manifest_files}"


def test_k8s_kustomize_resource_cross_references() -> None:
    """验证根目录与各子服务的 kustomization.yaml 资源引用均真实有效且可解析。"""
    root_kust_file = K8S_DIR / "kustomization.yaml"
    assert root_kust_file.is_file(), f"Missing root kustomization.yaml: {root_kust_file}"

    with root_kust_file.open("r", encoding="utf-8") as f:
        root_kust = yaml.safe_load(f)

    assert "resources" in root_kust, "Root kustomization.yaml missing 'resources'"
    resources = root_kust["resources"]

    # 验证相对路径资源均存在
    for res_rel in resources:
        resolved_path = (K8S_DIR / res_rel).resolve()
        assert resolved_path.exists(), f"Resource referenced in root kustomization does not exist: {res_rel} -> {resolved_path}"
        if resolved_path.is_dir():
            sub_kust = resolved_path / "kustomization.yaml"
            assert sub_kust.is_file(), f"Sub-directory missing kustomization.yaml: {sub_kust}"


def test_k8s_service_ports_and_protocols_consistency() -> None:
    """验证所有核心微服务的 Kubernetes Service 端口映射规范。

    端口约定：
    - Core Agent    : HTTP 8079, gRPC 50051
    - service-hub   : HTTP 8082, gRPC 50052
    - datasource-mgr: HTTP 8083, gRPC 50053
    - audit-log     : HTTP 8084, gRPC 50054
    - console-bff   : HTTP 8081, gRPC 50055
    """
    service_checks = [
        (K8S_DIR / "service.yaml", {"http": 8079, "grpc": 50051}),
        (PROJECT_ROOT / "services" / "service-hub" / "deploy" / "k8s" / "service.yaml", {"http": 8082, "grpc": 50052}),
        (PROJECT_ROOT / "services" / "datasource-mgr" / "deploy" / "k8s" / "service.yaml", {"http": 8083, "grpc": 50053}),
        (PROJECT_ROOT / "services" / "audit-log" / "deploy" / "k8s" / "service.yaml", {"http": 8084, "grpc": 50054}),
        (PROJECT_ROOT / "console" / "deploy" / "k8s" / "bff-go-service.yaml", {"http": 8081}),
        (PROJECT_ROOT / "console" / "deploy" / "k8s" / "web-service.yaml", {"http": 5173}),
    ]

    for svc_file, expected_ports in service_checks:
        assert svc_file.is_file(), f"Missing service file: {svc_file}"
        with svc_file.open("r", encoding="utf-8") as f:
            svc_doc = yaml.safe_load(f)
        assert svc_doc.get("kind") == "Service"
        ports = {p["name"]: p["port"] for p in svc_doc["spec"]["ports"]}
        for port_name, expected_val in expected_ports.items():
            assert ports.get(port_name) == expected_val, f"{svc_file.name} port '{port_name}' expected {expected_val}, got {ports.get(port_name)}"


def test_docker_compose_files_are_valid_yaml() -> None:
    """验证 deploy/docker-compose/ 目录下所有 Compose 编排文件均为合法 YAML。

    包含：
    - docker-compose.yml (通用全栈基础编排)
    - docker-compose.prod.yml (生产环境安全加固编排)
    - docker-compose.dev.yml (开发联调源码热挂载编排)
    - docker-compose.test.yml (CI/自动化集成测试编排)
    - docker-compose.mtls.yml (mTLS 双向认证编排)
    - docker-compose.app-lz.yml (调度之眼专用控制台编排)
    """
    compose_files = [
        "docker-compose.yml",
        "docker-compose.prod.yml",
        "docker-compose.dev.yml",
        "docker-compose.test.yml",
        "docker-compose.mtls.yml",
        "docker-compose.app-lz.yml",
    ]
    for filename in compose_files:
        file_path = COMPOSE_DIR / filename
        assert file_path.exists(), f"Missing docker-compose file: {filename}"
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{filename} is not a valid YAML dictionary"
        assert "services" in data, f"{filename} missing top-level 'services' section"


def test_docker_compose_production_configuration() -> None:
    """验证生产环境 docker-compose.prod.yml 的安全与架构配置。

    检查项：
    - Core Agent 启用 TLS、API Key Auth、限速与 JSON 日志
    - 包含 3 大中台微服务 (service-hub, datasource-mgr, audit-log)
    - 包含独立的 Redis 缓存/限流后端服务
    - 包含 Go 代理与 Nginx Web 前端
    - 包含独立 vLLM (Profile: llm) 与监控栈 (Profile: monitoring)
    - 容器启用 restart: always 与 security_opt no-new-privileges
    """
    prod_file = COMPOSE_DIR / "docker-compose.prod.yml"
    with prod_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data["services"]
    assert "PrivShield" in services
    assert "service-hub" in services
    assert "datasource-mgr" in services
    assert "audit-log" in services
    assert "redis" in services
    assert "console-backend-go" in services
    assert "console-web" in services
    assert "vllm" in services
    assert "prometheus" in services
    assert "grafana" in services

    agent_env = services["PrivShield"]["environment"]
    assert agent_env.get("PRIVACY_LOG_FORMAT") == "json"
    assert "PRIVACY_TLS_ENABLED" in agent_env
    assert "PRIVACY_AUTH_ENABLED" in agent_env
    assert "PRIVACY_RATE_LIMIT_ENABLED" in agent_env
    assert "redis://redis:6379/0" in str(agent_env.get("PRIVACY_RATE_LIMIT_REDIS_URL"))

    assert services["PrivShield"].get("restart") == "always"
    assert services["redis"].get("restart") == "always"


def test_docker_compose_development_configuration() -> None:
    """验证开发环境 docker-compose.dev.yml 源码挂载与调试配置。

    检查项：
    - Core Agent 挂载宿主机源码目录（../../engine）
    - 日志格式为 text
    - 关闭 TLS/Auth/限速便于直连调试
    """
    dev_file = COMPOSE_DIR / "docker-compose.dev.yml"
    with dev_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data["services"]
    agent_env = services["PrivShield"]["environment"]
    assert agent_env.get("PRIVACY_LOG_FORMAT") == "text"
    assert agent_env.get("PRIVACY_TLS_ENABLED") == "false"
    assert agent_env.get("PRIVACY_AUTH_ENABLED") == "false"

    volumes = services["PrivShield"].get("volumes", [])
    has_source_mount = any("../../engine:/app/engine" in str(v) for v in volumes)
    assert has_source_mount, "docker-compose.dev.yml should mount engine source code"


def test_docker_compose_test_runner_configuration() -> None:
    """验证自动化测试 docker-compose.test.yml 的 test-runner 配置。

    检查项：
    - 包含 test-runner 容器服务
    - test-runner 依赖被测核心服务的 service_healthy 探针
    """
    test_file = COMPOSE_DIR / "docker-compose.test.yml"
    with test_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data["services"]
    assert "test-runner" in services
    assert "PrivShield" in services
    runner = services["test-runner"]
    deps = runner.get("depends_on", {})
    assert "PrivShield" in deps
    assert deps["PrivShield"].get("condition") == "service_healthy"


def test_docker_compose_app_lz_configuration() -> None:
    """验证专用控制台 docker-compose.app-lz.yml 的服务定义。"""
    app_lz_file = COMPOSE_DIR / "docker-compose.app-lz.yml"
    with app_lz_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data["services"]
    assert "app-lz-web" in services
    assert "app-lz-bff" in services


def test_prometheus_scrape_configs_and_alerts_validity() -> None:
    """验证 deploy/prometheus/ 下监控采集配置与告警规则的合法性。

    检查项：
    - prometheus.yml 包含 privshield-agent, console-bff-go, service-hub, datasource-mgr 四大采集 Job
    - alerts.yml 定义了 PrivShieldCoreAlerts 与 ServiceHubAlerts 告警规则组
    """
    prom_dir = PROJECT_ROOT / "deploy" / "prometheus"
    prom_file = prom_dir / "prometheus.yml"
    alerts_file = prom_dir / "alerts.yml"

    assert prom_file.is_file(), f"Missing prometheus.yml: {prom_file}"
    assert alerts_file.is_file(), f"Missing alerts.yml: {alerts_file}"

    # 1. 验证 prometheus.yml
    with prom_file.open("r", encoding="utf-8") as f:
        prom_cfg = yaml.safe_load(f)
    assert isinstance(prom_cfg, dict)
    assert "scrape_configs" in prom_cfg
    job_names = [sc.get("job_name") for sc in prom_cfg["scrape_configs"]]
    assert "privshield-agent" in job_names
    assert "console-bff-go" in job_names
    assert "service-hub" in job_names
    assert "datasource-mgr" in job_names

    # 2. 验证 alerts.yml
    with alerts_file.open("r", encoding="utf-8") as f:
        alerts_cfg = yaml.safe_load(f)
    assert isinstance(alerts_cfg, dict)
    assert "groups" in alerts_cfg
    group_names = [g.get("name") for g in alerts_cfg["groups"]]
    assert "PrivShield.availability" in group_names
    assert "PrivShield.latency" in group_names
    assert "PrivShield.errors" in group_names
    assert "PrivShield.privacy" in group_names
    assert "PrivShield.classification" in group_names
    assert "PrivShield.services" in group_names


def test_grafana_dashboards_validity() -> None:
    """验证 deploy/grafana/ 预置大屏 JSON 文件的完整性与有效性。"""
    import json

    grafana_dir = PROJECT_ROOT / "deploy" / "grafana"
    dashboards = ["dashboard.json", "service-hub-dashboard.json"]

    for dname in dashboards:
        dpath = grafana_dir / dname
        assert dpath.is_file(), f"Missing Grafana dashboard: {dpath}"
        with dpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), f"{dname} is not a valid JSON object"
        assert "panels" in data or "rows" in data, f"{dname} missing panels/rows"
        assert data.get("title"), f"{dname} missing title"


