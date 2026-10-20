# PrivShield Python 隐私引擎服务命令入口
#
# 仅覆盖本服务：源码在 engine/，测试在 tests/，部署资产在 deploy/。
# 全栈编排、生产门禁与 Go 微服务命令由仓库根 Makefile 负责。

.PHONY: help install install-dev test test-unit test-integration test-cov conformance \
        lint format typecheck check proto clean \
        terminator-test terminator-lint terminator-build terminator-check \
        docker-core docker-ml docker-terminator build

VERSION ?= 2.0.0
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
GO ?= go
GRPC_TOOLS_VERSION ?= 1.82.1

help:
	@echo "Available targets:"
	@echo "  install        - 安装核心运行依赖"
	@echo "  install-dev    - 可编辑安装 core + dev extras"
	@echo "  test           - 运行全部 pytest"
	@echo "  test-unit      - 排除 integration / slow 标记"
	@echo "  test-integration - 仅运行 integration 标记"
	@echo "  test-cov       - 覆盖率报告"
	@echo "  conformance    - 跨语言一致性夹具（需要 Go 参考实现）"
	@echo "  lint           - ruff 静态检查"
	@echo "  format         - ruff 自动格式化"
	@echo "  typecheck      - mypy 类型检查"
	@echo "  check          - lint + typecheck + terminator 静态检查"
	@echo "  proto          - 从 proto/privacy.proto 重新生成 gRPC stub"
	@echo "  terminator-test  - TLCP 终结器 Go 测试（独立模块，GOWORK=off）"
	@echo "  terminator-lint  - TLCP 终结器 go vet"
	@echo "  terminator-build - 编译终结器到 bin/privshield-python-terminator"
	@echo "  terminator-check - terminator-lint + terminator-test"
	@echo "  docker-core    - 构建 core 镜像"
	@echo "  docker-ml      - 构建 ml 镜像"
	@echo "  docker-terminator - 构建 TLCP 终结器镜像"
	@echo "  build          - 构建 wheel 与 sdist"

install:
	$(PIP) install -r requirements-core.txt

install-dev:
	$(PIP) install -e ".[dev]"

# ── Quality ──────────────────────────────────────────────────

lint:
	$(PYTHON) -m ruff check engine/ tests/

format:
	$(PYTHON) -m ruff format engine/ tests/
	$(PYTHON) -m ruff check --fix engine/ tests/

typecheck:
	$(PYTHON) -m mypy

check: lint typecheck terminator-lint

# ── Testing ──────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -q --tb=short

test-unit:
	$(PYTHON) -m pytest tests/ -q --tb=short -m "not integration and not slow"

test-integration:
	$(PYTHON) -m pytest tests/ -q --tb=short -m integration

test-cov:
	$(PYTHON) -m pytest tests/ -q --tb=short --cov=engine --cov-report=term-missing \
		-m "not integration and not slow"

# 跨语言夹具由 Go 参考实现与 Python 同时消费；权威副本同步自上游
# contracts/conformance/。
conformance:
	$(PYTHON) -m pytest tests/conformance -q --tb=short

# ── TLCP terminator ──────────────────────────────────────────
# 终结器是独立 Go 模块（terminator/go.mod），依赖的 Go pkg/ 子集已
# vendor 在 terminator/third_party/pkg，可脱离 Go 仓库独立构建。
TERMINATOR_DIR = terminator
TERMINATOR_GOWORK = GOWORK=off

terminator-test:
	cd $(TERMINATOR_DIR) && $(TERMINATOR_GOWORK) $(GO) test -count=1 ./...

terminator-lint:
	cd $(TERMINATOR_DIR) && $(TERMINATOR_GOWORK) $(GO) vet ./...

terminator-build:
	@mkdir -p bin
	cd $(TERMINATOR_DIR) && $(TERMINATOR_GOWORK) CGO_ENABLED=0 $(GO) build \
		-trimpath -ldflags="-s -w -X 'main.Version=$(VERSION)'" \
		-o ../bin/privshield-python-terminator .

terminator-check: terminator-lint terminator-test

# ── Contracts ────────────────────────────────────────────────

# stub 必须由固定版本 grpcio-tools 生成，禁止手改。
# grpcio-tools 固定生成绝对导入（import privacy_pb2），作为 engine 包内模块会导入失败，
# 因此生成后统一改写为相对导入，保证 make proto 可重复执行。
proto:
	$(PIP) install "grpcio-tools==$(GRPC_TOOLS_VERSION)"
	$(PYTHON) -m grpc_tools.protoc -I proto --python_out=engine --grpc_python_out=engine proto/privacy.proto
	sed -i 's/^import privacy_pb2 as privacy__pb2$$/from . import privacy_pb2 as privacy__pb2/' engine/privacy_pb2_grpc.py

# ── Build ────────────────────────────────────────────────────

build:
	$(PYTHON) -m build

docker-core:
	docker build --target core -t privshield-python:$(VERSION) .

docker-ml:
	docker build --target ml -t privshield-python:$(VERSION)-ml .

docker-terminator:
	docker build -f $(TERMINATOR_DIR)/Dockerfile \
		-t privshield-python-terminator:$(VERSION) .

clean:
	rm -rf bin .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
