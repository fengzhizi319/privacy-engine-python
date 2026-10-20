"""基于 ONNX Runtime 的本地轻量级命名实体识别（Small-NER）引擎。

中文说明：
提供纯 Python 实现的 BERT Tokenizer 以及高效的 BIO 标记解析器。
支持 ONNX Runtime 和 ModelScope 两种推理后端，均具备延迟加载与自动降级能力。

本模块是三层分类漏斗的第二层（Layer-2），在规则引擎（Layer-1）之后执行。
当规则引擎无法确定分类结果（置信度不足或等级 <= L3）时，NER 引擎通过
识别文本中的医疗实体（疾病、药物、手术、身体部位等）来辅助分类决策。

架构设计：
- SimpleChineseBertTokenizer：纯 Python 分词器，无第三方依赖
- ONNXSmallNerEngine：ONNX Runtime 推理后端（推荐，轻量高效）
- TensorRTSmallNerEngine：NVIDIA GPU TensorRT 加速后端
- ModelScopeSmallNerEngine：ModelScope 管道推理后端（兼容，需 PyTorch）

降级策略：
- onnxruntime 未安装或模型文件不存在 → 回退到 ModelScope 引擎
- modelscope 未安装或 PyTorch 不可用 → 回退到 NoOpSmallNerEngine（空实现）

English Description:
Local lightweight Named Entity Recognition (Small-NER) engine based on ONNX Runtime.
Provides a pure-Python BERT Tokenizer and an efficient BIO tag parser.
Supports both ONNX Runtime and ModelScope inference backends with lazy-loading
and graceful degradation capabilities.

===================================================================================
              NER 引擎推理流程 / NER Engine Inference Flow
===================================================================================

  engine.extract(text)
    │
    ├─① 文本预处理
    │   ├─ 短文本 (<=120字): 直接推理
    │   └─ 长文本 (>120字): 智能分句 (句号/分号/换行) + 滑动窗口 (120字+20字重叠)
    │
    ├─② Tokenizer 编码
    │   └─ SimpleChineseBertTokenizer.encode(text, max_len=128)
    │       → [CLS] 单字切分 token1...tokenN [SEP] [PAD]...
    │       → (input_ids, attention_mask, token_type_ids)
    │
    ├─③ 模型推理 (ONNX / TensorRT / ModelScope)
    │   └─ BERT 前向传播 → logits (seq_len, num_labels)
    │
    ├─④ 序列标注解码
    │   ├─ ONNX/TensorRT: BIO 标签解析 (B-xxx/I-xxx → 实体)
    │   └─ ModelScope: 直接返回实体列表
    │
    ├─⑤ 标签映射
    │   └─ 原始标签 → 标准标签 (dis→MEDICAL_DISEASE, dru→MEDICATION...)
    │
    └─⑥ 返回 list[dict] : [{"label": "MEDICAL_DISEASE", "text": "...", "confidence": 0.9}]

  支持的实体类型 (CMeEE 标准):
  dis=疾病, sym=症状, mic=微生物, dru=药物, pro=手术,
  bod=身体部位, ite=检查项目, dep=科室, equ=医疗设备
===================================================================================
"""

# 启用延迟注解求值，允许在类型提示中引用尚未定义的类名
from __future__ import annotations

# 导入操作系统接口，用于文件路径拼接和存在性检查
import os
# 导入时间模块，用于测量推理耗时
import time
# 导入类型注解工具：Any 用于通用类型，cast 用于类型断言
from typing import Any, cast

# 导入结构化日志工厂函数
from ..observability.logging_config import get_logger
# 导入 Prometheus 指标：
# - CLASSIFICATION_NER_DURATION：NER 推理延迟直方图（按引擎标签）
# - CLASSIFICATION_NER_TOTAL：NER 调用次数计数器（按状态标签）
from ..observability.metrics import (
    CLASSIFICATION_NER_DURATION,
    CLASSIFICATION_NER_TOTAL,
    NER_INFERENCE_SECONDS,
)
# 导入 Small-NER 引擎抽象基类
from .base import SmallNerEngine

# 创建模块级结构化日志器
logger = get_logger(__name__)


def observe_ner_inference(duration: float, device: str, batch_size: int = 1) -> None:
    """记录 Go 同名指标 privshield_ner_inference_seconds{device,batch_size}。

    每个 extract() 调用按单条文本推理，批大小固定为 1；
    device 取值收敛到 Go 的 "cpu" / "cuda" 词表（MLX 的 "mps" 为 Python 超集）。
    """
    NER_INFERENCE_SECONDS.labels(device=device, batch_size=str(batch_size)).observe(duration)

# 内置 NER 原始标签→标准标签映射（CMeEE 医疗 NER 默认）。
# 可通过 taxonomy YAML 的 ner_label_mapping 字段覆盖。
DEFAULT_NER_LABEL_MAPPING: dict[str, str] = {
    "dis": "MEDICAL_DISEASE",   # 疾病
    "sym": "MEDICAL_DISEASE",   # 症状（归入疾病大类）
    "mic": "MEDICAL_DISEASE",   # 微生物（归入疾病大类）
    "dru": "MEDICATION",        # 药物
    "pro": "SURGERY",           # 手术/操作
    "bod": "BODY_PART",         # 身体部位
    "ite": "EXAMINATION",       # 检查项目
    "dep": "DEPARTMENT",        # 科室
    "equ": "EQUIPMENT",         # 医疗设备
    "GENE": "GENOMIC_HINT",     # 基因（ModelScope 特有）
}


class SimpleChineseBertTokenizer:
    """纯 Python 实现的轻量级中文 BERT 分词器 / Lightweight Chinese BERT Tokenizer.

    设计目标：
    - 零第三方依赖：不依赖 transformers / tokenizers / jieba 等库
    - 毫秒级分词：简单的字符级切分 + 词表查找
    - 大小写折叠：兼容医学缩写（如 HIV → hiv）

    分词策略：
    - 中文：逐字切分（每个汉字为一个 token）
    - 英文：逐字母切分 + 大小写折叠
    - 未登录词：映射为 [UNK]

    English Description:
    A pure-Python lightweight Chinese BERT tokenizer with no third-party tokenization
    library dependencies (e.g. transformers / tokenizers), ensuring millisecond-level
    inference efficiency and compatibility.
    """

    def __init__(self, vocab_path: str):
        """初始化分词器 / Initialize Tokenizer.

        执行步骤 / Execution Steps:
        1. 逐行读取 vocab.txt 构建 token→id 映射（行号即为 ID）。
        2. 缓存特殊 token ID（[UNK], [CLS], [SEP], [PAD]）。

        vocab.txt 格式：每行一个 token，行号（从0开始）即为该 token 的 ID。
        例如第 0 行是 [PAD]，第 101 行是 [CLS]。

        Args:
            vocab_path: vocab.txt 词表文件路径 / Path to vocab.txt vocabulary file.
        """
        # 初始化词表字典：token字符串 → 整数ID
        self.vocab: dict[str, int] = {}
        # 逐行读取词表文件，行号即为 token ID
        with open(vocab_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()  # 去除行尾换行符
                self.vocab[token] = idx  # 建立 token → ID 映射

        # 缓存特殊 token 的 ID（带默认值防止词表不完整）
        self.unk_id = self.vocab.get("[UNK]", 100)  # 未登录词 ID
        self.cls_id = self.vocab.get("[CLS]", 101)  # 序列起始标记 ID
        self.sep_id = self.vocab.get("[SEP]", 102)  # 序列结束标记 ID
        self.pad_id = self.vocab.get("[PAD]", 0)    # 填充标记 ID

    def tokenize(self, text: str) -> list[str]:
        """对中文进行单字/字符级切分 / Tokenize Chinese Text at Character Level.

        分词逻辑 / Tokenization logic:
        1. 遍历文本中的每个字符 / Iterate over each character in the text
        2. 如果字符在词表中 → 直接使用 / If character is in vocab → use directly
        3. 如果是字母且小写形式在词表中 → 使用小写形式（大小写折叠） / If alphabet and lowercase form is in vocab → use lowercase form (case folding)
        4. 否则 → 替换为 [UNK] / Otherwise → replace with [UNK]

        大小写折叠说明 / Case folding description:
        中文 BERT 词表通常只包含小写英文字母，但医学文本中常出现
        大写缩写（如 HIV、AIDS、BRCA1），折叠为小写可提升识别稳定性。
        Chinese BERT vocab usually only contains lowercase English letters, but medical texts often contain uppercase abbreviations (e.g., HIV, AIDS, BRCA1). Folding them to lowercase improves recognition stability.

        Args:
            text: 待分词的文本 / Text to tokenize.

        Returns:
            token 列表 / List of tokens.
        """
        tokens: list[str] = []  # 存放分词结果
        for char in text:
            # 情况1：字符直接在词表中（中文汉字、小写字母、数字等）
            if char in self.vocab:
                tokens.append(char)
            # 情况2：字母的大写形式不在词表，但小写形式在（大小写折叠）
            elif char.isalpha() and char.lower() in self.vocab:
                tokens.append(char.lower())
            # 情况3：完全未登录的字符（特殊符号等）
            else:
                tokens.append("[UNK]")
        return tokens

    def encode(self, text: str, max_len: int = 128) -> tuple[list[int], list[int], list[int]]:
        """将文本编码为 BERT 输入张量数据结构 / Encode Text to BERT Input Tensors.

        BERT 输入格式：[CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD] ...

        执行步骤 / Execution Steps:
        1. 分词并在首尾添加 [CLS] 和 [SEP] 特殊标记。
        2. 将 token 映射为 vocab ID（input_ids）。
        3. 生成 attention_mask（有效位置为1，填充位置为0）。
        4. 生成 token_type_ids（单句输入全为0）。
        5. 按 max_len 进行右侧 padding 对齐。

        Args:
            text: 待编码文本 / Text to encode.
            max_len: 最大序列长度（默认128） / Maximum sequence length.

        Returns:
            (input_ids, attention_mask, token_type_ids) 三元组，每个都是长度为 max_len 的整数列表。
        """
        # 构造 token 序列：[CLS] + 分词结果（截断到 max_len-2）+ [SEP]
        tokens = ["[CLS]", *self.tokenize(text)[:max_len - 2], "[SEP]"]
        # 将 token 映射为词表 ID，未登录词使用 unk_id
        input_ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        # attention_mask：有效 token 位置为 1（模型应关注这些位置）
        attention_mask = [1] * len(input_ids)
        # token_type_ids：单句输入全为 0（区分句子A/句子B，此处只有句子A）
        token_type_ids = [0] * len(input_ids)

        # 右侧 Padding：将序列补齐到 max_len 长度
        padding_len = max_len - len(input_ids)
        if padding_len > 0:
            input_ids += [self.pad_id] * padding_len      # 填充位使用 pad_id
            attention_mask += [0] * padding_len            # 填充位 mask 为 0（忽略）
            token_type_ids += [0] * padding_len            # 填充位 type 为 0

        return input_ids, attention_mask, token_type_ids


class ONNXSmallNerEngine(SmallNerEngine):
    """基于 ONNX Runtime 的本地医疗 NER 模型推理引擎 / ONNX Runtime Medical NER Engine.

    使用 ONNX Runtime 加载本地 CMeEE（中文医学命名实体识别）模型。
    模型基于 BERT 架构，输出 BIO 序列标注（B-dis/I-dis/B-dru/I-dru 等）。

    特性：
    - 延迟加载：首次调用 extract() 时才加载模型（避免启动阻塞）
    - 自动降级：onnxruntime 未安装或模型文件不存在时抛出异常，
      调用方捕获后回退到 ModelScope 引擎或 NoOp 空实现
    - 纯 Python 分词：不依赖 transformers 的 tokenizer

    支持的实体类型（CMeEE 标准）：
    - dis: 疾病 → 映射为 MEDICAL_DISEASE
    - dru: 药物 → 映射为 MEDICATION
    - pro: 手术/操作 → 映射为 SURGERY
    - sym: 症状 → 映射为 MEDICAL_DISEASE
    - ite: 检查项目
    - bod: 身体部位 → 映射为 BODY_PART

    English Description:
    Loads a local CMeEE medical entity recognition model via ONNX Runtime,
    with lazy-loading and graceful degradation support.
    """

    def __init__(self, model_path: str | None = None, vocab_path: str | None = None, label_mapping: dict[str, str] | None = None, device: str | None = None, max_len: int = 128):
        """初始化 ONNX NER 引擎 / Initialize ONNX NER Engine.

        仅设置路径和状态标志，不实际加载模型（延迟加载策略）。

        Args:
            model_path: ONNX 模型文件路径（默认 .models/raner_cmeee.onnx）。
            vocab_path: vocab.txt 词表文件路径（默认 .models/vocab.txt）。
            label_mapping: 原始标签→标准标签映射（默认使用 DEFAULT_NER_LABEL_MAPPING）。
            device: 目标计算设备（"cuda" / "cpu" / None）。影响 ONNX Runtime
                Execution Provider 选择；None 时自动检测（优先 CUDA）。
        """
        # 计算项目根目录（从当前文件向上两级）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # 设置模型文件路径（使用默认路径或用户指定路径）
        self.model_path = model_path or os.path.join(project_root, ".models", "raner_cmeee.onnx")
        # 设置词表文件路径
        self.vocab_path = vocab_path or os.path.join(project_root, ".models", "vocab.txt")
        # 原始标签→标准标签映射（可配置，默认使用内置医疗映射）
        self.label_mapping = label_mapping or DEFAULT_NER_LABEL_MAPPING
        # 目标计算设备（影响 Execution Provider 选择）
        self.device = device
        # 最大序列长度（默认 128，可配置）
        self.max_len = max_len
        # ONNX 推理会话（延迟初始化）
        self.session: Any | None = None
        # BERT 分词器实例（延迟初始化）
        self.tokenizer: SimpleChineseBertTokenizer | None = None
        # 初始化状态标志
        self._initialized = False
        # 初始化错误缓存（避免重复尝试已知失败的初始化）
        self._init_error: Exception | None = None

    def _metric_device(self) -> str:
        """返回指标 device 标签（对齐 Go 的 cpu/cuda 词表）。"""
        return "cuda" if (self.device or "").lower().startswith("cuda") else "cpu"

    def _lazy_init(self):
        """延迟加载模型 / Lazy-Load ONNX Model.

        首次调用时执行实际的模型加载 / Execute actual model loading on the first call:
        1. 检查 onnxruntime 是否可用 / Check if onnxruntime is available
        2. 验证模型文件和词表文件是否存在 / Verify if model and vocab files exist
        3. 创建 ONNX InferenceSession / Create ONNX InferenceSession
        4. 初始化 BERT 分词器 / Initialize BERT Tokenizer

        如果初始化失败，缓存错误并在后续调用中直接抛出（不重复尝试）。
        If initialization fails, cache the error and throw it directly in subsequent calls (do not retry).

        Raises:
            FileNotFoundError: 模型或词表文件不存在 / Model or vocab file does not exist.
            ImportError: onnxruntime 未安装 / onnxruntime is not installed.
        """
        # 已初始化则直接返回（避免重复加载）
        if self._initialized:
            return

        # 之前初始化失败过，直接抛出缓存的错误（不重复尝试）
        if self._init_error:
            raise self._init_error

        try:
            # 尝试导入 onnxruntime（未安装时抛出 ImportError）
            import onnxruntime as ort

            # 验证 ONNX 模型文件存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"未找到本地 ONNX 模型文件: {self.model_path}")
            # 验证词表文件存在
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(f"未找到本地 vocab 词表文件: {self.vocab_path}")

            # 创建 ONNX 推理会话（加载模型到内存）
            # 根据 device 参数选择 Execution Provider
            providers: list[str] | None = None
            if self.device == "cpu":
                providers = ["CPUExecutionProvider"]
            elif self.device == "cuda":
                available = ort.get_available_providers()
                providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
            self.session = ort.InferenceSession(self.model_path, providers=providers) if providers else ort.InferenceSession(self.model_path)
            # 初始化纯 Python BERT 分词器
            self.tokenizer = SimpleChineseBertTokenizer(self.vocab_path)
            # 标记初始化成功
            self._initialized = True
            # 记录初始化成功日志
            logger.info(
                "onnx_ner_engine_initialized",
                extra={"model_path": self.model_path, "engine": "onnx"},
            )
        except Exception as e:
            # 缓存初始化错误，后续调用直接抛出
            self._init_error = e
            # 记录初始化失败警告日志
            logger.warning(
                "onnx_ner_engine_init_failed",
                extra={"error": str(e), "model_path": self.model_path},
            )
            raise e

    def _parse_bioes_tags(self, tokens: list[str], label_indices: list[int], probs: list[float]) -> list[dict[str, Any]]:
        """解析 BIOES 序列标注 (37 类 RaNER / CMeEE 标准模型) / Parse BIOES Sequence Labels.

        BIOES 标注方案 (37 类):
        - B-XXX: 实体起始 (Begin)
        - I-XXX: 实体内部 (Inside)
        - E-XXX: 实体结束 (End)
        - S-XXX: 单字实体 (Single)
        - O: 非实体 (Outside)
        """
        bioes_label_map: dict[int, str] = {
            0: "O",
            1: "B-bod", 2: "S-bod",
            3: "B-dep", 4: "S-dep",
            5: "B-dis", 6: "S-dis",
            7: "B-dru", 8: "S-dru",
            9: "B-equ", 10: "S-equ",
            11: "B-ite", 12: "S-ite",
            13: "B-mic", 14: "S-mic",
            15: "B-pro", 16: "S-pro",
            17: "B-sym", 18: "S-sym",
            19: "I-bod", 20: "E-bod",
            21: "I-dep", 22: "E-dep",
            23: "I-dis", 24: "E-dis",
            25: "I-dru", 26: "E-dru",
            27: "I-equ", 28: "E-equ",
            29: "I-ite", 30: "E-ite",
            31: "I-mic", 32: "E-mic",
            33: "I-pro", 34: "E-pro",
            35: "I-sym", 36: "E-sym",
        }

        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for idx in range(1, len(tokens) - 1):
            token = tokens[idx]
            if token == "[SEP]" or token == "[PAD]":  # noqa: S105 —— 分词器特殊标记字面量，非口令
                break

            label_idx = label_indices[idx]
            prob = probs[idx]
            tag = bioes_label_map.get(label_idx, "O")

            if tag == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif tag.startswith("S-"):
                if current_entity:
                    entities.append(current_entity)
                ent_type = tag.split("-")[1]
                entities.append({
                    "text": token,
                    "label": ent_type,
                    "confidence": prob,
                })
                current_entity = None
            elif tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                ent_type = tag.split("-")[1]
                current_entity = {
                    "text": token,
                    "label": ent_type,
                    "confidence": prob,
                }
            elif tag.startswith("I-") and current_entity:
                ent_type = tag.split("-")[1]
                if ent_type == current_entity["label"]:
                    current_entity["text"] += token
                    current_entity["confidence"] = min(current_entity["confidence"], prob)
                else:
                    entities.append(current_entity)
                    current_entity = None
            elif tag.startswith("E-") and current_entity:
                ent_type = tag.split("-")[1]
                if ent_type == current_entity["label"]:
                    current_entity["text"] += token
                    current_entity["confidence"] = min(current_entity["confidence"], prob)
                    entities.append(current_entity)
                    current_entity = None
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def _parse_bio_tags(self, tokens: list[str], label_indices: list[int], probs: list[float]) -> list[dict[str, Any]]:
        """解析 BIO 序列标注 (13 类旧版 BIO) / Parse BIO Sequence Labels.

        BIO 标注方案 / BIO Labeling Scheme:
        - B-XXX: 实体起始 (Begin)
        - I-XXX: 实体内部 (Inside)
        - O: 非实体 (Outside)
        """
        # CMeEE 标签索引映射表（索引 0 为 O，1-12 为 B/I 标签对）
        label_map = {
            1: "B-dis", 2: "I-dis",    # 疾病（disease）
            3: "B-dru", 4: "I-dru",    # 药物（drug）
            5: "B-pro", 6: "I-pro",    # 手术/操作（procedure）
            7: "B-sym", 8: "I-sym",    # 症状（symptom）
            9: "B-ite", 10: "I-ite",   # 检查项目（item）
            11: "B-bod", 12: "I-bod",  # 身体部位（body）
            13: "B-dep", 14: "I-dep",  # 科室（department）
            15: "B-equ", 16: "I-equ",  # 医疗设备（equipment）
            17: "B-mic", 18: "I-mic",  # 微生物（microbe）
        }

        entities: list[dict[str, Any]] = []  # 已完成的实体列表
        current_entity: dict[str, Any] | None = None  # 当前正在构建的实体

        # 遍历 token 序列（跳过 index 0 的 [CLS] 和末尾的 [SEP]/[PAD]）
        for idx in range(1, len(tokens) - 1):
            token = tokens[idx]
            # 遇到 [SEP] 或 [PAD] 表示有效序列结束
            if token == "[SEP]" or token == "[PAD]":  # noqa: S105 —— 分词器特殊标记字面量，非口令
                break

            # 获取当前 token 的预测标签和概率
            label_idx = label_indices[idx]
            prob = probs[idx]
            # 查表获取 BIO 标签字符串，未命中则为 "O"
            tag = label_map.get(label_idx, "O")

            if tag.startswith("B-"):
                # B- 标签：新实体开始
                # 如果前一个实体尚未保存，先保存
                if current_entity:
                    entities.append(current_entity)
                # 提取实体类型（如 "B-dis" → "dis"）
                ent_type = tag.split("-")[1]
                # 开始构建新实体
                current_entity = {
                    "text": token,        # 实体文本（逐字累积）
                    "label": ent_type,    # 实体类型
                    "confidence": prob,   # 置信度（取所有字的最小值）
                }
            elif tag.startswith("I-") and current_entity:
                # I- 标签：实体内部（需要有正在构建的实体）
                ent_type = tag.split("-")[1]
                if ent_type == current_entity["label"]:
                    # 类型匹配：合并当前字符到实体文本
                    current_entity["text"] += token
                    # 置信度取最小值（木桶原则：整体置信度取决于最不确定的字）
                    current_entity["confidence"] = min(current_entity["confidence"], prob)
                else:
                    # 类型不匹配：结束当前实体，丢弃不匹配的 I- 标签
                    entities.append(current_entity)
                    current_entity = None
            else:
                # O 标签或无当前实体时的 I- 标签：结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # 序列结束后，如果还有未保存的实体则保存
        if current_entity:
            entities.append(current_entity)

        return entities

    @staticmethod
    def _chunk_text(text: str, max_chunk_len: int = 120) -> list[str]:
        """将长文本按句号/分号/换行符切分为不大于 max_chunk_len 的子句，超长单句走滑动窗口。"""
        import re
        if not text or len(text) <= max_chunk_len:
            return [text]

        chunks: list[str] = []
        raw_sentences = [s for s in re.split(r"(?<=[。；;\n\r])", text) if s]
        curr_chunk = ""

        for s in raw_sentences:
            if len(s) > max_chunk_len:
                if curr_chunk:
                    chunks.append(curr_chunk)
                    curr_chunk = ""
                step = max(max_chunk_len - 20, 10)
                for i in range(0, len(s), step):
                    chunks.append(s[i:i + max_chunk_len])
            elif len(curr_chunk) + len(s) <= max_chunk_len:
                curr_chunk += s
            else:
                chunks.append(curr_chunk)
                curr_chunk = s

        if curr_chunk:
            chunks.append(curr_chunk)

        return chunks if chunks else [text]

    def _extract_single_chunk(self, text: str, eff_max_len: int = 128) -> list[dict[str, Any]]:
        """单块文本的 ONNX NER 推理。"""
        assert self.tokenizer is not None and self.session is not None
        input_ids, attention_mask, token_type_ids = self.tokenizer.encode(text, max_len=eff_max_len)
        inputs = {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "token_type_ids": [token_type_ids],
        }
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]
        num_labels = logits.shape[-1] if len(logits.shape) > 1 else 13

        import numpy as np
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        label_indices = np.argmax(probs, axis=-1).tolist()
        token_probs = [probs[i, label_indices[i]] for i in range(len(label_indices))]
        tokens = ["[CLS]", *self.tokenizer.tokenize(text)[:eff_max_len - 2], "[SEP]"]

        if num_labels >= 37:
            entities = self._parse_bioes_tags(tokens, label_indices, token_probs)
        else:
            entities = self._parse_bio_tags(tokens, label_indices, token_probs)

        for ent in entities:
            raw_label = ent["label"]
            if raw_label in self.label_mapping:
                ent["label"] = self.label_mapping[raw_label]

        return entities

    def extract(self, text: str, max_len: int | None = None) -> list[dict[str, Any]]:
        """提取输入文本中的医疗实体 / Extract Medical Entities from Text.

        支持长文本 (>128 Token) 自动智能分句与滑动窗口切片推理，
        防止 128 Token 截断导致的末尾实体漏检。

        Args:
            text: 目标文本片段 / Target text segment.
            max_len: 可选的最大序列长度，默认使用实例配置 max_len。

        Returns:
            命名实体字典列表，每个字典含 text/label/confidence。
            初始化失败或推理异常时返回空列表（优雅降级）。
        """
        try:
            # 延迟初始化（首次调用时加载模型）
            self._lazy_init()
        except Exception:
            # 初始化失败：递增失败计数指标，返回空列表
            CLASSIFICATION_NER_TOTAL.labels(status="init_failed").inc()
            return []

        # 断言初始化成功（类型检查用）
        assert self.tokenizer is not None and self.session is not None
        start_time = time.monotonic()
        try:
            eff_max_len = max_len or getattr(self, "max_len", 128)
            max_chunk_len = max(eff_max_len - 2, 10)

            if len(text) > max_chunk_len:
                chunks = self._chunk_text(text, max_chunk_len=max_chunk_len)
                merged_entities: list[dict[str, Any]] = []
                seen_keys: set[tuple[str, str]] = set()

                for chunk in chunks:
                    chunk_ents = self._extract_single_chunk(chunk, eff_max_len)
                    for ent in chunk_ents:
                        key = (ent["text"], ent["label"])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            merged_entities.append(ent)
                        else:
                            for existing in merged_entities:
                                if (existing["text"], existing["label"]) == key:
                                    existing["confidence"] = max(existing["confidence"], ent["confidence"])
                duration = time.monotonic() - start_time
                CLASSIFICATION_NER_TOTAL.labels(status="success").inc()
                CLASSIFICATION_NER_DURATION.labels(engine="onnx").observe(duration)
                observe_ner_inference(duration, self._metric_device())
                logger.debug(
                    "onnx_ner_extract_completed",
                    extra={"entity_count": len(merged_entities), "duration_s": round(duration, 4)},
                )
                return merged_entities

            entities = self._extract_single_chunk(text, eff_max_len)
            duration = time.monotonic() - start_time
            CLASSIFICATION_NER_TOTAL.labels(status="success").inc()
            CLASSIFICATION_NER_DURATION.labels(engine="onnx").observe(duration)
            observe_ner_inference(duration, self._metric_device())
            logger.debug(
                "onnx_ner_extract_completed",
                extra={"entity_count": len(entities), "duration_s": round(duration, 4)},
            )
            return entities

        except Exception as e:
            # 推理异常：记录错误指标和日志，返回空列表（优雅降级）
            duration = time.monotonic() - start_time
            CLASSIFICATION_NER_TOTAL.labels(status="error").inc()
            CLASSIFICATION_NER_DURATION.labels(engine="onnx").observe(duration)
            observe_ner_inference(duration, self._metric_device())
            logger.warning(
                "onnx_ner_extract_error",
                extra={"error": str(e), "duration_s": round(duration, 4)},
            )
            return []


class TensorRTSmallNerEngine(ONNXSmallNerEngine):
    """基于 NVIDIA TensorRT 极致加速的本地医疗 NER 引擎 / TensorRT Medical NER Engine.

    纯 C++ 硬件加速引擎（零 PyTorch 依赖），通过 ONNX Runtime TensorRT Execution Provider
    或 TensorRT C++ 原生引擎构建并加载编译后的优化图（FP16 模式 + 引擎缓存），
    实现高性能、极低延迟的医疗命名实体识别。

    核心优势：
    - 零 PyTorch 依赖：完全脱离 PyTorch 及其算力卡顿/版本冲突限制
    - 算子融合与 FP16 模式：充分发挥 NVIDIA GPU Tensor Cores 硬件性能
    - 引擎持久化缓存：自动生成并加载 .engine 缓存文件，二次启动毫秒级响应
    """

    def _metric_device(self) -> str:
        return "cuda"

    def _lazy_init(self):
        """延迟加载并编译 TensorRT 引擎 / Lazy-Load & Compile TensorRT Engine."""
        if self._initialized:
            return
        if self._init_error:
            raise self._init_error

        try:
            self._preload_nvidia_libs()
            import onnxruntime as ort

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"未找到本地 ONNX 模型文件: {self.model_path}")
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(f"未找到本地 vocab 词表文件: {self.vocab_path}")

            available_providers = ort.get_available_providers()
            trt_cache_dir = os.path.dirname(self.model_path)

            # 配置 TensorRT 专属属性（FP16 精度、自动引擎缓存）
            trt_options = {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
            }

            if "TensorrtExecutionProvider" in available_providers:
                providers = [
                    ("TensorrtExecutionProvider", trt_options),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            elif "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                raise RuntimeError("TensorRT 和 CUDA ExecutionProvider 在当前环境均不可用")

            # 创建基于 TensorRT 硬件加速的 InferenceSession
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.tokenizer = SimpleChineseBertTokenizer(self.vocab_path)
            self._initialized = True
            active_provider = self.session.get_providers()[0]
            logger.info(
                "tensorrt_ner_engine_initialized",
                extra={
                    "model_path": self.model_path,
                    "engine": "tensorrt",
                    "provider": active_provider,
                },
            )
        except Exception as e:
            self._init_error = e
            logger.warning(
                "tensorrt_ner_engine_init_failed",
                extra={"error": str(e), "model_path": self.model_path},
            )
            raise e


class ModelScopeSmallNerEngine(SmallNerEngine):
    """基于 ModelScope 官方推理管道的本地医疗 NER 引擎 / ModelScope Medical NER Engine.

    使用达摩院 RaNER 医疗实体识别微调模型（CMeEE 数据集），
    通过 ModelScope pipeline 接口进行推理。

    与 ONNXSmallNerEngine 的区别：
    - 需要 PyTorch + transformers + modelscope 完整依赖
    - 推理速度稍慢（PyTorch 动态图 vs ONNX 静态图）
    - 兼容性更好（不需要手动转换 ONNX 格式）
    - 包含多项兼容性 Patch（适配不同版本的 transformers/modelscope）

    降级策略：
    - modelscope 未安装 → 抛出 ImportError → 回退到 NoOpSmallNerEngine
    - PyTorch 不可用 → 抛出异常 → 回退到 NoOpSmallNerEngine

    English Description:
    Uses DAMO Academy RaNER medical entity recognition fine-tuned model via ModelScope
    pipeline, with lazy-loading and graceful degradation support.
    """

    def __init__(
        self,
        model_id: str = "damo/nlp_raner_named-entity-recognition_chinese-base-cmeee",
        label_mapping: dict[str, str] | None = None,
        device: str | None = None,
    ):
        """初始化 ModelScope NER 引擎 / Initialize ModelScope NER Engine.

        仅设置模型引用和状态标志，不实际加载模型（延迟加载策略）。

        Args:
            model_id: ModelScope 上的模型 ID，默认使用达摩院 RaNER CMeEE 微调模型。
            label_mapping: 原始标签→标准标签映射（默认使用 DEFAULT_NER_LABEL_MAPPING）。
            device: 目标设备 ("cuda", "cpu", "mps" 等)。
        """
        # 保存模型 ID（用于从 Hub 下载或标识本地模型）
        self.model_id = model_id
        # 原始标签→标准标签映射（可配置，默认使用内置医疗映射）
        self.label_mapping = label_mapping or DEFAULT_NER_LABEL_MAPPING
        self.device = device
        # 实际解析后的推理设备（初始化时按 CUDA 可用性确定）
        self._resolved_device = "cpu"
        # 计算本地模型目录路径（download_ner_model.py 下载的位置）
        # 优先使用本地已下载的模型，避免推理时再次从 Hub 拉取（离线友好）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        local_model_dir = os.path.join(project_root, ".models", "raner_cmeee")
        self.local_model_dir = local_model_dir
        # ModelScope pipeline 实例（延迟初始化）
        self.pipeline: Any | None = None
        # 初始化状态标志
        self._initialized = False
        # 初始化错误缓存
        self._init_error: Exception | None = None

    # 注：CUDA/Triton 共享库预加载沿用基类 SmallNerEngine._preload_nvidia_libs，
    # 此处不再重复实现（旧副本每次调用都会向 LD_LIBRARY_PATH 追加整批目录，
    # 增长超过 ARG_MAX 后会让子进程 exec 报 Errno 7）。

    def _metric_device(self) -> str:
        """返回指标 device 标签（对齐 Go 的 cpu/cuda 词表）。"""
        return "cuda" if self._resolved_device == "cuda" else "cpu"

    @classmethod
    def _is_cuda_compatible(cls, torch: Any) -> bool:
        """验证当前 PyTorch 是否真能在检测到的 CUDA 设备上执行 kernel。"""
        cls._preload_nvidia_libs()
        if not torch.cuda.is_available():
            return False
        try:
            a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            b = torch.tensor([1.0, 1.0, 1.0], device="cuda")
            _ = (a + b).sum().item()
            return True
        except RuntimeError:
            return False

    def _lazy_init(self):
        """延迟加载 ModelScope 管道 / Lazy-Load ModelScope Pipeline.

        首次调用时执行实际的管道初始化，包含多项兼容性 Patch：

        Patch 1: transformers.onnx Dummy 模块注入
        - 问题：新版 transformers 移除了 transformers.onnx 模块，
          但 ModelScope 的推理脚本仍有该 legacy 导入
        - 方案：在 sys.modules 中注入 Dummy 模块

        Patch 2: PretrainedConfig 属性注入
        - 问题：ModelScope 模型的 Config 未正确初始化某些类属性
        - 方案：动态添加默认属性值

        Patch 3: get_extended_attention_mask 切面拦截
        - 问题：ModelScope 传入 torch.device 作为第三参数，
          但新版 transformers 已将该参数改为 dtype
        - 方案：拦截方法调用，自动丢弃 device 参数

        Patch 4: get_head_mask 方法绑定
        - 问题：ModelScope 的 BertModel 未继承 PreTrainedModel，缺失该方法
        - 方案：动态绑定 PreTrainedModel.get_head_mask

        Raises:
            Exception: 初始化失败时抛出。
        """
        # 已初始化则直接返回
        if self._initialized:
            return

        # 之前初始化失败过，直接抛出缓存的错误
        if self._init_error:
            raise self._init_error

        try:
            # === Patch 1: transformers.onnx Dummy 模块注入 ===
            import sys
            import types

            # 如果 transformers.onnx 不在已加载模块中，注入 Dummy
            if "transformers.onnx" not in sys.modules:
                dummy_onnx = types.ModuleType("transformers.onnx")

                # 创建占位类（ModelScope 脚本只需要这些名字存在）
                class DummyOnnxConfig:
                    pass

                # 注入 OnnxConfig 和 OnnxConfigWithPast 占位
                cast("Any", dummy_onnx).OnnxConfig = DummyOnnxConfig
                cast("Any", dummy_onnx).OnnxConfigWithPast = DummyOnnxConfig
                # 注册到 sys.modules，使 import transformers.onnx 不报错
                sys.modules["transformers.onnx"] = dummy_onnx

            # === Patch 2: PretrainedConfig 属性注入 ===
            from transformers import PretrainedConfig, PreTrainedModel
            # 动态添加 ModelScope 模型可能缺失的类属性默认值
            PretrainedConfig.is_decoder = False                # 是否为解码器
            PretrainedConfig.add_cross_attention = False       # 是否添加交叉注意力
            cast("Any", PretrainedConfig).bad_words_ids = None # 禁止词 ID 列表
            PretrainedConfig.chunk_size_feed_forward = 0       # 前馈分块大小
            PretrainedConfig.pruned_heads = {}                 # 已剪枝的注意力头
            PretrainedConfig.tie_word_embeddings = True        # 是否共享嵌入权重

            # === Patch 3: get_extended_attention_mask 切面拦截 ===
            # 保存原始方法引用
            orig_get_extended_attention = PreTrainedModel.get_extended_attention_mask

            def patched_get_extended_attention_mask(self, attention_mask, input_shape, *args, **kwargs):
                """修补版：自动丢弃 ModelScope 传入的 torch.device 参数。"""
                import torch
                new_args = list(args)
                # 如果第一个位置参数是 torch.device，则丢弃它
                if len(new_args) > 0 and isinstance(new_args[0], torch.device):
                    new_args = new_args[1:]
                # 同时丢弃关键字参数中的 device
                kwargs.pop("device", None)
                # 调用原始方法
                return orig_get_extended_attention(self, attention_mask, input_shape, *new_args, **kwargs)

            # 用修补版替换原始方法
            cast("Any", PreTrainedModel).get_extended_attention_mask = patched_get_extended_attention_mask

            # === Patch 4: get_head_mask 方法绑定 ===
            try:
                from modelscope.models.nlp.bert.backbone import BertModel
                # 如果 ModelScope 的 BertModel 缺少 get_head_mask 方法，则绑定
                if not hasattr(BertModel, "get_head_mask"):
                    BertModel.get_head_mask = PreTrainedModel.get_head_mask
            except ImportError:
                pass  # 如果无法导入 ModelScope 的 BertModel，跳过此 Patch

            # === 加载 ModelScope Pipeline ===
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            # 优先使用本地已下载的模型目录，否则回退到 Hub 模型 ID
            model_ref = self.model_id
            if os.path.isdir(self.local_model_dir):
                model_ref = self.local_model_dir  # 使用本地目录（离线友好）

            # 记录加载日志
            logger.info(
                "modelscope_ner_pipeline_loading",
                extra={"model_id": self.model_id, "model_ref": model_ref},
            )
            # 检测 CUDA 是否真正可用；算力不兼容时强制 CPU，避免初始化崩溃
            import torch

            device = "cuda" if self._is_cuda_compatible(torch) else "cpu"
            self._resolved_device = device
            if device == "cpu":
                logger.info(
                    "modelscope_ner_select_cpu",
                    extra={"reason": "cuda_not_compatible_or_unavailable"},
                )
            # 创建命名实体识别管道
            self.pipeline = pipeline(Tasks.named_entity_recognition, model=model_ref, device=device)
            # ModelScope 某些版本 pipeline 参数不会自动将权重搬移到 GPU，必须显式调用 model.to(device)
            if device and device != "cpu" and hasattr(self.pipeline, "model") and hasattr(self.pipeline.model, "to"):
                try:
                    self.pipeline.model.to(device)
                except Exception as e:
                    logger.warning("modelscope_ner_model_to_device_failed", extra={"device": device, "error": str(e)})
            # 标记初始化成功
            self._initialized = True
            logger.info(
                "modelscope_ner_engine_initialized",
                extra={"model_id": self.model_id, "engine": "modelscope", "device": device},
            )
        except Exception as e:
            # 缓存初始化错误
            self._init_error = e
            logger.warning(
                "modelscope_ner_engine_init_failed",
                extra={"error": str(e), "model_id": self.model_id},
            )
            raise e

    def _extract_single_chunk(self, text: str) -> list[dict[str, Any]]:
        """调用 ModelScope pipeline 提取单块文本命名实体。"""
        assert self.pipeline is not None
        res = self.pipeline(text)
        output = res.get("output", [])
        entities: list[dict[str, Any]] = []
        for item in output:
            raw_label = item.get("type", "")
            span = item.get("span", "")
            label = self.label_mapping.get(raw_label, raw_label)
            entities.append(
                {
                    "text": span,
                    "label": label,
                    "confidence": 1.0,
                }
            )
        return entities

    def extract(self, text: str) -> list[dict[str, Any]]:
        """调用 ModelScope pipeline 提取命名实体 / Extract Entities via ModelScope Pipeline.

        支持长文本 (>120 Token) 智能分句切片推理，防止长病文书截断漏检。

        Args:
            text: 目标文本 / Target text.

        Returns:
            命名实体字典列表，每个字典含 text/label/confidence。
            初始化失败或推理异常时返回空列表。
        """
        try:
            # 延迟初始化
            self._lazy_init()
        except Exception:
            # 初始化失败：递增失败计数，返回空列表
            CLASSIFICATION_NER_TOTAL.labels(status="init_failed").inc()
            return []

        # 断言管道已初始化
        assert self.pipeline is not None
        # 记录推理开始时间
        start_time = time.monotonic()
        try:
            max_chunk_len = 120
            if len(text) > max_chunk_len:
                chunks = ONNXSmallNerEngine._chunk_text(text, max_chunk_len=max_chunk_len)
                merged_entities: list[dict[str, Any]] = []
                seen_keys: set[tuple[str, str]] = set()

                for chunk in chunks:
                    chunk_ents = self._extract_single_chunk(chunk)
                    for ent in chunk_ents:
                        key = (ent["text"], ent["label"])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            merged_entities.append(ent)
                duration = time.monotonic() - start_time
                CLASSIFICATION_NER_TOTAL.labels(status="success").inc()
                CLASSIFICATION_NER_DURATION.labels(engine="modelscope").observe(duration)
                observe_ner_inference(duration, self._metric_device())
                logger.debug(
                    "modelscope_ner_extract_completed",
                    extra={"entity_count": len(merged_entities), "duration_s": round(duration, 4)},
                )
                return merged_entities

            entities = self._extract_single_chunk(text)
            duration = time.monotonic() - start_time
            CLASSIFICATION_NER_TOTAL.labels(status="success").inc()
            CLASSIFICATION_NER_DURATION.labels(engine="modelscope").observe(duration)
            observe_ner_inference(duration, self._metric_device())
            logger.debug(
                "modelscope_ner_extract_completed",
                extra={"entity_count": len(entities), "duration_s": round(duration, 4)},
            )
            return entities
        except Exception as e:
            # 推理异常：记录错误指标和日志，返回空列表
            duration = time.monotonic() - start_time
            CLASSIFICATION_NER_TOTAL.labels(status="error").inc()
            CLASSIFICATION_NER_DURATION.labels(engine="modelscope").observe(duration)
            observe_ner_inference(duration, self._metric_device())
            logger.warning(
                "modelscope_ner_extract_error",
                extra={"error": str(e), "duration_s": round(duration, 4)},
            )
            return []
