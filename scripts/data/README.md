# 数据生成与规则扩充脚本 (scripts/data)

本目录包含 **数联天下 · 数盾 (`PrivShield`)** 用于生成高仿真业务测试数据集、多模态图文数据以及基于大模型/规则自动化扩充敏感特征词表的工具脚本。

每个脚本均支持独立运行，以下为各脚本的详细说明与独立启动代码。

---

## 目录索引

- [`expand_keywords_with_llm.py` (LLM/同义词规则关键词扩充)](#expand_keywords_with_llmpy)
- [`gen_medical_images.py` (多模态医疗病历与报告图像生成)](#gen_medical_imagespy)
- [`gen_yaml_from_doc.py` (行业标准文档自动转 YAML 规则)](#gen_yaml_from_docpy)
- [`generate_kangyang_data.py` (康养综合测试数据集生成)](#generate_kangyang_datapy)
- [`generate_medical_data.py` (临床医疗测试数据集生成)](#generate_medical_datapy)
- [`generate_yibao_data.py` (国家医保结算测试数据集生成)](#generate_yibao_datapy)

---

## 详细功能与启动命令

### `expand_keywords_with_llm.py`
- **作用说明**: 针对 `rules/domains/*.yaml` 中的规则定义，调用大语言模型（如 Qwen/DeepSeek/OpenAI）或离线内置近义词库，自动扩充同义词、缩写、拼音首字母及行业别名，增强 Layer-1 规则引擎的召回率。
- **参数选项**:
  - `-i, --in-place`: 直接就地修改覆盖原 YAML 文件。
  - `-o, --output <PATH>`: 输出保存到指定的新 YAML 文件。
  - `--api-key <KEY>`: 大模型 API Key（不指定则使用离线同义词库）。
  - `--api-base <URL>`: 大模型兼容 API Base URL。
  - `--model <NAME>`: 模型名称（如 `qwen-plus`, `deepseek-chat`）。
- **执行命令**:
  ```bash
  # 1. 终端预览扩充后的 YAML 内容
  python scripts/data/expand_keywords_with_llm.py rules/domains/general-pii.yaml
  ```
  ```bash
  # 2. 直接覆盖就地更新原 YAML 规则文件
  python scripts/data/expand_keywords_with_llm.py rules/domains/general-pii.yaml -i
  ```
  ```bash
  # 3. 指定输出到新文件
  python scripts/data/expand_keywords_with_llm.py rules/domains/finance.yaml -o rules/domains/finance_expanded.yaml
  ```

---

### `gen_medical_images.py`
- **作用说明**: 使用 Pillow 本地绘制高保真中文医疗影像与报告单据（如门诊病历、放射检查、处方底方），用于验证多模态 VLM / OCR 图像敏感数据打码与分类分级能力。
- **参数选项**:
  - `--list`: 列出所有可用的单据与病历渲染模板。
  - 环境变量 `PLA_CJK_FONT`: 指定中文字体文件路径。
- **执行命令**:
  ```bash
  # 批量生成全套病历图片至 console/web/src/assets/medical/ 目录
  python scripts/data/gen_medical_images.py
  ```
  ```bash
  # 查看支持的病历与报告模板列表
  python scripts/data/gen_medical_images.py --list
  ```

---

### `gen_yaml_from_doc.py`
- **作用说明**: 自动化将政策法规、地方标准或行业规范 Markdown 文档（如《四川省健康医疗大数据应用指南.md》）转化为 PrivShield 结构化规则 YAML。
- **参数选项**:
  - `--domain <NAME>`: 领域标识符（如 `sc_health`）。
  - `-o, --output <PATH>`: 指定生成的规则 YAML 文件路径。
  - `--api-key <KEY>`: 联动大模型进行语义解析与同义词扩展。
- **执行命令**:
  ```bash
  # 提取 Markdown 标准文档并直接生成领域规则 YAML
  python scripts/data/gen_yaml_from_doc.py docs/standard/四川省健康医疗大数据应用指南.md \
    --domain sc_health \
    -o rules/domains/sc_health_auto.yaml
  ```

---

### `generate_kangyang_data.py`
- **作用说明**: 生成包含 27 个维度的康养（老年健康、残疾人护理、长期照护）高仿真综合测试数据集 CSV，涵盖评估量表、病程记录、残疾人证号与身份证号合法校验和。
- **执行命令**:
  ```bash
  # 生成默认 100 条康养仿真测试数据集 (输出至 data/kangyang.csv)
  python scripts/data/generate_kangyang_data.py
  ```

---

### `generate_medical_data.py`
- **作用说明**: 生成包含多维度临床诊疗病历、身份信息、诊断名称、用药清单、主诉及既往病史的高仿真医疗测试数据集 CSV。
- **参数选项**:
  - `-o, --output <PATH>`: 指定输出文件路径（默认 `data/kangyang.csv`）。
  - `-n, --count <INT>`: 生成记录条数（默认 50 条）。
- **执行命令**:
  ```bash
  # 默认生成医疗数据集
  python scripts/data/generate_medical_data.py
  ```
  ```bash
  # 自定义生成 100 条医疗数据集并保存至指定路径
  python scripts/data/generate_medical_data.py --output data/medical_samples_100.csv --count 100
  ```

---

### `generate_yibao_data.py`
- **作用说明**: 依据国家医疗保障局结算标准，生成 18 个核心维度的医保结算测试数据集（含 ICD-10 编码、结算流水号、就诊科室、住院天数等）。
- **参数选项**:
  - `-o, --output <PATH>`: 指定输出路径（默认保存至 `engine/medical_pipeline/samples/yibao.csv`）。
  - `-n, --count <INT>`: 生成记录条目数。
- **执行命令**:
  ```bash
  # 默认生成医保数据集
  python scripts/data/generate_yibao_data.py
  ```
  ```bash
  # 生成 50 条医保记录并输出至指定路径
  python scripts/data/generate_yibao_data.py --output data/yibao_test.csv --count 50
  ```
