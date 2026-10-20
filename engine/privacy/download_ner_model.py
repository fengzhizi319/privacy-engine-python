"""下载 Small-NER ONNX 模型与词表文件 / Download Small-NER ONNX Model and Vocabulary.

中文说明：
支持通过 ModelScope 社区（首选）或 Hugging Face 镜像站下载 CMeEE 命名实体识别模型。
下载策略采用优先级回退机制：
1. 优先使用 ModelScope SDK 下载完整模型仓库（含 PyTorch 权重、vocab.txt 等）。
2. 若 ModelScope 不可用，回退至 Hugging Face 镜像站下载轻量 ONNX 模型与词表。

下载产物存储在项目根目录下的 .models/ 目录中：
- .models/raner_cmeee/       — ModelScope 完整模型仓库
- .models/raner_cmeee.onnx   — ONNX 轻量推理模型
- .models/vocab.txt          — BERT 词表文件

供 classification_ner.py 中的 ONNXSmallNerEngine 延迟加载使用。

English Description:
Download utility for the Small-NER (CMeEE Named Entity Recognition) ONNX model and vocabulary.
Supports ModelScope (preferred) and Hugging Face mirror as fallback.
Artifacts are stored under .models/ in the project root for lazy-loading
by ONNXSmallNerEngine in classification_ner.py.

Usage:
    python -m engine.privacy.download_ner_model
"""

import os
import shutil
import sys
import urllib.request


def download_via_modelscope(model_id: str, local_dir: str) -> bool:
    """使用 ModelScope SDK 下载 CMeEE 命名实体识别模型 / Download CMeEE NER Model via ModelScope.

    中文说明：
    下载整个模型仓库快照（含 PyTorch 权重、vocab.txt、config 等）到本地目录。
    ModelScope 为国内首选下载通道，速度最快。

    English Description:
    Downloads the full model repository snapshot (PyTorch weights, vocab.txt, config, etc.)
    to the specified local directory. ModelScope is the fastest channel in China.

    Args:
        model_id: ModelScope 模型标识符 / ModelScope model identifier
            (e.g. "iic/nlp_raner_named-entity-recognition_chinese-base-cmeee").
        local_dir: 本地保存目录 / Local directory to save the model.

    Returns:
        True 表示下载成功 / True if download succeeded.
    """
    try:
        from modelscope import snapshot_download

        print(f"[*] 正在通过 ModelScope 社区高速下载模型 {model_id} ...")
        # 下载整个仓库文件到本地指定文件夹中（含 PyTorch 权重、vocab.txt 等）
        snapshot_download(model_id, local_dir=local_dir)
        print("[+] ModelScope 下载完成！")
        return True
    except ImportError:
        print("[-] 未检测到 modelscope 库，跳过 ModelScope 本地下载通道。")
        return False
    except Exception as e:
        print(f"[-] ModelScope 下载遇到异常: {e}")
        return False


def download_file(url: str, target_path: str) -> bool:
    """使用 urllib 下载指定 URL 文件并保存至本地 / Download File from URL via urllib.

    中文说明：
    采用 1MB 分块读取，防止大文件下载时内存溢出。
    支持通过环境变量 HF_TOKEN 携带 Hugging Face 授权令牌。

    English Description:
    Downloads a file from the given URL using chunked reading (1MB chunks)
    to prevent memory overflow for large files.
    Supports Hugging Face auth token via HF_TOKEN environment variable.

    Args:
        url: 文件下载 URL / File download URL.
        target_path: 本地保存路径 / Local file path to save.

    Returns:
        True 表示下载成功 / True if download succeeded.
    """
    print(f"[*] 正在从镜像站下载 {url} -> {target_path} ...")

    # 构造 HTTP 请求头，模拟浏览器访问以避免被拒绝
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    # 若配置了 HF_TOKEN，携带 Bearer 授权头访问私有/受限资源
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        print("[*] 已携带本地 HF_TOKEN 进行授权访问。")

    req = urllib.request.Request(url, headers=headers)  # noqa: S310 trusted model-download endpoint

    try:
        # timeout=60：限制连接建立与单次 socket 读阻塞的最长秒数，
        # 防止镜像站无响应时下载进程无限挂起（分块下载期间每次 read 均受约束）
        with urllib.request.urlopen(req, timeout=60) as response, open(target_path, "wb") as out_file:  # noqa: S310
            # 采用 1MB 分块读取，防止大文件下载时内存溢出
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"[+] 下载完成并保存: {target_path}")
        return True
    except Exception as e:
        print(f"[-] 下载失败: {e}")
        return False


def main():
    """模型下载主入口 / Main Entry Point for NER Model Download.

    执行步骤 / Execution Steps:
    1. 确定模型保存路径（项目根目录/.models/）。
       (Determine model save path: project_root/.models/)
    2. 优先尝试 ModelScope 下载完整模型仓库，并同步 vocab.txt 到 .models/ 根目录。
       (Try ModelScope download first, sync vocab.txt to .models/ root)
    3. 若 ModelScope 失败，回退至 Hugging Face 镜像下载 ONNX 模型与词表。
       (Fall back to Hugging Face mirror for ONNX model and vocab)
    4. 根据结果设置退出码。
       (Set exit code based on result)
    """
    from engine.env_loader import load_env_file
    load_env_file()

    # CMeEE 命名实体识别模型（中文医学命名实体识别）
    model_id = "iic/nlp_raner_named-entity-recognition_chinese-base-cmeee"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.environ.get(
        "PRIVACY_MODELS_DIR",
        os.path.join(project_root, ".models")
    )
    os.makedirs(models_dir, exist_ok=True)

    # 优先尝试从 ModelScope 进行下载（国内速度最快）
    raner_cmeee_dir = os.path.join(models_dir, "raner_cmeee")
    if download_via_modelscope(model_id, raner_cmeee_dir):
        # 复制其中的 vocab.txt 供本地通用 Tokenizer 使用
        vocab_src = os.path.join(raner_cmeee_dir, "vocab.txt")
        vocab_dst = os.path.join(models_dir, "vocab.txt")
        if os.path.exists(vocab_src):
            shutil.copy(vocab_src, vocab_dst)
            print(f"[+] 已同步 ModelScope 词表文件到: {vocab_dst}")
        print("[+] ModelScope 模型及词表配置成功！")
        sys.exit(0)

    # 如果 ModelScope 下载失败或未安装，回退至从 Hugging Face 镜像站下载 ONNX 轻量模型
    print("[*] 正在切换至 Hugging Face 镜像源下载 ONNX 轻量模型...")
    vocab_url = "https://hf-mirror.com/datasets/ZTYNKFX/cmeee-bucket/resolve/main/vocab.txt"
    model_url = "https://hf-mirror.com/datasets/ZTYNKFX/cmeee-bucket/resolve/main/raner_cmeee.onnx"

    vocab_target = os.path.join(models_dir, "vocab.txt")
    model_target = os.path.join(models_dir, "raner_cmeee.onnx")

    # 1. 下载词表文件（BERT 分词器所需）
    if not download_file(vocab_url, vocab_target):
        print("[-] 词表下载失败，退出。")
        sys.exit(1)

    # 2. 下载 ONNX 模型文件（轻量级推理引擎所需）
    if not download_file(model_url, model_target):
        print("[-] ONNX 模型下载失败，退出。")
        sys.exit(1)

    print("[+] ONNX 模型及词表下载配置完毕！")
    sys.exit(0)


if __name__ == "__main__":
    main()
