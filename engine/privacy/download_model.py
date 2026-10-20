"""本地微调大模型一键下载工具 / One-Click Local Fine-tuned LLM Download Utility.

中文说明：
支持通过 ModelScope (首选) 或 Hugging Face 镜像站高速下载 Qwen3.5-0.8B-Privacy-Classifier-Smoother
微调模型权重（医疗隐私分类专用纯文本模型）。
下载策略采用优先级回退机制：
1. 优先使用 ModelScope SDK（国内速度最快）。
2. 若 ModelScope 不可用则回退至 Hugging Face 镜像站（hf-mirror.com）。

模型默认存储在项目根目录下的 .models/Qwen3.5-0.8B-Privacy-Classifier-Smoother 中，
供 llm_engines.py 中的 Qwen3Classifier 延迟加载使用。

English Description:
One-click download utility for the Qwen3.5-0.8B-Privacy-Classifier-Smoother fine-tuned LLM
(medical privacy classification text-only model).
Supports ModelScope (preferred) and Hugging Face mirror as fallback.
The model is stored under .models/Qwen3.5-0.8B-Privacy-Classifier-Smoother in the project root
for lazy-loading by Qwen3Classifier in llm_engines.py.

Usage:
    python -m engine.privacy.download_model
"""

import os
import sys


def download_via_modelscope(model_id: str, local_dir: str) -> bool:
    """尝试通过 ModelScope SDK 下载模型 / Download Model via ModelScope SDK.

    中文说明：
    ModelScope 为国内首选下载通道，速度最快。需要预装 modelscope 库。
    下载整个模型仓库快照（含权重、tokenizer、config 等）到本地目录。

    English Description:
    Attempts to download the model via ModelScope SDK (fastest in China).
    Requires the modelscope package. Downloads the full model snapshot
    (weights, tokenizer, config, etc.) to the specified local directory.

    Args:
        model_id: ModelScope 模型标识符 / ModelScope model identifier
            (e.g. "Qwen/Qwen3.5-0.8B-Privacy-Classifier-Smoother").
        local_dir: 本地保存目录 / Local directory to save the model.

    Returns:
        True 表示下载成功 / True if download succeeded.
    """
    try:
        from modelscope import snapshot_download
        print(f"[*] 正在通过 ModelScope 下载模型 {model_id} ...")
        snapshot_download(model_id, local_dir=local_dir)
        print("[+] ModelScope 下载完成！")
        return True
    except ImportError:
        print("[-] 未检测到 modelscope 库。可通过 'pip install modelscope' 安装以获取最快下载速度。")
        return False
    except Exception as e:
        print(f"[-] ModelScope 下载遇到错误: {e}")
        return False


def download_via_huggingface(model_id: str, local_dir: str) -> bool:
    """尝试通过 Hugging Face 镜像源下载模型 / Download Model via Hugging Face Mirror.

    中文说明：
    作为 ModelScope 的回退方案。默认使用 hf-mirror.com 国内镜像加速，
    可通过环境变量 HF_ENDPOINT 自定义镜像地址。
    跳过 .msgpack/.h5/.ot 等非必要格式文件以节省带宽。

    English Description:
    Fallback download channel when ModelScope is unavailable.
    Defaults to hf-mirror.com for accelerated access in China;
    customizable via HF_ENDPOINT environment variable.
    Skips non-essential format files (.msgpack/.h5/.ot) to save bandwidth.

    Args:
        model_id: Hugging Face 仓库 ID / Hugging Face repository ID
            (e.g. "Qwen/Qwen3.5-0.8B-Privacy-Classifier-Smoother").
        local_dir: 本地保存目录 / Local directory to save the model.

    Returns:
        True 表示下载成功 / True if download succeeded.
    """
    try:
        # 如果未设置，默认配置国内高速镜像源
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from huggingface_hub import snapshot_download
        print(f"[*] 正在通过 Hugging Face 镜像下载模型 {model_id} (HF_ENDPOINT={os.environ['HF_ENDPOINT']}) ...")
        snapshot_download(repo_id=model_id, local_dir=local_dir, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
        print("[+] Hugging Face 下载完成！")
        return True
    except ImportError:
        print("[-] 未检测到 huggingface_hub 库。可通过 'pip install huggingface_hub' 安装。")
        return False
    except Exception as e:
        print(f"[-] Hugging Face 下载遇到错误: {e}")
        return False


def main():
    """模型下载主入口 / Main Entry Point for Model Download.

    执行步骤 / Execution Steps:
    1. 确定模型保存路径（项目根目录/.models/Qwen3.5-0.8B-Privacy-Classifier-Smoother）。
       (Determine model save path: project_root/.models/Qwen3.5-0.8B-Privacy-Classifier-Smoother)
    2. 优先尝试 ModelScope 下载。
       (Try ModelScope download first)
    3. 若失败则回退至 Hugging Face 镜像下载。
       (Fall back to Hugging Face mirror if ModelScope fails)
    4. 根据结果输出成功/失败的信息并设置退出码。
       (Output success/failure message and set exit code)
    """
    from engine.env_loader import load_env_file
    load_env_file()

    # 微调模型标识（本地合并导出的 Qwen3.5-0.8B 隐私分类器）
    model_id = os.environ.get("PRIVACY_LLM_MODEL_NAME", "Qwen/Qwen3.5-0.8B-Privacy-Classifier-Smoother")

    # 优先读取环境变量 PRIVACY_LLM_MODEL_PATH，未配置时默认存放在项目根目录下的 .models 目录中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    local_dir = os.environ.get(
        "PRIVACY_LLM_MODEL_PATH",
        os.path.join(project_root, ".models", "Qwen3.5-0.8B-Privacy-Classifier-Smoother")
    )

    print(f"[*] 目标保存路径: {local_dir}")

    # 如果模型目录已存在且包含权重文件，跳过下载
    model_weight = os.path.join(local_dir, "model.safetensors")
    if os.path.exists(model_weight):
        print(f"[+] 模型已存在: {local_dir}，跳过下载。")
        sys.exit(0)

    os.makedirs(local_dir, exist_ok=True)

    # 优先使用 ModelScope（国内速度最快）
    success = download_via_modelscope(model_id, local_dir)
    if not success:
        print("[*] 正在切换至 Hugging Face 镜像重试...")
        success = download_via_huggingface(model_id, local_dir)

    if success:
        print(f"[+] 大模型下载成功！已保存在: {local_dir}")
        sys.exit(0)
    else:
        print("[-] 错误：无法下载模型。请确保安装了 modelscope 或 huggingface_hub:")
        print("    pip install modelscope huggingface_hub")
        sys.exit(1)


if __name__ == "__main__":
    main()
