"""Model downloader for sherpa-onnx FunASR Nano models."""

from __future__ import annotations

import logging
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger("wyoming-sherpa-funasr")

# 模型配置
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2"
MODEL_NAME = "sherpa-onnx-funasr-nano-int8-2025-12-30"

# 必需的模型文件
REQUIRED_FILES = [
    "encoder_adaptor.int8.onnx",
    "llm.int8.onnx",
    "embedding.int8.onnx",
    "Qwen3-0.6B/tokenizer.json",
    "Qwen3-0.6B/vocab.json",
    "Qwen3-0.6B/merges.txt",
]

# 下载块大小（64KB）
DOWNLOAD_CHUNK_SIZE = 65536
# 进度报告间隔（每 1% 报告一次）
PROGRESS_REPORT_INTERVAL = 0.01


def check_model_exists(model_dir: Path) -> bool:
    """检查模型文件是否存在且完整。

    Args:
        model_dir: 模型目录路径

    Returns:
        True 如果所有必需文件存在，否则 False
    """
    if not model_dir.exists():
        return False

    for filename in REQUIRED_FILES:
        file_path = model_dir / filename
        if not file_path.exists():
            LOGGER.debug("Missing model file: %s", filename)
            return False

    LOGGER.info("Model files verified: %s", model_dir)
    return True


def download_model(
    dest_dir: Path,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Path:
    """下载并解压模型文件。

    Args:
        dest_dir: 目标目录（父目录）
        progress_callback: 进度回调函数 (downloaded, total, percent)

    Returns:
        模型目录路径

    Raises:
        RuntimeError: 下载或解压失败
    """
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_path = dest_dir / MODEL_NAME
    if check_model_exists(model_path):
        LOGGER.info("Model already exists, skipping download")
        return model_path

    LOGGER.info("Downloading model from %s", MODEL_URL)

    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            # 获取文件大小
            with urllib.request.urlopen(MODEL_URL, timeout=300) as response:
                total_size = int(response.getheader("Content-Length", 0))

            # 下载文件（使用更大的块）
            downloaded = 0
            last_report_percent = 0.0

            with urllib.request.urlopen(MODEL_URL, timeout=300) as response, open(tmp_path, "wb") as out_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    # 报告进度（每 1% 报告一次）
                    if total_size > 0:
                        current_percent = downloaded / total_size * 100
                        if current_percent - last_report_percent >= PROGRESS_REPORT_INTERVAL * 100:
                            if progress_callback:
                                progress_callback(downloaded, total_size, current_percent)
                            last_report_percent = current_percent

            # 最终进度报告
            if progress_callback and total_size > 0:
                progress_callback(downloaded, total_size, 100.0)

            LOGGER.info("Download completed, extracting...")

            # 解压文件
            with tarfile.open(tmp_path, "r:bz2") as tar:
                tar.extractall(dest_dir)

            # 验证解压后的文件
            if not check_model_exists(model_path):
                raise RuntimeError("Model verification failed after extraction")

            LOGGER.info("Model downloaded and extracted to: %s", model_path)
            return model_path

        except Exception as e:
            LOGGER.error("Download failed: %s", e)
            raise RuntimeError(f"Failed to download model: {e}") from e
        finally:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()


def format_size(size_bytes: int) -> str:
    """格式化文件大小显示。

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的大小字符串 (如 "100.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
