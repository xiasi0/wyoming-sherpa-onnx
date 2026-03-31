"""Qwen3-ASR model validation and download helpers."""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger("wyoming-sherpa-onnx")

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"
)
MODEL_NAME = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"

DOWNLOAD_CHUNK_SIZE = 65536
PROGRESS_REPORT_INTERVAL = 0.01


def check_model_exists(model_dir: Path) -> bool:
    """Verify that a Qwen3-ASR model directory looks usable."""
    if not model_dir.exists() or not model_dir.is_dir():
        return False

    conv_frontend = model_dir / "conv_frontend.onnx"
    encoder = model_dir / "encoder.int8.onnx"
    decoder = model_dir / "decoder.int8.onnx"
    tokenizer_dir = model_dir / "tokenizer"

    if not conv_frontend.exists():
        LOGGER.debug("Missing conv_frontend.onnx under model dir: %s", model_dir)
        return False

    if not encoder.exists():
        LOGGER.debug("Missing encoder.int8.onnx under model dir: %s", model_dir)
        return False

    if not decoder.exists():
        LOGGER.debug("Missing decoder.int8.onnx under model dir: %s", model_dir)
        return False

    if not tokenizer_dir.is_dir():
        LOGGER.debug("Missing tokenizer directory under model dir: %s", model_dir)
        return False

    has_tokenizer_assets = any((tokenizer_dir / name).exists() for name in ("vocab.json", "merges.txt"))
    if not has_tokenizer_assets:
        LOGGER.debug("Tokenizer directory is missing vocab/merges files under model dir: %s", model_dir)
        return False

    LOGGER.debug("Qwen3-ASR model files verified: %s", model_dir)
    return True


def download_model(
    dest_dir: Path,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Path:
    """Download and extract the default Qwen3-ASR model."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_path = dest_dir / MODEL_NAME
    if check_model_exists(model_path):
        LOGGER.info("Model already exists, skipping download")
        return model_path

    LOGGER.info("Downloading model from %s", MODEL_URL)

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".tar.bz2")
        os.close(fd)

        with urllib.request.urlopen(MODEL_URL, timeout=300) as response:
            total_size = int(response.getheader("Content-Length", 0))

        downloaded = 0
        last_report_percent = 0.0

        with urllib.request.urlopen(MODEL_URL, timeout=300) as response, open(tmp_path, "wb") as out_file:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    current_percent = downloaded / total_size * 100
                    if current_percent - last_report_percent >= PROGRESS_REPORT_INTERVAL * 100:
                        if progress_callback is not None:
                            progress_callback(downloaded, total_size, current_percent)
                        last_report_percent = current_percent

        if progress_callback is not None and total_size > 0:
            progress_callback(downloaded, total_size, 100.0)

        LOGGER.info("Download completed, extracting...")
        with tarfile.open(tmp_path, "r:bz2") as tar:
            tar.extractall(dest_dir)

        if not check_model_exists(model_path):
            raise RuntimeError("Model verification failed after extraction")

        LOGGER.info("Model downloaded and extracted to: %s", model_path)
        return model_path
    except Exception as exc:
        LOGGER.error("Download failed: %s", exc)
        raise RuntimeError(f"Failed to download model: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
