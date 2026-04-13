"""Qwen3-ASR model validation and download helpers."""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

LOGGER = logging.getLogger("wyoming-sherpa-onnx")

MODEL_REPO_ID = "zengshuishui/Qwen3-ASR-onnx"
MODEL_REPO_PAGE = "https://www.modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/files"

CORE_REQUIRED_FILES = (
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
)


MODEL_PROFILES: dict[str, str] = {
    "0.6b": "model_0.6B",
    "1.7b": "model_1.7B",
}


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

    has_tokenizer_assets = all(
        (tokenizer_dir / name).exists()
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json")
    )
    if not has_tokenizer_assets:
        LOGGER.debug(
            "Tokenizer directory is missing required files (vocab/merges/tokenizer_config) under model dir: %s",
            model_dir,
        )
        return False

    LOGGER.debug("Qwen3-ASR model files verified: %s", model_dir)
    return True


def select_model_profile_key(model_dir: Path, model_name: str = "") -> str:
    """Choose 0.6B/1.7B profile key based on selected model directory/name."""
    hint = f"{model_dir.name} {model_name}".lower()
    if "1.7" in hint:
        return "1.7b"
    return "0.6b"


def get_model_hint_url() -> str:
    return MODEL_REPO_PAGE


def check_speaker_model_exists(model_file: Path) -> bool:
    if not model_file.exists() or not model_file.is_file():
        return False
    if model_file.suffix.lower() != ".onnx":
        LOGGER.debug("Speaker model is not ONNX: %s", model_file)
        return False
    return True


def check_denoise_model_exists(model_file: Path) -> bool:
    if not model_file.exists() or not model_file.is_file():
        return False
    if model_file.suffix.lower() != ".onnx":
        LOGGER.debug("Denoise model is not ONNX: %s", model_file)
        return False
    return True


def get_speaker_model_hint_url(model_url: str) -> str:
    return model_url


def download_speaker_model(model_dir: Path, model_file: str, model_url: str) -> Path:
    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    target_file = model_dir / model_file

    if check_speaker_model_exists(target_file):
        LOGGER.info("Speaker model already exists, skipping download: %s", target_file)
        return target_file

    LOGGER.info(
        "Downloading speaker model: file=%s target=%s",
        model_file,
        target_file,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="speaker-ms-") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            tmp_file = tmp_dir / model_file
            with urllib.request.urlopen(model_url, timeout=120) as resp, tmp_file.open("wb") as f:
                shutil.copyfileobj(resp, f)
            shutil.copy2(tmp_file, target_file)

        if not check_speaker_model_exists(target_file):
            raise RuntimeError("Speaker model verification failed after download")
        LOGGER.info("Speaker model downloaded to: %s", target_file)
        return target_file
    except Exception as exc:
        LOGGER.error("Speaker model download failed: %s", exc)
        raise RuntimeError(f"Failed to download speaker model: {exc}") from exc


def download_denoise_model(model_dir: Path, model_file: str, model_url: str) -> Path:
    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    target_file = model_dir / model_file

    if check_denoise_model_exists(target_file):
        LOGGER.info("Denoise model already exists, skipping download: %s", target_file)
        return target_file

    LOGGER.info("Downloading denoise model: file=%s target=%s", model_file, target_file)
    try:
        with tempfile.TemporaryDirectory(prefix="denoise-model-") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            tmp_file = tmp_dir / model_file
            with urllib.request.urlopen(model_url, timeout=120) as resp, tmp_file.open("wb") as f:
                shutil.copyfileobj(resp, f)
            shutil.copy2(tmp_file, target_file)

        if not check_denoise_model_exists(target_file):
            raise RuntimeError("Denoise model verification failed after download")
        LOGGER.info("Denoise model downloaded to: %s", target_file)
        return target_file
    except Exception as exc:
        LOGGER.error("Denoise model download failed: %s", exc)
        raise RuntimeError(f"Failed to download denoise model: {exc}") from exc


def _copy_tokenizer_from_snapshot(tmp_dir: Path, model_path: Path) -> bool:
    """Copy tokenizer assets from repository root tokenizer directory."""
    candidate = tmp_dir / "tokenizer"
    vocab = candidate / "vocab.json"
    merges = candidate / "merges.txt"
    tokenizer_config = candidate / "tokenizer_config.json"
    if not (vocab.exists() and merges.exists() and tokenizer_config.exists()):
        return False

    dst_dir = model_path / "tokenizer"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocab, dst_dir / "vocab.json")
    shutil.copy2(merges, dst_dir / "merges.txt")
    shutil.copy2(tokenizer_config, dst_dir / "tokenizer_config.json")
    LOGGER.info("Tokenizer assets copied from snapshot root path: %s", candidate)
    return True


def download_model(
    model_dir: Path,
    model_name: str = "",
) -> Path:
    """Download only required files for the selected Qwen3-ASR model profile."""
    profile_key = select_model_profile_key(model_dir=model_dir, model_name=model_name)
    profile_subdir = MODEL_PROFILES[profile_key]
    model_path = model_dir.resolve()
    model_path.mkdir(parents=True, exist_ok=True)

    if check_model_exists(model_path):
        LOGGER.info("Model already exists, skipping download: %s", model_path)
        return model_path

    core_patterns = [f"{profile_subdir}/{name}" for name in CORE_REQUIRED_FILES]
    LOGGER.info(
        "Downloading required files from ModelScope: repo=%s profile=%s target=%s",
        MODEL_REPO_ID,
        profile_key,
        model_path,
    )
    LOGGER.debug("ModelScope core allow_patterns=%s", core_patterns)

    try:
        with tempfile.TemporaryDirectory(prefix="qwen3-asr-ms-") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            snapshot_download(
                model_id=MODEL_REPO_ID,
                revision="master",
                allow_patterns=core_patterns,
                local_dir=str(tmp_dir),
            )

            source_root = tmp_dir / profile_subdir
            if not source_root.exists():
                raise RuntimeError(f"Missing downloaded source directory: {source_root}")

            for rel in CORE_REQUIRED_FILES:
                src = source_root / rel
                dst = model_path / rel
                if not src.exists():
                    raise RuntimeError(f"Required file missing from ModelScope snapshot: {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            tokenizer_patterns = [
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer_config.json",
            ]
            LOGGER.debug("ModelScope tokenizer allow_patterns=%s", tokenizer_patterns)
            snapshot_download(
                model_id=MODEL_REPO_ID,
                revision="master",
                allow_patterns=tokenizer_patterns,
                local_dir=str(tmp_dir),
            )

            if not _copy_tokenizer_from_snapshot(tmp_dir=tmp_dir, model_path=model_path):
                raise RuntimeError(
                    "Tokenizer files not found in snapshot root tokenizer directory"
                )

        if not check_model_exists(model_path):
            raise RuntimeError("Model verification failed after extraction")

        LOGGER.info("Model downloaded to: %s", model_path)
        return model_path
    except Exception as exc:
        LOGGER.error("Download failed: %s", exc)
        raise RuntimeError(f"Failed to download model: {exc}") from exc


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
