from __future__ import annotations

import asyncio
import logging
import signal
import socket
import sys
from pathlib import Path

from app.config import parse_args
from app.discovery import WyomingDiscovery
from app.server import WyomingAsrServer


def _resolve_advertise_host(bind_host: str) -> str:
    """Resolve the host to advertise to clients."""
    if bind_host not in ("0.0.0.0", "::"):
        return bind_host
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


async def main() -> None:
    cfg = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("wyoming-sherpa-onnx")

    default_speaker_refs_dir = cfg.speaker_reference_dir
    if cfg.speaker_gate:
        default_speaker_refs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Speaker reference directory: %s", default_speaker_refs_dir)
        root_wavs = sorted(default_speaker_refs_dir.glob("*.wav"))
        if root_wavs:
            logger.warning(
                "Found %d wav(s) directly under %s; they are ignored. "
                "Please place files under subdirectories like %s/<speaker_id>/*.wav.",
                len(root_wavs),
                default_speaker_refs_dir,
                default_speaker_refs_dir,
            )

    if cfg.speaker_gate and not cfg.speaker_reference_wavs:
        logger.error(
            "Speaker gate is enabled but no reference wavs provided. "
            "Put wavs under %s/<speaker_id>/*.wav or use --speaker-reference-wavs "
            "with files in speaker subdirectories.",
            default_speaker_refs_dir,
        )
        sys.exit(1)
    if cfg.speaker_gate:
        speaker_ids: set[str] = set()
        for p in cfg.speaker_reference_wavs:
            resolved = p.resolve()
            try:
                rel = resolved.relative_to(cfg.speaker_reference_dir)
                speaker_ids.add(rel.parts[0] if len(rel.parts) >= 2 else "default")
            except ValueError:
                speaker_ids.add(resolved.parent.name or "default")
        logger.info(
            "Speaker gate references loaded (%d wavs, %d speakers): %s",
            len(cfg.speaker_reference_wavs),
            len(speaker_ids),
            ", ".join(str(p) for p in cfg.speaker_reference_wavs),
        )

    from app.downloader import (
        check_denoise_model_exists,
        check_model_exists,
        check_speaker_model_exists,
        download_denoise_model,
        download_model,
        download_speaker_model,
        get_model_hint_url,
        get_speaker_model_hint_url,
    )

    if not check_model_exists(cfg.model_dir):
        if cfg.auto_download:
            logger.info("Model not found, starting auto-download...")
            try:
                model_path = download_model(
                    model_dir=cfg.model_dir,
                    model_name=cfg.model_name,
                )
                logger.info("Model downloaded to: %s", model_path)
                cfg.model_dir = model_path
            except Exception as exc:
                logger.error("Failed to download model: %s", exc)
                logger.error("Please download model manually from:\n%s", get_model_hint_url())
                sys.exit(1)
        else:
            logger.error("Qwen3-ASR model directory is invalid: %s", cfg.model_dir)
            logger.error(
                "The directory must contain ONNX model files and a tokenizer/tokens file."
            )
            logger.error("Download URL:\n%s", get_model_hint_url())
            sys.exit(1)
    else:
        logger.info("Model verified: %s", cfg.model_dir)

    if cfg.speaker_gate:
        speaker_model_path = cfg.speaker_model_dir / cfg.speaker_model_file
        if not check_speaker_model_exists(speaker_model_path):
            if cfg.auto_download:
                logger.info("Speaker model not found, starting auto-download...")
                try:
                    speaker_model_path = download_speaker_model(
                        model_dir=cfg.speaker_model_dir,
                        model_file=cfg.speaker_model_file,
                        model_url=cfg.speaker_model_url,
                    )
                    logger.info("Speaker model downloaded to: %s", speaker_model_path)
                except Exception as exc:
                    logger.error("Failed to download speaker model: %s", exc)
                    logger.error(
                        "Please download speaker model manually from:\n%s",
                        get_speaker_model_hint_url(cfg.speaker_model_url),
                    )
                    sys.exit(1)
            else:
                logger.error("Speaker model file is invalid: %s", speaker_model_path)
                logger.error(
                    "Please download speaker model manually from:\n%s",
                    get_speaker_model_hint_url(cfg.speaker_model_url),
                )
                sys.exit(1)

    if cfg.denoise_enabled:
        denoise_model_path = cfg.denoise_model_dir / cfg.denoise_model_file
        if not check_denoise_model_exists(denoise_model_path):
            if cfg.auto_download:
                logger.info("Denoise model not found, starting auto-download...")
                try:
                    denoise_model_path = download_denoise_model(
                        model_dir=cfg.denoise_model_dir,
                        model_file=cfg.denoise_model_file,
                        model_url=cfg.denoise_model_url,
                    )
                    logger.info("Denoise model downloaded to: %s", denoise_model_path)
                except Exception as exc:
                    logger.error("Failed to download denoise model: %s", exc)
                    logger.error(
                        "Please download denoise model manually from:\n%s",
                        cfg.denoise_model_url,
                    )
                    sys.exit(1)
            else:
                logger.error("Denoise model file is invalid: %s", denoise_model_path)
                logger.error(
                    "Please download denoise model manually from:\n%s",
                    cfg.denoise_model_url,
                )
                sys.exit(1)

    discovery = None
    if cfg.enable_zeroconf:
        discovery = WyomingDiscovery(
            service_name=cfg.service_name,
            host=_resolve_advertise_host(cfg.host),
            port=cfg.port,
            model_name=cfg.model_name,
        )
        await discovery.start()

    server = WyomingAsrServer(cfg)
    shutdown_event = asyncio.Event()
    server_task: asyncio.Task[None] | None = None

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows 不支持 add_signal_handler
            pass

    try:
        server_task = asyncio.create_task(server.run())
        await asyncio.sleep(0)
        logger.info("Service ready at %s:%s", cfg.host, cfg.port)
        await shutdown_event.wait()
    except asyncio.CancelledError:
        logger.info("Server cancelled")
    finally:
        await server.stop()
        if server_task is not None:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
        if discovery is not None:
            await discovery.stop()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
