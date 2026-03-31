from __future__ import annotations

import asyncio
import logging
import signal
import socket
import sys

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


_LAST_DOWNLOAD_LOG_PERCENT = -1


def _download_progress(downloaded: int, total: int, percent: float) -> None:
    """Emit Docker-friendly download progress logs."""
    global _LAST_DOWNLOAD_LOG_PERCENT

    from app.downloader import format_size

    if total <= 0:
        return

    percent_bucket = int(percent)
    if percent_bucket == _LAST_DOWNLOAD_LOG_PERCENT and percent < 100.0:
        return

    _LAST_DOWNLOAD_LOG_PERCENT = percent_bucket
    logging.getLogger("wyoming-sherpa-onnx").info(
        "Model download progress: %.1f%% (%s/%s)",
        percent,
        format_size(downloaded),
        format_size(total),
    )


async def main() -> None:
    global _LAST_DOWNLOAD_LOG_PERCENT

    cfg = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("wyoming-sherpa-onnx")

    from app.downloader import MODEL_URL, check_model_exists, download_model

    if not check_model_exists(cfg.model_dir):
        if cfg.auto_download:
            logger.info("Model not found, starting auto-download...")
            _LAST_DOWNLOAD_LOG_PERCENT = -1
            try:
                model_path = download_model(
                    dest_dir=cfg.model_dir.parent,
                    progress_callback=_download_progress,
                )
                logger.info("Model downloaded to: %s", model_path)
                cfg.model_dir = model_path
            except Exception as exc:
                logger.error("Failed to download model: %s", exc)
                logger.error("Please download model manually from:\n%s", MODEL_URL)
                sys.exit(1)
        else:
            logger.error("Qwen3-ASR model directory is invalid: %s", cfg.model_dir)
            logger.error(
                "The directory must contain ONNX model files and a tokenizer/tokens file."
            )
            logger.error("Download URL:\n%s", MODEL_URL)
            sys.exit(1)
    else:
        logger.info("Model verified: %s", cfg.model_dir)

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
