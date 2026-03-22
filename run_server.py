from __future__ import annotations

import asyncio
import logging
import os
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


def _download_progress(downloaded: int, total: int, percent: float) -> None:
    """显示下载进度。

    Args:
        downloaded: 已下载字节数
        total: 总字节数
        percent: 百分比
    """
    from app.downloader import format_size

    if total > 0:
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = "=" * filled + "-" * (bar_length - filled)
        sys.stdout.write(
            f"\rDownloading: [{bar}] {percent:.1f}% "
            f"({format_size(downloaded)}/{format_size(total)})"
        )
        sys.stdout.flush()


async def main() -> None:
    cfg = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("wyoming-sherpa-onnx")

    # 检查和下载模型
    from app.downloader import check_model_exists, download_model

    if not check_model_exists(cfg.model_dir):
        if cfg.auto_download:
            logger.info("Model not found, starting auto-download...")
            try:
                model_path = download_model(
                    dest_dir=cfg.model_dir.parent,
                    progress_callback=_download_progress,
                )
                logger.info("")  # 换行
                logger.info("Model downloaded to: %s", model_path)
                cfg.model_dir = model_path
            except Exception as e:
                logger.error("Failed to download model: %s", e)
                logger.error(
                    "Please download model manually or run with --no-auto-download "
                    "if you want to skip model validation."
                )
                sys.exit(1)
        else:
            logger.error("Model directory does not exist: %s", cfg.model_dir)
            logger.error(
                "Please download the model manually from:\n"
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                "sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2\n\n"
                "Or run with --auto-download to enable automatic download."
            )
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
        await server.run()
        logger.info("Service ready at %s:%s", cfg.host, cfg.port)
        await shutdown_event.wait()
    except asyncio.CancelledError:
        logger.info("Server cancelled")
    finally:
        if discovery is not None:
            await discovery.stop()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
