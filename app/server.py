from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .asr_engine import AudioFormat, Qwen3AsrEngine
from .config import AppConfig
from .protocol import read_message, write_message

LOGGER = logging.getLogger("wyoming-sherpa-onnx")
_MAX_AUDIO_SECONDS = 30.0


def _is_disconnect_error(exc: BaseException) -> bool:
    return isinstance(exc, (BrokenPipeError, ConnectionResetError))


@dataclass(slots=True)
class SessionState:
    transcribe_opts: dict[str, Any]
    audio_format: AudioFormat | None
    stream: Any | None
    chunk_count: int
    total_bytes: int
    over_limit: bool


class WyomingAsrServer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.engine = Qwen3AsrEngine(
            model_dir=cfg.model_dir,
            sample_rate=cfg.sample_rate,
            num_threads=cfg.num_threads,
            hotwords=cfg.hotwords,
        )
        self._info_cache: dict[str, Any] | None = None
        self._server: asyncio.AbstractServer | None = None

    async def handle_client(self, reader, writer) -> None:
        peer = self._get_peername(writer)
        LOGGER.info("Client connected: %s", peer)
        state = SessionState(
            transcribe_opts={},
            audio_format=None,
            stream=None,
            chunk_count=0,
            total_bytes=0,
            over_limit=False,
        )

        try:
            while True:
                msg = await read_message(reader)
                data = {**msg.data, **msg.extra_data}
                LOGGER.debug(
                    "[%s] Received message type=%s data_keys=%s payload_bytes=%s",
                    peer,
                    msg.msg_type,
                    sorted(data.keys()),
                    len(msg.payload),
                )

                if msg.msg_type == "describe":
                    await write_message(writer, "info", self._get_info())
                elif msg.msg_type == "transcribe":
                    state.transcribe_opts = data
                elif msg.msg_type == "audio-start":
                    state.stream = self.engine.create_stream()
                    state.audio_format = AudioFormat(
                        rate=int(data.get("rate", self.cfg.sample_rate)),
                        width=int(data.get("width", 2)),
                        channels=int(data.get("channels", 1)),
                    )
                    state.chunk_count = 0
                    state.total_bytes = 0
                    state.over_limit = False
                    LOGGER.debug(
                        "[%s] Audio stream started: %dHz, %d-bit, %d channels",
                        peer,
                        state.audio_format.rate,
                        state.audio_format.width * 8,
                        state.audio_format.channels,
                    )
                elif msg.msg_type == "audio-chunk":
                    if msg.payload and state.stream is not None and state.audio_format is not None:
                        bytes_per_sample = state.audio_format.width * state.audio_format.channels
                        total_bytes = state.total_bytes + len(msg.payload)
                        audio_duration = total_bytes / (bytes_per_sample * state.audio_format.rate)
                        if audio_duration > _MAX_AUDIO_SECONDS:
                            state.over_limit = True
                            state.stream = None
                            LOGGER.warning(
                                "[%s] Audio exceeded max duration %.2fs, dropping request at %.2fs",
                                peer,
                                _MAX_AUDIO_SECONDS,
                                audio_duration,
                            )
                            continue
                        self.engine.feed_audio_to_stream(state.stream, msg.payload, state.audio_format)
                        state.chunk_count += 1
                        state.total_bytes += len(msg.payload)
                        LOGGER.debug(
                            "[%s] Audio chunk #%d received (%.1f KB)",
                            peer,
                            state.chunk_count,
                            len(msg.payload) / 1024,
                        )
                elif msg.msg_type == "audio-stop":
                    text = ""
                    if state.over_limit:
                        LOGGER.warning(
                            "[%s] Returning empty transcript because audio exceeded %.2fs limit",
                            peer,
                            _MAX_AUDIO_SECONDS,
                        )
                    elif state.stream is not None:
                        audio_duration = 0.0
                        if state.audio_format and state.total_bytes > 0:
                            bytes_per_sample = state.audio_format.width * state.audio_format.channels
                            audio_duration = state.total_bytes / (bytes_per_sample * state.audio_format.rate)

                        start_time = time.time()
                        text = self.engine.finish_stream(state.stream)
                        infer_time = time.time() - start_time
                        rtf = infer_time / audio_duration if audio_duration > 0 else 0.0

                        state.stream = None

                        LOGGER.info(
                            "[%s] Recognition completed: audio=%.2fs, inference=%.3fs, RTF=%.2f, result=\"%s\"",
                            peer,
                            audio_duration,
                            infer_time,
                            rtf,
                            text if text else "(silence)",
                        )

                    await write_message(
                        writer,
                        "transcript",
                        {
                            "text": text,
                            "language": state.transcribe_opts.get("language", "zh"),
                        },
                    )
                    LOGGER.debug("[%s] Sent transcript: %r", peer, text)
                    state.over_limit = False
                else:
                    LOGGER.debug("Ignoring unsupported message type: %s", msg.msg_type)
        except EOFError:
            LOGGER.info("Client disconnected: %s", peer)
        except (BrokenPipeError, ConnectionResetError) as exc:
            LOGGER.info("Client connection closed while streaming: %s (%s)", peer, exc)
        except asyncio.CancelledError:
            LOGGER.info("Client session cancelled: %s", peer)
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Client session error: %s", exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass

            try:
                await writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                if not _is_disconnect_error(exc):
                    LOGGER.debug("Error while closing client stream %s: %s", peer, exc)

    def _get_peername(self, writer) -> str:
        """获取客户端地址，安全处理异常。"""
        try:
            peer = writer.get_extra_info("peername")
            return str(peer) if peer else "unknown"
        except Exception:
            return "unknown"

    def _get_info(self) -> dict[str, Any]:
        """获取 Wyoming 协议 info 响应（带缓存）。"""
        if self._info_cache is None:
            self._info_cache = {
                "asr": [
                    {
                        "name": self.cfg.model_name,
                        "attribution": {
                            "name": "k2-fsa/sherpa-onnx",
                            "url": "https://github.com/k2-fsa/sherpa-onnx",
                        },
                        "installed": True,
                        "models": [
                            {
                                "name": self.cfg.model_name,
                                "languages": ["zh", "en"],
                                "attribution": {
                                    "name": "k2-fsa/sherpa-onnx",
                                    "url": "https://github.com/k2-fsa/sherpa-onnx",
                                },
                                "installed": True,
                                "description": "Qwen3-ASR via sherpa-onnx",
                            }
                        ],
                        "supports_transcript_streaming": False,
                    }
                ],
            }
        return self._info_cache

    async def run(self) -> None:
        self._server = await asyncio.start_server(self.handle_client, self.cfg.host, self.cfg.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in self._server.sockets or [])
        LOGGER.info("Wyoming server listening on %s", addrs)
        LOGGER.info("Wyoming Qwen3-ASR service started successfully")
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is None:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None
        LOGGER.info("Wyoming server stopped")
