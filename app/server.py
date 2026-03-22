from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .asr_engine import AudioFormat, FunAsrNanoEngine
from .config import AppConfig
from .protocol import read_message, write_message

LOGGER = logging.getLogger("wyoming-sherpa-funasr")

# 预计算常量，避免重复计算
_BYTES_PER_SAMPLE = {2: 2, 4: 4, 1: 1}  # width -> bytes


@dataclass(slots=True)
class SessionState:
    transcribe_opts: dict[str, Any]
    audio_format: AudioFormat | None
    stream: Any | None  # sherpa_onnx.OfflineStream
    chunk_count: int
    total_bytes: int


class WyomingAsrServer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.engine = FunAsrNanoEngine(
            model_dir=cfg.model_dir,
            sample_rate=cfg.sample_rate,
            num_threads=cfg.num_threads,
        )
        # 缓存 info 响应，避免每次构建
        self._info_cache: dict[str, Any] | None = None

    async def handle_client(self, reader, writer) -> None:
        peer = writer.get_extra_info("peername")
        LOGGER.info("Client connected: %s", peer)
        state = SessionState(
            transcribe_opts={},
            audio_format=None,
            stream=None,
            chunk_count=0,
            total_bytes=0,
        )

        try:
            while True:
                msg = await read_message(reader)
                data = {**msg.data, **msg.extra_data}

                if msg.msg_type == "describe":
                    await write_message(writer, "info", self._get_info())
                elif msg.msg_type == "transcribe":
                    state.transcribe_opts = data
                elif msg.msg_type == "audio-start":
                    # 创建新的识别流
                    state.stream = self.engine.create_stream()
                    state.audio_format = AudioFormat(
                        rate=int(data.get("rate", self.cfg.sample_rate)),
                        width=int(data.get("width", 2)),
                        channels=int(data.get("channels", 1)),
                    )
                    state.chunk_count = 0
                    state.total_bytes = 0
                    LOGGER.debug(
                        "[%s] Audio stream started: %dHz, %d-bit, %d channels",
                        peer,
                        state.audio_format.rate,
                        state.audio_format.width * 8,
                        state.audio_format.channels,
                    )
                elif msg.msg_type == "audio-chunk":
                    # 流式馈送音频数据
                    if msg.payload and state.stream is not None and state.audio_format is not None:
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
                    # 完成识别
                    text = ""
                    if state.stream is not None:
                        # 计算音频时长
                        audio_duration = 0.0
                        if state.audio_format and state.total_bytes > 0:
                            bytes_per_sample = state.audio_format.width * state.audio_format.channels
                            audio_duration = state.total_bytes / (bytes_per_sample * state.audio_format.rate)

                        # 执行识别并计时
                        start_time = time.time()
                        text = self.engine.finish_stream(state.stream)
                        infer_time = time.time() - start_time
                        state.stream = None

                        # 计算 RTF (Real Time Factor)
                        rtf = infer_time / audio_duration if audio_duration > 0 else 0.0

                        # 打印识别结果
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
                else:
                    LOGGER.debug("Ignoring unsupported message type: %s", msg.msg_type)
        except EOFError:
            LOGGER.info("Client disconnected: %s", peer)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Client session error: %s", exc)
        finally:
            writer.close()
            await writer.wait_closed()

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
                                "languages": ["zh", "en", "ja"],
                                "attribution": {
                                    "name": "k2-fsa/sherpa-onnx",
                                    "url": "https://github.com/k2-fsa/sherpa-onnx",
                                },
                                "installed": True,
                                "description": "FunASR Nano 2512 INT8 via sherpa-onnx",
                            }
                        ],
                        "supports_transcript_streaming": False,
                    }
                ],
            }
        return self._info_cache

    async def run(self) -> None:
        server = await asyncio.start_server(self.handle_client, self.cfg.host, self.cfg.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        LOGGER.info("Wyoming server listening on %s", addrs)
        async with server:
            await server.serve_forever()
