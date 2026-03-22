from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WyomingMessage:
    msg_type: str
    data: dict[str, Any]
    extra_data: dict[str, Any]
    payload: bytes


async def read_message(reader) -> WyomingMessage:
    """读取 Wyoming 协议消息。

    Args:
        reader: asyncio.StreamReader

    Returns:
        WyomingMessage 对象

    Raises:
        EOFError: 客户端关闭连接
    """
    header_line = await reader.readline()
    if not header_line:
        raise EOFError("Client closed connection")
    header = json.loads(header_line.decode("utf-8"))

    data_length = int(header.get("data_length", 0) or 0)
    payload_length = int(header.get("payload_length", 0) or 0)

    extra_data: dict[str, Any] = {}
    if data_length > 0:
        raw_data = await reader.readexactly(data_length)
        extra_data = json.loads(raw_data.decode("utf-8"))

    payload = b""
    if payload_length > 0:
        payload = await reader.readexactly(payload_length)

    return WyomingMessage(
        msg_type=header["type"],
        data=header.get("data") or {},
        extra_data=extra_data,
        payload=payload,
    )


def encode_header(message_type: str, data: dict[str, Any], payload: bytes) -> bytes:
    """编码 Wyoming 协议消息头。

    Args:
        message_type: 消息类型
        data: 消息数据
        payload: 二进制负载

    Returns:
        编码后的消息头（包含换行符）
    """
    header: dict[str, Any] = {"type": message_type, "data": data}
    if payload:
        header["payload_length"] = len(payload)
    return (json.dumps(header, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")


async def write_message(
    writer,
    message_type: str,
    data: dict[str, Any],
    payload: bytes = b"",
) -> None:
    """写入 Wyoming 协议消息。

    Args:
        writer: asyncio.StreamWriter
        message_type: 消息类型
        data: 消息数据
        payload: 二进制负载
    """
    writer.write(encode_header(message_type, data, payload))
    if payload:
        writer.write(payload)
    await writer.drain()
