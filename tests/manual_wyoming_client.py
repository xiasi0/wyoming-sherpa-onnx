import json
import socket

import numpy as np
import soundfile as sf


def send(sock, typ, data=None, payload=b""):
    data = data or {}
    header = {"type": typ, "data": data}
    if payload:
        header["payload_length"] = len(payload)
    sock.sendall((json.dumps(header, ensure_ascii=True) + "\n").encode("utf-8"))
    if payload:
        sock.sendall(payload)


def recv_line(sock):
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise EOFError("socket closed")
        if b == b"\n":
            return bytes(buf)
        buf.extend(b)


def recv_msg(sock):
    header = json.loads(recv_line(sock).decode("utf-8"))
    payload = b""
    if header.get("payload_length"):
        n = int(header["payload_length"])
        while len(payload) < n:
            payload += sock.recv(n - len(payload))
    return header, payload


if __name__ == "__main__":
    wav, sr = sf.read("sample.wav", dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16).tobytes()

    with socket.create_connection(("127.0.0.1", 10300), timeout=10) as s:
        send(s, "describe")
        print("describe =>", recv_msg(s)[0])

        send(s, "transcribe", {"language": "zh"})
        send(s, "audio-start", {"rate": sr, "width": 2, "channels": 1})

        chunk = 3200
        for i in range(0, len(pcm), chunk):
            send(s, "audio-chunk", {"rate": sr, "width": 2, "channels": 1}, pcm[i : i + chunk])

        send(s, "audio-stop")
        while True:
            header, _ = recv_msg(s)
            print("recv =>", header)
            if header["type"] == "transcript":
                break
