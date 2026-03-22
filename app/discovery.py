from __future__ import annotations

import socket
from dataclasses import dataclass, field

from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf


@dataclass(slots=True)
class WyomingDiscovery:
    service_name: str
    host: str
    port: int
    model_name: str
    _zc: AsyncZeroconf | None = field(init=False, default=None)
    _info: ServiceInfo | None = field(init=False, default=None)

    async def start(self) -> None:
        addr = socket.inet_aton(self.host)
        self._info = ServiceInfo(
            type_="_wyoming._tcp.local.",
            name=f"{self.service_name}._wyoming._tcp.local.",
            addresses=[addr],
            port=self.port,
            properties={
                "name": self.service_name,
                "asr": "true",
                "model": self.model_name,
            },
            server=f"{self.service_name}.local.",
        )
        self._zc = AsyncZeroconf(ip_version=IPVersion.All)
        await self._zc.async_register_service(self._info)

    async def stop(self) -> None:
        if self._zc and self._info:
            await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()
