"""mDNS advertisement so the glasses find the laptop without a hardcoded IP.

Registers the host as ``lstk-eye.local`` and the HTTP endpoint as an
``_lstk-eye._tcp.local.`` service. The firmware resolves the ``.local`` name
first and falls back to a configured IP. Failure to advertise is logged and
otherwise ignored - discovery is a convenience, not a dependency.
"""

import logging
import socket

log = logging.getLogger(__name__)


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent; picks the default route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class ZeroconfAdvertiser:
    def __init__(self, port: int):
        self._port = port
        self._zc = None
        self._info = None

    def start(self) -> None:
        try:
            from zeroconf import ServiceInfo, Zeroconf

            ip = _local_ip()
            self._info = ServiceInfo(
                "_lstk-eye._tcp.local.",
                "lstk-eye._lstk-eye._tcp.local.",
                addresses=[socket.inet_aton(ip)],
                port=self._port,
                server="lstk-eye.local.",
            )
            self._zc = Zeroconf()
            self._zc.register_service(self._info)
            log.info("mDNS: advertising lstk-eye.local (%s:%d)", ip, self._port)
        except Exception:
            log.warning("mDNS advertisement failed; use the IP directly", exc_info=True)
            self._zc = None

    def stop(self) -> None:
        if self._zc is not None:
            try:
                self._zc.unregister_service(self._info)
                self._zc.close()
            except Exception:
                pass
            self._zc = None
