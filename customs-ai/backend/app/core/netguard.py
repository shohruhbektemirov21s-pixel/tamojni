"""Offline by design (Tamoyil 4) — tashqi tarmoq guard.

socket.connect'ni o'rab oladi: loopback (127.0.0.1/::1) ruxsat, qolgan hamma
narsa NetworkBlocked. CI'da no-network testida ishlatiladi; production'da ham
defense-in-depth sifatida yoqilishi mumkin.
"""
from __future__ import annotations

import socket

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_installed = False


class NetworkBlocked(Exception):
    pass


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _is_loopback(address) -> bool:  # noqa: ANN001
    try:
        host = address[0]
    except (TypeError, IndexError):
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    return isinstance(host, str) and host.startswith("127.")


def install_network_guard(allow_loopback: bool = True) -> None:
    global _installed
    if _installed:
        return
    _installed = True

    def guarded_connect(self, address):  # noqa: ANN001
        if allow_loopback and _is_loopback(address):
            return _real_connect(self, address)
        raise NetworkBlocked(f"Tashqi tarmoq taqiqlangan (offline by design): {address!r}")

    def guarded_connect_ex(self, address):  # noqa: ANN001
        if allow_loopback and _is_loopback(address):
            return _real_connect_ex(self, address)
        raise NetworkBlocked(f"Tashqi tarmoq taqiqlangan (offline by design): {address!r}")

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


def remove_network_guard() -> None:
    global _installed
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex
    _installed = False
