"""Offline by design (Tamoyil 4): tashqi socket = fail; e2e baribir ishlaydi."""
from __future__ import annotations

import socket

import pytest

from app.core.netguard import NetworkBlocked, install_network_guard, remove_network_guard
from tests.helpers import PNG, client_for, wait_done


def test_external_socket_blocked_but_pipeline_runs(tmp_path):
    install_network_guard(allow_loopback=True)
    try:
        # tashqi ulanish bloklanadi
        with pytest.raises(NetworkBlocked):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect(("8.8.8.8", 53))
            finally:
                s.close()

        # offline pipeline (mock, loopback) baribir to'liq ishlaydi
        with client_for(tmp_path) as client:
            case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
            result = wait_done(client, case_id)
            assert result["status"] == "DONE"
            assert result["risk"] is not None
    finally:
        remove_network_guard()
