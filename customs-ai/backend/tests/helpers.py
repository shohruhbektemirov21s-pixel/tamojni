"""Test yordamchilari — pytest-asyncio'siz (TestClient + polling)."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

PNG = ("scan.png", b"\x89PNG\r\n\x1a\nfake", "image/png")
WAV = ("audio.wav", b"RIFFfake_audio_data", "audio/wav")


def client_for(tmp_path: Path, **overrides) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        use_mocks=True,
        manage_ollama=False,
        **overrides,
    )
    return TestClient(create_app(settings))


def wait_done(client: TestClient, case_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/cases/{case_id}")
        if r.status_code == 200 and r.json()["status"] in ("DONE", "FAILED"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"case {case_id} belgilangan vaqtda yakunlanmadi")
