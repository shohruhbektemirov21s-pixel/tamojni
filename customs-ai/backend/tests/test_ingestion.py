"""Scanner ingestion — WatchedFolderSource (partial/dedup/reject + real observer).

Ikki qatlam:
  1) Sof yadro: `_ingest_file` to'g'ridan chaqiriladi (watchdog'siz, deterministik).
  2) Real observer: scanner_enabled=True, papkaga fayl tushiriladi -> avto case.

Tezlik uchun stability tekshiruvi qisqartirilgan (stable_checks=1, poll=0.01).
"""
from __future__ import annotations

import time

from app.ingestion import WatchedFolderSource
from tests.helpers import client_for

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-scan-bytes"
FAST = dict(scanner_stable_checks=1, scanner_poll_interval_s=0.01, scanner_stable_timeout_s=3.0)


def _make_source(client, folder):
    st = client.app.state
    return WatchedFolderSource(
        folder=folder, intake=st.case_intake, event_bus=st.event_bus,
        settings=st.settings, repo=st.repo,
    )


def _drain_until(wsconn, stop_type, limit=40):
    types = []
    for _ in range(limit):
        ev = wsconn.receive_json()
        types.append(ev["type"])
        if ev["type"] == stop_type:
            break
    return types


def test_ingest_file_creates_case_and_emits_scan_ingested(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    with client_for(tmp_path, **FAST) as client:
        with client.websocket_connect("/ws") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"

            f = inbox / "scan1.png"
            f.write_bytes(PNG_BYTES)
            src = _make_source(client, inbox)
            case_id = src._ingest_file(f).result(timeout=5)  # Future -> case_id

            types = _drain_until(wsconn, "case_done")

        assert "scan_ingested" in types
        assert "case_created" in types
        assert "tier1_done" in types

        # case haqiqatan yozildi va source teglandi
        actions = client.get(f"/cases/{case_id}/audit").json()["entries"]
        created = next(e for e in actions if e["action"] == "CASE_CREATED")
        assert created["payload"]["source"].startswith("scanner:")


def test_partial_file_skipped_until_stable(tmp_path):
    """Hajm hali o'sayotgan fayl -> partial deb skip (None)."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    # timeout juda qisqa + fayl "o'smoqda" simulyatsiyasi qilib bo'lmaydi sinxron;
    # buning o'rniga bo'sh-emas, lekin stable bo'lmaydigan timeout=0 holatini sinaymiz.
    with client_for(tmp_path, scanner_stable_checks=99, scanner_poll_interval_s=0.01,
                    scanner_stable_timeout_s=0.05) as client:
        f = inbox / "growing.png"
        f.write_bytes(PNG_BYTES)
        src = _make_source(client, inbox)
        assert src._ingest_file(f) is None  # stable bo'lolmadi -> skip


def test_duplicate_sha_skipped(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    with client_for(tmp_path, **FAST) as client:
        src = _make_source(client, inbox)
        a = inbox / "a.png"; a.write_bytes(PNG_BYTES)
        b = inbox / "b.png"; b.write_bytes(PNG_BYTES)  # bir xil kontent
        assert src._ingest_file(a) is not None       # birinchi -> case
        assert src._ingest_file(b) is None            # dublikat -> skip


def test_empty_and_noncandidate_skipped(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    with client_for(tmp_path, **FAST) as client:
        src = _make_source(client, inbox)
        empty = inbox / "empty.png"; empty.write_bytes(b"")
        tmpf = inbox / "scan.png.tmp"; tmpf.write_bytes(PNG_BYTES)
        txt = inbox / "note.txt"; txt.write_bytes(PNG_BYTES)
        assert src._ingest_file(empty) is None        # bo'sh
        assert src._ingest_file(tmpf) is None          # .tmp
        assert src._ingest_file(txt) is None           # kengaytma ro'yxatda yo'q


def _wait_cases(client, n=1, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        items = client.get("/cases").json()["items"]
        if len(items) >= n:
            return items
        time.sleep(0.05)
    raise AssertionError("scanner belgilangan vaqtda case yaratmadi")


def test_watchdog_observer_end_to_end(tmp_path):
    """Real watchdog: papkaga fayl tushadi -> avtomatik case ochiladi va DONE bo'ladi."""
    inbox = tmp_path / "inbox"
    with client_for(tmp_path, scanner_enabled=True, scanner_folder=inbox, **FAST) as client:
        # lifespan start() papkani yaratib, observer'ni ishga tushirdi
        (inbox / "auto.png").write_bytes(PNG_BYTES)

        items = _wait_cases(client, n=1, timeout=10.0)
        case_id = items[0]["case_id"]

        deadline = time.time() + 10
        while time.time() < deadline:
            res = client.get(f"/cases/{case_id}").json()
            if res["status"] in ("DONE", "FAILED"):
                break
            time.sleep(0.05)
        assert res["status"] == "DONE"

        actions = [e["action"] for e in client.get(f"/cases/{case_id}/audit").json()["entries"]]
        assert "CASE_CREATED" in actions
        assert "RISK_COMPUTED" in actions
