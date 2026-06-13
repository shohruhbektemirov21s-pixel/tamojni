"""Fayl saqlash + sha256 (chain-of-custody, Tamoyil 3).

Disk to'lganda DiskFull (503) — mavjud case'lar saqlanadi (§10).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path

from app.core.errors import DiskFull

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(name: str) -> str:
    name = _SAFE.sub("_", name.strip()) or "file"
    return name[:120]


class FileStore:
    def __init__(self, base_dir: Path, min_free_mb: int) -> None:
        self.base = Path(base_dir)
        self.min_free_mb = min_free_mb
        self.base.mkdir(parents=True, exist_ok=True)

    def check_capacity(self) -> None:
        usage = shutil.disk_usage(self.base)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < self.min_free_mb:
            raise DiskFull(
                "Diskda joy yetarli emas",
                detail={"free_mb": round(free_mb), "min_mb": self.min_free_mb},
            )

    def save(self, case_id: str, kind: str, filename: str, data: bytes) -> dict:
        """-> {"kind", "path", "sha256"} (Attachment uchun)."""
        self.check_capacity()
        case_dir = self.base / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / f"{kind}_{uuid.uuid4().hex[:8]}_{_safe_name(filename)}"
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        return {"kind": kind, "path": str(path), "sha256": sha}
