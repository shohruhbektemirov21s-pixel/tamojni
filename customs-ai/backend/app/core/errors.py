"""Yagona xatolik modeli.

Barcha API xatoliklari §7 dagi kontraktga amal qiladi:
    {"error": {"code": ..., "message": ..., "detail": ...}}
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Barcha domen xatoliklari shundan meros oladi."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        detail: Any = None,
        *,
        code: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


class ValidationFailed(AppError):
    code = "validation_failed"
    http_status = 400


class NotFound(AppError):
    code = "not_found"
    http_status = 404


class Conflict(AppError):
    code = "conflict"
    http_status = 409


class DiskFull(AppError):
    code = "disk_full"
    http_status = 503


class ServiceUnavailable(AppError):
    code = "service_unavailable"
    http_status = 503


class AuditImmutable(AppError):
    """Audit jurnali append-only — UPDATE/DELETE taqiqlanadi (Tamoyil 3)."""

    code = "audit_immutable"
    http_status = 500
