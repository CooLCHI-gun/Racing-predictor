from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import jsonify


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def success_response(
    data: Any,
    message: str = "Request successful",
    status_code: int = 200,
    meta: dict[str, Any] | None = None,
):
    payload = {
        "success": True,
        "message": message,
        "timestamp": utc_timestamp(),
        "data": data,
    }
    if meta:
        payload["meta"] = meta
    return jsonify(payload), status_code


def error_response(message: str, code: str, status_code: int):
    payload = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
        "timestamp": utc_timestamp(),
    }
    return jsonify(payload), status_code
