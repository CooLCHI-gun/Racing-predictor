from __future__ import annotations

from flask import Blueprint, current_app

from app.utils.responses import success_response

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health_check():
    data = {
        "status": "ok",
        "service": current_app.config.get("APP_NAME"),
        "environment": "testing" if current_app.config.get("TESTING") else "runtime",
        "version": current_app.config.get("API_VERSION"),
    }
    return success_response(data=data, message="Health check passed")
