from __future__ import annotations

from flask import Flask
from flask_cors import CORS

cors = CORS()


def init_extensions(app: Flask) -> None:
    cors.init_app(
        app,
        resources={r"/api/*": {"origins": app.config.get("CORS_ALLOWED_ORIGINS", "*")}},
        supports_credentials=False,
    )
