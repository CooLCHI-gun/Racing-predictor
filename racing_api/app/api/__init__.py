from __future__ import annotations

from flask import Flask

from app.api.health import health_bp
from app.api.predictions import predictions_bp
from app.api.results import results_bp
from app.api.summary import summary_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(predictions_bp, url_prefix="/api")
    app.register_blueprint(results_bp, url_prefix="/api")
    app.register_blueprint(summary_bp, url_prefix="/api")
