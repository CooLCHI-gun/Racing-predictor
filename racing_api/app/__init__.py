from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask

from app.api import register_blueprints
from app.config import config_by_name
from app.extensions import init_extensions
from app.utils.errors import register_error_handlers


def create_app(config_name: str | None = None) -> Flask:
    """Application factory for creating configured Flask instances."""
    load_dotenv()

    app = Flask(__name__)

    selected_config = config_name or os.getenv("FLASK_ENV", "development")
    config_class: type[Any] = config_by_name.get(selected_config, config_by_name["development"])
    app.config.from_object(config_class)

    init_extensions(app)
    register_blueprints(app)
    register_error_handlers(app)

    return app
