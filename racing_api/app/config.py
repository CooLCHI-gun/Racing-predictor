from __future__ import annotations

import os


class BaseConfig:
    APP_NAME = "Racing Predictor API"
    API_VERSION = "v1"
    JSON_SORT_KEYS = False
    PROPAGATE_EXCEPTIONS = True
    HOST = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_RUN_PORT", "5000"))


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = False


class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False


class TestingConfig(BaseConfig):
    DEBUG = False
    TESTING = True


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
