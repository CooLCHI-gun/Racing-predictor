from __future__ import annotations

from http import HTTPStatus

from flask import Flask, request
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from app.utils.responses import error_response


class APIError(Exception):
    def __init__(self, message: str, code: str = "api_error", status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(APIError):
    def __init__(self, message: str = "Resource not found", code: str = "not_found"):
        super().__init__(message=message, code=code, status_code=404)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        return error_response(error.message, error.code, error.status_code)

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        message = error.errors()[0]["msg"] if error.errors() else "Invalid request parameters"
        return error_response(message, "validation_error", 400)

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        status = error.code or 500
        description = error.description if isinstance(error.description, str) else "Request failed"
        if status == HTTPStatus.NOT_FOUND and request.path.startswith("/api"):
            return error_response("Endpoint not found", "not_found", 404)
        return error_response(description, f"http_{status}", status)

    @app.errorhandler(Exception)
    def handle_unexpected_error(_error: Exception):
        return error_response("Internal server error", "internal_server_error", 500)
