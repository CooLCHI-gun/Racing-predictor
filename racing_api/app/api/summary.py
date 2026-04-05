from __future__ import annotations

from flask import Blueprint, request

from app.schemas.summary_schema import SummaryOut, SummaryWindowQuery
from app.services.summary_service import get_summary
from app.utils.responses import success_response

summary_bp = Blueprint("summary", __name__)


@summary_bp.get("/summary")
def summary():
    query = SummaryWindowQuery.model_validate(request.args.to_dict())
    data = SummaryOut.model_validate(get_summary(window=query.window)).model_dump(mode="json")
    return success_response(data=data)
