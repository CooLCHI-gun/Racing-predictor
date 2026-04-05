from __future__ import annotations

from flask import Blueprint, request

from app.schemas.result_schema import ResultOut, ResultQuery
from app.services.result_service import get_result_by_race_id, get_results
from app.utils.errors import NotFoundError
from app.utils.responses import success_response

results_bp = Blueprint("results", __name__)


@results_bp.get("/results")
def list_results():
    query = ResultQuery.model_validate(request.args.to_dict())
    records = get_results(status_filter=query.status, venue_filter=query.venue)
    data = [ResultOut.model_validate(item).model_dump(mode="json") for item in records]
    return success_response(
        data=data,
        meta={
            "count": len(data),
            "filters": {
                "status": query.status,
                "venue": query.venue,
            },
        },
    )


@results_bp.get("/results/<string:race_id>")
def result_detail(race_id: str):
    record = get_result_by_race_id(race_id)
    if record is None:
        raise NotFoundError(message=f"Result not found for race_id '{race_id}'")

    data = ResultOut.model_validate(record).model_dump(mode="json")
    return success_response(data=data)
