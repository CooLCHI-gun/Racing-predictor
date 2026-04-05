from __future__ import annotations

from flask import Blueprint, request

from app.schemas.prediction_schema import PredictionOut, PredictionQuery
from app.services.prediction_service import get_prediction_by_race_id, get_predictions
from app.utils.errors import NotFoundError
from app.utils.responses import success_response

predictions_bp = Blueprint("predictions", __name__)


@predictions_bp.get("/predictions")
def list_predictions():
    query = PredictionQuery.model_validate(request.args.to_dict())
    records = get_predictions(date_filter=query.date, venue_filter=query.venue)
    data = [PredictionOut.model_validate(item).model_dump(mode="json") for item in records]
    return success_response(
        data=data,
        meta={
            "count": len(data),
            "filters": {
                "date": query.date.isoformat() if query.date else None,
                "venue": query.venue,
            },
        },
    )


@predictions_bp.get("/predictions/<string:race_id>")
def prediction_detail(race_id: str):
    record = get_prediction_by_race_id(race_id)
    if record is None:
        raise NotFoundError(message=f"Prediction not found for race_id '{race_id}'")

    data = PredictionOut.model_validate(record).model_dump(mode="json")
    return success_response(data=data)
