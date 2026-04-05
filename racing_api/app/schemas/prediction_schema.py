from __future__ import annotations

from datetime import date as dt_date
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TopPick(BaseModel):
    horse_name: str
    predicted_rank: int = Field(ge=1, le=20)
    win_probability: float = Field(ge=0.0, le=1.0)
    odds_decimal: float | None = Field(default=None, ge=1.0)


class PredictionOut(BaseModel):
    race_id: str
    race_name: str
    event_date: dt_date
    venue: str
    top_picks: list[TopPick]
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_version: str
    prediction_timestamp: datetime
    result_status: Literal["upcoming", "completed"]


class PredictionQuery(BaseModel):
    date: dt_date | None = None
    venue: str | None = None

    @field_validator("venue")
    @classmethod
    def normalize_venue(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("venue must not be empty")
        return normalized
