from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ResultOut(BaseModel):
    race_id: str
    race_name: str
    event_date: date
    venue: str
    status: Literal["completed", "pending"]
    actual_winner: str | None = None
    predicted_winner: str
    hit_or_miss: Literal["hit", "miss", "pending"]
    stake: float = Field(ge=0.0)
    payout: float = Field(ge=0.0)
    roi: float


class ResultQuery(BaseModel):
    status: Literal["completed", "pending"] | None = None
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
