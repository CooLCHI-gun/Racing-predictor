from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class SummaryWindowQuery(BaseModel):
    window: str = "7d"

    @field_validator("window")
    @classmethod
    def validate_window(cls, value: str) -> str:
        raw = value.strip().lower()
        if raw == "all":
            return raw
        if len(raw) < 2 or not raw.endswith("d"):
            raise ValueError("window must be in format like 7d, 30d, or all")
        days_part = raw[:-1]
        if not days_part.isdigit() or int(days_part) <= 0:
            raise ValueError("window days must be a positive integer")
        return raw


class RoiStats(BaseModel):
    total_stake: float = Field(ge=0.0)
    total_payout: float = Field(ge=0.0)
    net_profit: float
    roi_percent: float


class SummaryOut(BaseModel):
    window: str
    from_date: date | None = None
    to_date: date | None = None
    races_in_window: int = Field(ge=0)
    completed_races: int = Field(ge=0)
    hit_count: int = Field(ge=0)
    miss_count: int = Field(ge=0)
    hit_rate_percent: float = Field(ge=0.0)
    roi_stats: RoiStats
    model_version: str
