from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

from app.utils.text import normalize_display_name, normalize_name

DEFAULT_MODEL_VERSION = "racing-transformer-2.3.1"
VENUE_CODE_MAP = {
    "SHA TIN": "ST",
    "ST": "ST",
    "HAPPY VALLEY": "HV",
    "HV": "HV",
}
ENABLE_REMOTE_NAME_LOOKUP = os.getenv("RACING_API_ENABLE_REMOTE_NAME_LOOKUP", "1") != "0"

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _history_file_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "predictions" / "history.json"


def _historical_results_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "historical_results.json"


@lru_cache(maxsize=1)
def load_history_records() -> list[dict]:
    file_path = _history_file_path()
    if not file_path.exists():
        return []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(loaded, dict):
        records = list(loaded.values())
    elif isinstance(loaded, list):
        records = loaded
    else:
        return []

    return [record for record in records if isinstance(record, dict) and record.get("race_id")]


def normalize_venue_code(venue: str | None) -> str | None:
    if venue is None:
        return None
    return VENUE_CODE_MAP.get(venue.strip().upper(), venue.strip().upper())


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 1.0:
        return None
    return parsed


def _as_horse_name(value: object) -> str:
    if isinstance(value, int):
        return f"Horse #{value}"
    if isinstance(value, str) and value.strip():
        return normalize_display_name(value.strip())
    return "Unknown Horse"


def _to_int(value: object) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_race_id(race_id: str) -> tuple[str, str, int] | None:
    parts = (race_id or "").split("_")
    if len(parts) != 3:
        return None

    race_date = parts[0]
    venue_code = normalize_venue_code(parts[1])
    race_no = _to_int(parts[2])
    if venue_code is None or race_no is None:
        return None
    return race_date, venue_code, race_no


@lru_cache(maxsize=1)
def _load_historical_results_file() -> dict:
    path = _historical_results_path()
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        out: dict[str, dict] = {}
        for item in raw:
            if isinstance(item, dict) and item.get("race_id"):
                out[str(item["race_id"])] = item
        return out
    return {}


@lru_cache(maxsize=64)
def _meeting_horse_names_from_history_file(race_date: str, venue_code: str) -> dict[int, dict[int, str]]:
    raw = _load_historical_results_file()
    result: dict[int, dict[int, str]] = {}

    for item in raw.values():
        if not isinstance(item, dict):
            continue
        race_id = str(item.get("race_id") or "")
        parsed = _parse_race_id(race_id)
        if not parsed:
            continue
        item_date, item_venue, item_race_no = parsed
        if item_date != race_date or item_venue != venue_code:
            continue

        horse_rows = item.get("all_horses") or []
        for horse_row in horse_rows:
            if not isinstance(horse_row, dict):
                continue
            horse_no = _to_int(horse_row.get("horse_num"))
            horse_name = normalize_display_name(str(horse_row.get("horse_name") or "").strip())
            if horse_no and horse_name:
                result.setdefault(item_race_no, {})[horse_no] = horse_name

    return result


@lru_cache(maxsize=64)
def _meeting_horse_names_from_results_fetcher(race_date: str, venue_code: str) -> dict[int, dict[int, str]]:
    if not ENABLE_REMOTE_NAME_LOOKUP:
        return {}

    try:
        from scraper.results_fetcher import fetch_day_results

        day_results = fetch_day_results(race_date=race_date, venue_code=venue_code)
    except Exception as exc:
        logger.debug("results_fetcher lookup failed for %s %s: %s", race_date, venue_code, exc)
        return {}

    result: dict[int, dict[int, str]] = {}
    for race_result in day_results:
        race_no = _to_int(getattr(race_result, "race_number", None))
        if race_no is None:
            continue
        for horse_row in getattr(race_result, "all_horses", []) or []:
            horse_no = _to_int(horse_row.get("horse_num")) if isinstance(horse_row, dict) else None
            horse_name = normalize_display_name(str(horse_row.get("horse_name") or "").strip()) if isinstance(horse_row, dict) else ""
            if horse_no and horse_name:
                result.setdefault(race_no, {})[horse_no] = horse_name

    return result


@lru_cache(maxsize=64)
def _meeting_horse_names_from_racecard(race_date: str, venue_code: str) -> dict[int, dict[int, str]]:
    if not ENABLE_REMOTE_NAME_LOOKUP:
        return {}

    try:
        from scraper.racecard import fetch_racecard

        races = fetch_racecard(race_date=race_date, venue_code=venue_code, auto_next_if_empty=False)
    except Exception as exc:
        logger.debug("racecard lookup failed for %s %s: %s", race_date, venue_code, exc)
        return {}

    result: dict[int, dict[int, str]] = {}
    for race in races:
        race_no = _to_int(getattr(race, "race_number", None))
        if race_no is None:
            continue
        for horse in getattr(race, "horses", []) or []:
            horse_no = _to_int(getattr(horse, "horse_number", None))
            horse_name = normalize_display_name(str(getattr(horse, "horse_name", "") or "").strip())
            if horse_no and horse_name:
                result.setdefault(race_no, {})[horse_no] = horse_name

    return result


@lru_cache(maxsize=256)
def get_horse_name_map_for_race(race_id: str) -> dict[int, str]:
    parsed = _parse_race_id(race_id)
    if not parsed:
        return {}

    race_date, venue_code, race_no = parsed
    sources = [
        _meeting_horse_names_from_history_file,
        _meeting_horse_names_from_results_fetcher,
        _meeting_horse_names_from_racecard,
    ]

    for source in sources:
        meeting_map = source(race_date, venue_code)
        if race_no in meeting_map and meeting_map[race_no]:
            return meeting_map[race_no]

    return {}


def resolve_horse_name(race_id: str, horse_number: object) -> str:
    horse_no = _to_int(horse_number)
    fallback = _as_horse_name(horse_number)
    if horse_no is None:
        return fallback

    mapped = get_horse_name_map_for_race(race_id).get(horse_no)
    if mapped:
        return normalize_display_name(mapped)

    # Assumption: historical storage may not include horse name snapshots for older races.
    logger.debug("Horse name fallback used for race_id=%s horse_no=%s", race_id, horse_no)
    return fallback


def _build_top_picks(record: dict) -> list[dict]:
    race_id = str(record.get("race_id") or "")
    top3 = record.get("predicted_top3") or []
    top3_odds = record.get("predicted_top3_odds") or []
    confidence = float(record.get("model_confidence") or 0.5)
    confidence = max(0.0, min(confidence, 1.0))
    weights = [0.5, 0.3, 0.2]

    picks: list[dict] = []
    for idx, horse_no in enumerate(top3[:3], start=1):
        weight = weights[idx - 1] if idx - 1 < len(weights) else 0.1
        odds = top3_odds[idx - 1] if idx - 1 < len(top3_odds) else None
        picks.append(
            {
                "horse_name": resolve_horse_name(race_id, horse_no),
                "predicted_rank": idx,
                "win_probability": round(confidence * weight, 4),
                "odds_decimal": _float_or_none(odds),
            }
        )

    return picks


def map_history_to_prediction(record: dict) -> dict:
    race_number = record.get("race_number")
    race_class = record.get("race_class") or "Unknown Class"
    race_name = f"Race {race_number} ({race_class})" if race_number else f"Race ({race_class})"
    race_name = normalize_display_name(race_name)

    prediction_timestamp = (
        record.get("created_at")
        or record.get("result_recorded_at")
        or f"{record.get('race_date', '1970-01-01')}T00:00:00"
    )
    if isinstance(prediction_timestamp, str) and "Z" not in prediction_timestamp and "+" not in prediction_timestamp:
        prediction_timestamp = f"{prediction_timestamp}+00:00"

    status = "completed" if record.get("actual_winner") is not None or record.get("is_real_result") else "upcoming"

    return {
        "race_id": record.get("race_id"),
        "race_name": race_name,
        "event_date": record.get("race_date"),
        "venue": normalize_venue_code(record.get("venue")) or "NA",
        "top_picks": _build_top_picks(record),
        "confidence_score": max(0.0, min(float(record.get("model_confidence") or 0.0), 1.0)),
        "model_version": str(record.get("model_version") or DEFAULT_MODEL_VERSION),
        "prediction_timestamp": prediction_timestamp,
        "result_status": status,
    }


def get_predictions(date_filter: date | None = None, venue_filter: str | None = None) -> list[dict]:
    predictions = [map_history_to_prediction(record) for record in load_history_records()]

    if date_filter:
        predictions = [item for item in predictions if item["event_date"] == date_filter.isoformat()]

    if venue_filter:
        normalized_filter = normalize_venue_code(venue_filter)
        predictions = [item for item in predictions if item["venue"] == normalized_filter]

    predictions.sort(key=lambda item: (item["event_date"], item["race_id"]), reverse=True)
    return predictions


def get_prediction_by_race_id(race_id: str) -> dict | None:
    for record in load_history_records():
        if record.get("race_id") == race_id:
            return map_history_to_prediction(record)
    return None
