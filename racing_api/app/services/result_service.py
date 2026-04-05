from __future__ import annotations

from app.services.prediction_service import (
    load_history_records,
    map_history_to_prediction,
    normalize_venue_code,
    resolve_horse_name,
)

DEFAULT_STAKE = 100.0


def _to_int(value: object) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def map_history_to_result(record: dict) -> dict:
    prediction_view = map_history_to_prediction(record)
    race_id = str(prediction_view["race_id"])
    is_completed = bool(record.get("actual_winner") is not None or record.get("is_real_result"))
    status = "completed" if is_completed else "pending"

    predicted_winner_no = _to_int(record.get("predicted_winner"))
    predicted_winner = (resolve_horse_name(race_id, predicted_winner_no) if predicted_winner_no else None) or (
        prediction_view["top_picks"][0]["horse_name"] if prediction_view["top_picks"] else "Unknown Horse"
    )

    actual_winner_no = _to_int(record.get("actual_winner"))
    actual_winner = resolve_horse_name(race_id, actual_winner_no) if (is_completed and actual_winner_no) else None

    if status == "pending":
        hit_or_miss = "pending"
    elif record.get("winner_correct") is not None:
        hit_or_miss = "hit" if bool(record.get("winner_correct")) else "miss"
    else:
        hit_or_miss = "hit" if actual_winner == predicted_winner else "miss"

    payout = float(record.get("win_bet_return") or 0.0)
    stake = DEFAULT_STAKE
    roi = (payout - stake) / stake if stake > 0 else 0.0

    return {
        "race_id": prediction_view["race_id"],
        "race_name": prediction_view["race_name"],
        "event_date": prediction_view["event_date"],
        "venue": normalize_venue_code(record.get("venue")) or prediction_view["venue"],
        "status": status,
        "actual_winner": actual_winner,
        "predicted_winner": predicted_winner,
        "hit_or_miss": hit_or_miss,
        "stake": stake,
        "payout": payout,
        "roi": round(roi, 4),
    }


def get_results(status_filter: str | None = None, venue_filter: str | None = None) -> list[dict]:
    results = [map_history_to_result(record) for record in load_history_records()]

    if status_filter:
        results = [item for item in results if item["status"] == status_filter]

    if venue_filter:
        normalized_filter = normalize_venue_code(venue_filter)
        results = [item for item in results if item["venue"] == normalized_filter]

    results.sort(key=lambda item: (item["event_date"], item["race_id"]), reverse=True)
    return results


def get_result_by_race_id(race_id: str) -> dict | None:
    for record in load_history_records():
        if record.get("race_id") == race_id:
            return map_history_to_result(record)
    return None
