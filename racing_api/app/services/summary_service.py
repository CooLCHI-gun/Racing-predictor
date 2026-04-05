from __future__ import annotations

from datetime import date, datetime, timedelta

from app.services.result_service import get_results

DEFAULT_MODEL_VERSION = "racing-transformer-2.3.1"


def _event_date(record: dict) -> date:
    return datetime.strptime(record["event_date"], "%Y-%m-%d").date()


def _window_start(window: str, records: list[dict]) -> date | None:
    if not records:
        return None
    if window == "all":
        return min(_event_date(item) for item in records)

    days = int(window[:-1])
    end_date = max(_event_date(item) for item in records)
    return end_date - timedelta(days=days - 1)


def get_summary(window: str = "7d") -> dict:
    all_results = get_results()

    if not all_results:
        return {
            "window": window,
            "from_date": None,
            "to_date": None,
            "races_in_window": 0,
            "completed_races": 0,
            "hit_count": 0,
            "miss_count": 0,
            "hit_rate_percent": 0.0,
            "roi_stats": {
                "total_stake": 0.0,
                "total_payout": 0.0,
                "net_profit": 0.0,
                "roi_percent": 0.0,
            },
            "model_version": DEFAULT_MODEL_VERSION,
        }

    from_date = _window_start(window, all_results)
    to_date = max(_event_date(item) for item in all_results)

    filtered = all_results
    if from_date is not None:
        filtered = [item for item in all_results if _event_date(item) >= from_date]

    total_predictions = len(filtered)
    completed = [item for item in filtered if item["status"] == "completed"]
    hits = [item for item in completed if item["hit_or_miss"] == "hit"]
    misses = [item for item in completed if item["hit_or_miss"] == "miss"]

    total_stake = sum(item["stake"] for item in completed)
    total_payout = sum(item["payout"] for item in completed)
    net_profit = total_payout - total_stake
    roi_percent = (net_profit / total_stake * 100) if total_stake else 0.0
    hit_rate = (len(hits) / len(completed) * 100) if completed else 0.0

    return {
        "window": window,
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
        "races_in_window": total_predictions,
        "completed_races": len(completed),
        "hit_count": len(hits),
        "miss_count": len(misses),
        "hit_rate_percent": round(hit_rate, 2),
        "roi_stats": {
            "total_stake": round(total_stake, 2),
            "total_payout": round(total_payout, 2),
            "net_profit": round(net_profit, 2),
            "roi_percent": round(roi_percent, 2),
        },
        "model_version": DEFAULT_MODEL_VERSION,
    }
