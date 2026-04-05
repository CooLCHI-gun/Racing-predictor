from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import create_app
from app.services import prediction_service, result_service

SAMPLE_HISTORY_RECORDS = [
    {
        "race_id": "2026-04-05_ST_1",
        "race_date": "2026-04-05",
        "race_number": 1,
        "venue": "Sha Tin",
        "race_class": "Class 4",
        "predicted_top3": [5, 2, 8],
        "predicted_winner": 5,
        "predicted_top3_odds": [4.2, 7.1, 12.0],
        "model_confidence": 0.72,
        "created_at": "2026-04-05T06:18:00",
        "actual_winner": 5,
        "winner_correct": True,
        "win_bet_return": 420.0,
        "is_real_result": True,
    },
    {
        "race_id": "2026-04-05_HV_2",
        "race_date": "2026-04-05",
        "race_number": 2,
        "venue": "Happy Valley",
        "race_class": "Class 3",
        "predicted_top3": [1, 6, 10],
        "predicted_winner": 1,
        "predicted_top3_odds": [5.5, 8.0, 13.4],
        "model_confidence": 0.63,
        "created_at": "2026-04-05T06:25:00",
        "actual_winner": 6,
        "winner_correct": False,
        "win_bet_return": 0.0,
        "is_real_result": True,
    },
]


@pytest.fixture(autouse=True)
def use_sample_history(monkeypatch):
    horse_name_map = {
        "2026-04-05_ST_1": {5: "Silver Comet", 2: "Eastern Crown", 8: "Swift Horizon"},
        "2026-04-05_HV_2": {1: "North Harbour", 6: "Thunder Bloom", 10: "Blue Tempest"},
    }

    def _resolve_horse_name(race_id: str, horse_no: object) -> str:
        try:
            num = int(str(horse_no).strip())
        except Exception:
            return f"Horse #{horse_no}" if horse_no is not None else "Unknown Horse"
        return horse_name_map.get(race_id, {}).get(num, f"Horse #{num}")

    monkeypatch.setattr(prediction_service, "ENABLE_REMOTE_NAME_LOOKUP", False)
    prediction_service.load_history_records.cache_clear()
    prediction_service.get_horse_name_map_for_race.cache_clear()
    monkeypatch.setattr(prediction_service, "load_history_records", lambda: SAMPLE_HISTORY_RECORDS)
    monkeypatch.setattr(prediction_service, "resolve_horse_name", _resolve_horse_name)
    monkeypatch.setattr(result_service, "load_history_records", lambda: SAMPLE_HISTORY_RECORDS)
    monkeypatch.setattr(result_service, "resolve_horse_name", _resolve_horse_name)


@pytest.fixture()
def client():
    app = create_app("testing")
    with app.test_client() as test_client:
        yield test_client
