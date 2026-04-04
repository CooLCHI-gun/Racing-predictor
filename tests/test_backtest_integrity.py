import json
import os
import tempfile
from datetime import date

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from model.backtest import Backtester, RacePredictionRecord
from model.predictor import PredictionResult
from features.builder import build_features
from scraper.racecard import Race, HorseEntry


class DummyPredictionResult:
    def __init__(self):
        self.race_number = 1
        self.race_date = date.today().strftime("%Y-%m-%d")
        self.venue = "Sha Tin"
        self.venue_code = "ST"
        self.distance = 1200
        self.race_class = "Class 4"
        self.going = "GOOD"
        self.start_time = "14:00"
        self.top5 = []
        self.confidence = 0.5


def test_backtester_record_prediction_and_result_cycle(tmp_path):
    history_path = tmp_path / "history.json"

    bt = Backtester(history_file=str(history_path))
    assert isinstance(bt, Backtester)

    pred = DummyPredictionResult()
    pred.top5 = [type("H", (), {"horse_number": 3, "win_odds": 5.0})()]  # one horse only

    race_id = bt.record_prediction(race_id="2026-04-03_ST_1", prediction_result=pred, race_date=pred.race_date)
    assert race_id == "2026-04-03_ST_1"

    record = bt._records[race_id]
    assert record.race_date == pred.race_date
    assert record.predicted_winner == 3

    with pytest.raises(ValueError):
        bt.record_result(race_id, actual_result=[], place_dividends={}, win_dividend=10.0)

    bt.record_result(race_id, actual_result=[3, 5, 7], win_dividend=50.0, place_dividends={3: 20.0})
    rec = bt._records[race_id]
    assert rec.winner_correct is True
    assert rec.top3_hit == 1

    summary = bt.calculate_roi(strategy="top1_win", period_days=30)
    assert summary.total_races >= 1


def test_backtester_auto_labels_feature_rows_on_result(tmp_path):
    history_path = tmp_path / "history.json"
    bt = Backtester(history_file=str(history_path))

    pred = DummyPredictionResult()
    pred.top5 = [
        type("H", (), {"horse_number": 3, "win_odds": 5.0})(),
        type("H", (), {"horse_number": 5, "win_odds": 6.0})(),
        type("H", (), {"horse_number": 7, "win_odds": 7.0})(),
    ]
    pred.feature_rows = [
        {"horse_number": 3, "elo_rating": 1530.0, "win_odds": 5.0},
        {"horse_number": 5, "elo_rating": 1510.0, "win_odds": 6.0},
        {"horse_number": 7, "elo_rating": 1490.0, "win_odds": 7.0},
    ]

    race_id = bt.record_prediction(race_id="2026-04-03_ST_9", prediction_result=pred, race_date=pred.race_date)
    bt.record_result(race_id, actual_result=[5, 3, 8], win_dividend=40.0, is_real_result=True)

    rec = bt._records[race_id]
    assert len(rec.feature_rows) == 3

    by_horse = {int(r["horse_number"]): r for r in rec.feature_rows}
    assert by_horse[5]["is_top3"] == 1
    assert by_horse[5]["is_winner"] == 1
    assert by_horse[5]["finish_position"] == 1
    assert by_horse[7]["is_top3"] == 0


def test_backtester_invalid_inputs(tmp_path):
    history_path = tmp_path / "history.json"
    bt = Backtester(history_file=str(history_path))

    with pytest.raises(ValueError):
        bt.record_prediction(race_id="", prediction_result=DummyPredictionResult(), race_date="2026-04-01")

    with pytest.raises(ValueError):
        bt.record_prediction(race_id="2026-04-03_ST_2", prediction_result=None, race_date="2026-04-03")

    with pytest.raises(ValueError):
        bt.record_result("2026-04-03_ST_2", actual_result=[0, -1], win_dividend=10.0)


def test_backtester_history_consistency_from_file(tmp_path):
    history_path = tmp_path / "history.json"
    sample_record = {
        "2026-04-01_ST_1": {
            "race_id": "2026-04-01_ST_1",
            "race_date": "2026-04-01",
            "race_number": 1,
            "venue": "Sha Tin",
            "distance": 1200,
            "race_class": "Class 4",
            "going": "GOOD",
            "predicted_top3": [1, 2, 3],
            "predicted_winner": 1,
            "predicted_winner_odds": 5.0,
            "predicted_top3_odds": [5.0, 6.0, 7.0],
            "actual_result": [1, 2, 3],
            "actual_winner": 1,
            "winner_correct": True,
            "top3_hit": 3,
            "trio_hit": True,
            "win_bet_return": 50.0,
            "place_bet_return": 20.0,
            "trio_bet_return": 100.0,
            "model_confidence": 0.8,
            "created_at": date.today().isoformat(),
            "result_recorded_at": date.today().isoformat(),
        }
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(sample_record, f, ensure_ascii=False)

    bt = Backtester(history_file=str(history_path))
    assert len(bt._records) == 1

    summary = bt.calculate_roi("top1_win", 30)
    assert summary.total_races == 1
    assert summary.winner_correct == 1


def test_backtester_load_history_ignores_unknown_fields(tmp_path):
    history_path = tmp_path / "history.json"
    payload = {
        "2026-04-01_ST_1": {
            "race_id": "2026-04-01_ST_1",
            "race_date": "2026-04-01",
            "race_number": 1,
            "venue": "Sha Tin",
            "distance": 1200,
            "race_class": "Class 4",
            "going": "GOOD",
            "predicted_top3": [1, 2, 3],
            "predicted_winner": 1,
            "predicted_winner_odds": 5.0,
            "predicted_top3_odds": [5.0, 6.0, 7.0],
            "feature_rows": [],
            "actual_result": [1, 2, 3],
            "actual_winner": 1,
            "winner_correct": True,
            "top3_hit": 3,
            "trio_hit": True,
            "win_bet_return": 50.0,
            "place_bet_return": 20.0,
            "trio_bet_return": 100.0,
            "model_confidence": 0.8,
            "unknown_extra_key": "should_be_ignored",
        }
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    bt = Backtester(history_file=str(history_path))
    assert "2026-04-01_ST_1" in bt._records


def test_build_features_race_datetime_timezone():
    race = Race(
        race_number=1,
        race_name="Test Race",
        venue="Sha Tin",
        venue_code="ST",
        distance=1200,
        track_type="Turf",
        track_config="A",
        race_class="Class 4",
        class_rating_upper=60,
        class_rating_lower=40,
        going="GOOD",
        prize=1000000,
        start_time="14:30",
        race_date="2026-04-03",
        horses=[
            HorseEntry(
                horse_number=1,
                horse_name="Test Horse",
                horse_name_ch="測試馬",
                horse_code="HK_123",
                jockey="A Jockey",
                jockey_code="AJ",
                jockey_allowance=0,
                trainer="A Trainer",
                trainer_code="AT",
                draw=1,
                weight=130,
                handicap_rating=70,
                rating_change=0,
                last_6_runs="1/2/3/4/5/6",
                gear="TT",
            )
        ]
    )

    from features.elo import ELOSystem
    from features.jockey_trainer import JockeyTrainerStats
    from features.draw import DrawStats

    elo = ELOSystem()
    jt = JockeyTrainerStats()
    ds = DrawStats()
    ds.bootstrap_from_mock()

    features_df = build_features(race, {}, {}, elo, jt, ds)
    assert "race_datetime" in features_df.columns
    dt = features_df.loc[0, "race_datetime"]
    assert getattr(dt, "tzinfo", None) is not None
    assert dt.tzinfo == ZoneInfo("Asia/Hong_Kong")


def test_backtester_repair_missing_future_and_invalid(tmp_path):
    history_path = tmp_path / "history.json"
    bad_records = {
        "2026-12-31_ST_1": {
            "race_id": "2026-12-31_ST_1",
            "race_date": "2026-12-31",
            "race_number": 1,
            "venue": "Sha Tin",
            "distance": 1200,
            "race_class": "Class 4",
            "going": "GOOD",
            "predicted_top3": [1, 2, 3],
            "predicted_winner": 1,
            "predicted_winner_odds": 5.0,
            "predicted_top3_odds": [5.0, 6.0, 7.0],
            "actual_result": [1, 2, 3],
            "actual_winner": 1,
        },
        "invalid_date": {
            "race_id": "invalid_date",
            "race_date": "2026-99-99",
            "race_number": 2,
            "venue": "Sha Tin",
            "distance": 1200,
            "race_class": "Class 4",
            "going": "GOOD",
            "predicted_top3": [1, 2, 3],
            "predicted_winner": 1,
            "predicted_winner_odds": 5.0,
            "predicted_top3_odds": [5.0, 6.0, 7.0],
            "actual_result": [1, 2, 3],
            "actual_winner": 1,
        },
        "valid": {
            "race_id": "2026-04-01_ST_1",
            "race_date": "2026-04-01",
            "race_number": 1,
            "venue": "Sha Tin",
            "distance": 1200,
            "race_class": "Class 4",
            "going": "GOOD",
            "predicted_top3": [1, 2, 3],
            "predicted_winner": 1,
            "predicted_winner_odds": 5.0,
            "predicted_top3_odds": [5.0, 6.0, 7.0],
            "actual_result": [1, 2, 3],
            "actual_winner": 1,
        },
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(bad_records, f, ensure_ascii=False)

    bt = Backtester(history_file=str(history_path), repair_missing=True)
    assert "valid" in bt._records
    assert bt._records["valid"].race_id == "2026-04-01_ST_1"
    assert "2026-12-31_ST_1" not in bt._records
    assert "invalid_date" not in bt._records


def test_odds_timestamp_integration_for_mock():
    from scraper.odds import mock_odds

    odds = mock_odds(race_number=1, n_horses=3)
    assert isinstance(odds, dict)
    for record in odds.values():
        assert "odds_ts" in record
        assert record["odds_ts"] is not None
        assert "T" in record["odds_ts"]


def test_odds_timestamp_for_playwright_path():
    from scraper.odds import fetch_odds

    # Use mock path by setting DEMO_MODE, no Playwright needed.
    from config import config
    config.DEMO_MODE = True

    odds = fetch_odds("2026-04-03", "ST", 1)
    for record in odds.values():
        assert "odds_ts" in record
        assert record["odds_ts"] is not None


def test_backtester_risk_guard_status(tmp_path):
    history_path = tmp_path / "history.json"
    today_str = date.today().strftime("%Y-%m-%d")
    bt = Backtester(history_file=str(history_path))
    bt._records = {
        "r1": RacePredictionRecord(
            race_id="r1",
            race_date=today_str,
            race_number=1,
            venue="Sha Tin",
            distance=1200,
            race_class="Class 4",
            going="GOOD",
            predicted_top3=[1, 2, 3],
            predicted_winner=1,
            predicted_winner_odds=5.0,
            predicted_top3_odds=[5.0, 6.0, 7.0],
            actual_result=[5, 6, 7],
            actual_winner=5,
            win_bet_return=0.0,
        ),
        "r2": RacePredictionRecord(
            race_id="r2",
            race_date=today_str,
            race_number=2,
            venue="Sha Tin",
            distance=1200,
            race_class="Class 4",
            going="GOOD",
            predicted_top3=[1, 2, 3],
            predicted_winner=1,
            predicted_winner_odds=5.0,
            predicted_top3_odds=[5.0, 6.0, 7.0],
            actual_result=[4, 6, 7],
            actual_winner=4,
            win_bet_return=0.0,
        ),
        "r3": RacePredictionRecord(
            race_id="r3",
            race_date=today_str,
            race_number=3,
            venue="Sha Tin",
            distance=1200,
            race_class="Class 4",
            going="GOOD",
            predicted_top3=[1, 2, 3],
            predicted_winner=1,
            predicted_winner_odds=5.0,
            predicted_top3_odds=[5.0, 6.0, 7.0],
            actual_result=[8, 6, 7],
            actual_winner=8,
            win_bet_return=0.0,
        ),
    }

    status = bt.get_risk_guard_status(target_date=today_str, consecutive_loss_limit=3, daily_loss_limit=20.0)
    assert status["consecutive_losses"] == 3
    assert status["halt_due_to_streak"] is True
    assert status["halt_due_to_daily_loss"] is True
    assert status["halt"] is True


def test_record_prediction_does_not_overwrite_settled_race(tmp_path):
    history_path = tmp_path / "history.json"
    bt = Backtester(history_file=str(history_path))
    race_id = "2026-04-03_ST_9"
    bt._records[race_id] = RacePredictionRecord(
        race_id=race_id,
        race_date="2026-04-03",
        race_number=9,
        venue="Sha Tin",
        distance=1200,
        race_class="Class 4",
        going="GOOD",
        predicted_top3=[1, 2, 3],
        predicted_winner=1,
        predicted_winner_odds=9.9,
        predicted_top3_odds=[9.9, 10.0, 10.1],
        actual_result=[4, 5, 6],
        actual_winner=4,
        win_bet_return=0.0,
    )

    pred = DummyPredictionResult()
    pred.top5 = [type("H", (), {"horse_number": 8, "win_odds": 3.0})()]
    bt.record_prediction(race_id=race_id, prediction_result=pred, race_date="2026-04-03")

    assert bt._records[race_id].predicted_winner == 1
    assert bt._records[race_id].actual_winner == 4
