"""
HKJC 賽馬預測系統 - 回測模組
Backtesting and ROI tracking for prediction performance.
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict, fields
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HISTORY_FILE = "data/predictions/history.json"


def _parse_date_str(date_str: str) -> date:
    """Parse 2026-04-15 style date string and validate format."""
    if not isinstance(date_str, str) or not date_str:
        raise ValueError(f"Invalid race_date: {date_str!r}")
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"race_date must be YYYY-MM-DD, got: {date_str}") from exc
    return parsed


def _log_sample(stage: str, records: List, n: int = 3) -> None:
    """Log small sample of backtest records for auditing."""
    logger.info("[%s] record count=%d", stage, len(records))
    for r in list(records)[:n]:
        logger.info(
            "[%s] sample: race_id=%s date=%s winner=%s top3=%s win_odds=%s",
            stage,
            getattr(r, 'race_id', '<NA>'),
            getattr(r, 'race_date', '<NA>'),
            getattr(r, 'predicted_winner', '<NA>'),
            getattr(r, 'predicted_top3', '<NA>'),
            getattr(r, 'predicted_winner_odds', '<NA>'),
        )


@dataclass
class RacePredictionRecord:
    """Stored record of a prediction and its outcome."""
    race_id: str                    # "{date}_{venue}_{race_number}"
    race_date: str                  # "YYYY-MM-DD"
    race_number: int
    venue: str
    distance: int
    race_class: str
    going: str

    # Predictions (top 3 picks by the model)
    predicted_top3: List[int]       # horse numbers
    predicted_winner: int           # top pick horse number

    # Market data at prediction time
    predicted_winner_odds: float
    predicted_top3_odds: List[float]
    feature_rows: List[Dict] = field(default_factory=list)

    # Actual results (filled in after race)
    actual_result: List[int] = field(default_factory=list)   # finish order
    actual_winner: int = 0

    # Outcomes
    winner_correct: bool = False    # predicted_winner == actual_winner
    top3_hit: int = 0               # how many of predicted_top3 finished top-3
    trio_hit: bool = False          # all 3 predicted_top3 finished in top-3

    # Financials (per $10 unit)
    win_bet_return: float = 0.0     # return from betting on predicted_winner
    place_bet_return: float = 0.0   # return from betting on place
    trio_bet_return: float = 0.0    # return from Trio bet

    # Metadata
    model_confidence: float = 0.0
    is_real_result: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    result_recorded_at: str = ""


@dataclass
class BacktestResult:
    """Aggregated backtest statistics for a period."""
    period_days: int
    total_races: int
    winner_correct: int
    top3_hit_rate: float            # avg proportion of correct top-3 picks
    trio_hit_rate: float
    win_roi: float                  # ROI on win bets (stake = $10/race)
    place_roi: float
    trio_roi: float
    total_invested: float
    total_returned: float
    net_profit: float
    races_with_value_bets: int = 0


class Backtester:
    """
    Records predictions and computes ROI / performance metrics.

    Supports multiple strategies:
    - "top1_win": Bet $10 win on top-1 predicted horse each race
    - "top1_place": Bet $10 place on top-1 predicted horse each race
    - "top3_place": Bet $10 place on each of top-3 picks
    - "trio": Bet $10 Trio on top-3 picks
    """

    def __init__(self, history_file: str = HISTORY_FILE, repair_missing: bool = False):
        self.history_file = history_file
        self.repair_missing = repair_missing
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        self._records: Dict[str, RacePredictionRecord] = {}
        self.load_history()
        if self.repair_missing:
            self._repair_history()
        self._assert_history_consistency()
        _log_sample("init", list(self._records.values()), n=3)

    def _repair_history(self) -> None:
        """Attempt to auto-fix history issues (invalid date, future race, bad top3)."""
        cleaned: Dict[str, RacePredictionRecord] = {}
        removed_count = 0

        for rid, rec in self._records.items():
            valid = True

            if not isinstance(rid, str) or not rid.strip():
                logger.warning("Repair: dropping invalid race_id entry: %r", rid)
                removed_count += 1
                continue

            if not isinstance(rec, RacePredictionRecord):
                logger.warning("Repair: dropping non-record object for race_id %s", rid)
                removed_count += 1
                continue

            try:
                d = _parse_date_str(rec.race_date)
            except ValueError:
                logger.warning("Repair: dropping record with invalid date %r for %s", rec.race_date, rid)
                removed_count += 1
                continue

            if d > date.today():
                logger.warning("Repair: dropping future-dated record %s (%s)", rid, rec.race_date)
                removed_count += 1
                continue

            if not isinstance(rec.predicted_top3, list):
                logger.warning("Repair: converted predicted_top3 to empty list for %s", rid)
                rec.predicted_top3 = []

            cleaned[rid] = rec

        self._records = cleaned
        if removed_count > 0:
            logger.info("Repair: removed %d bad history entries", removed_count)
            self.save_history()

    def _assert_history_consistency(self) -> None:
        """Check no duplicate IDs, date formats, and basic financial consistency."""
        race_ids = list(self._records.keys())
        if len(race_ids) != len(set(race_ids)):
            raise ValueError("Duplicate race_id found in history")

        for rid, rec in self._records.items():
            _parse_date_str(rec.race_date)
            if not isinstance(rec.predicted_top3, list):
                raise ValueError(f"predicted_top3 must be list for {rid}")
            if rec.predicted_winner and rec.predicted_winner not in rec.predicted_top3:
                logger.warning("predicted_winner not in predicted_top3 for %s", rid)

    def record_prediction(
        self,
        race_id: str,
        prediction_result,   # PredictionResult from predictor
        race_date: str,
    ) -> str:
        """
        Record a prediction for a race (before result is known).

        Args:
            race_id: Unique race identifier
            prediction_result: PredictionResult from RacePredictor
            race_date: "YYYY-MM-DD"

        Returns:
            race_id for later reference
        """
        if not race_id or not isinstance(race_id, str):
            raise ValueError("race_id must be non-empty string")
        _parse_date_str(race_date)

        existing = self._records.get(race_id)
        if existing and existing.actual_winner > 0:
            logger.warning(
                "Skip record_prediction for settled race %s (actual_winner=%s)",
                race_id,
                existing.actual_winner,
            )
            return race_id

        pr = prediction_result
        if pr is None:
            raise ValueError("prediction_result cannot be None")
        if not getattr(pr, 'top5', None):
            logger.warning("Prediction result has no top5; this race may have insufficient horses: %s", race_id)

        top3 = pr.top5[:3] if pr.top5 else []
        recommended = getattr(pr, "recommended_winner", None)
        selected = recommended if recommended is not None else (top3[0] if top3 else None)

        record = RacePredictionRecord(
            race_id=race_id,
            race_date=race_date,
            race_number=pr.race_number,
            venue=pr.venue,
            distance=pr.distance,
            race_class=pr.race_class,
            going=pr.going,
            predicted_top3=[h.horse_number for h in top3],
            predicted_winner=selected.horse_number if selected else 0,
            predicted_winner_odds=selected.win_odds if selected else 0.0,
            predicted_top3_odds=[h.win_odds for h in top3],
            feature_rows=list(getattr(pr, "feature_rows", []) or []),
            model_confidence=getattr(selected, "confidence", pr.confidence) if selected else pr.confidence,
        )
        self._records[race_id] = record
        if self.repair_missing:
            self._repair_history()
        self._assert_history_consistency()
        _log_sample("record_prediction", [record], n=1)
        self.save_history()
        logger.info(f"Recorded prediction for {race_id}: top3={record.predicted_top3}")
        return race_id

    def record_result(
        self,
        race_id: str,
        actual_result: List[int],
        trio_dividend: float = 0.0,
        place_dividends: Optional[Dict[int, float]] = None,
        win_dividend: Optional[float] = None,
        is_real_result: bool = False,
    ) -> Optional[RacePredictionRecord]:
        """
        Record the actual race result and compute financial outcomes.

        Args:
            race_id: Race identifier
            actual_result: Finishing order as list of horse numbers (1st, 2nd, 3rd...)
            trio_dividend: Trio dividend per $10
            place_dividends: Dict of horse_number -> place dividend per $10
            win_dividend: Win dividend per $10 for the winner

        Returns:
            Updated RacePredictionRecord
        """
        if not isinstance(actual_result, list) or not actual_result:
            raise ValueError("actual_result must be a non-empty list of horse numbers")
        if any(not isinstance(r, int) or r <= 0 for r in actual_result):
            raise ValueError("actual_result values must be positive integers")

        record = self._records.get(race_id)
        if not record:
            logger.warning(f"No prediction record found for {race_id}")
            return None

        record.actual_result = actual_result[:10]  # store top-10
        record.actual_winner = actual_result[0] if actual_result else 0
        record.is_real_result = bool(is_real_result)
        record.result_recorded_at = datetime.now().isoformat()

        race_date_obj = _parse_date_str(record.race_date)
        if race_date_obj > date.today():
            logger.warning("Actual result for race dated in future: %s", record.race_date)

        actual_top3 = set(actual_result[:3])

        # Winner correct?
        record.winner_correct = record.predicted_winner == record.actual_winner

        # Top-3 hits
        predicted_set = set(record.predicted_top3)
        record.top3_hit = len(predicted_set & actual_top3)

        # Trio hit: all 3 predicted horses finished in top-3
        record.trio_hit = (predicted_set == actual_top3 and len(predicted_set) == 3)

        # Financial calculations (per $10 stake)
        STAKE = 10.0

        # Win bet on predicted_winner
        if record.winner_correct:
            record.win_bet_return = win_dividend if win_dividend else record.predicted_winner_odds * STAKE
        else:
            record.win_bet_return = 0.0

        # Place bet on predicted_winner
        if record.predicted_winner in actual_top3:
            pd_dividend = (place_dividends or {}).get(record.predicted_winner, 0)
            record.place_bet_return = pd_dividend if pd_dividend else STAKE * 2.0
        else:
            record.place_bet_return = 0.0

        # Trio bet
        record.trio_bet_return = trio_dividend if record.trio_hit and trio_dividend else 0.0

        # Auto-label stored feature snapshot for true incremental training.
        if record.feature_rows:
            finish_pos_map = {horse_no: idx + 1 for idx, horse_no in enumerate(actual_result)}
            labeled_rows: List[Dict] = []
            for row in record.feature_rows:
                if not isinstance(row, dict):
                    continue
                horse_no = int(row.get("horse_number", 0) or 0)
                if horse_no <= 0:
                    continue
                out = dict(row)
                out["race_id"] = record.race_id
                out["race_date"] = record.race_date
                out["is_top3"] = 1 if horse_no in actual_top3 else 0
                out["is_winner"] = 1 if horse_no == record.actual_winner else 0
                out["finish_position"] = int(finish_pos_map.get(horse_no, 0))
                out["is_real_result"] = int(bool(record.is_real_result))
                out["label_recorded_at"] = record.result_recorded_at
                labeled_rows.append(out)
            record.feature_rows = labeled_rows

        self._records[race_id] = record
        if self.repair_missing:
            self._repair_history()
        self._assert_history_consistency()
        _log_sample("record_result", [record], n=1)
        self.save_history()

        logger.info(
            f"Result recorded for {race_id}: "
            f"winner={'✓' if record.winner_correct else '✗'}, "
            f"top3_hits={record.top3_hit}/3, "
            f"trio={'✓' if record.trio_hit else '✗'}"
        )
        return record

    def calculate_roi(self, strategy: str = "top1_win", period_days: int = 30, real_only: bool = False) -> BacktestResult:
        """
        Calculate ROI for a given strategy over a time period.

        Args:
            strategy: "top1_win" | "top1_place" | "top3_place" | "trio"
            period_days: Look-back window in days

        Returns:
            BacktestResult with aggregated metrics
        """
        if strategy not in {"top1_win", "top1_place", "top3_place", "trio"}:
            raise ValueError(f"Unknown strategy: {strategy}")

        cutoff = (date.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
        STAKE = 10.0

        recent = [
            r for r in self._records.values()
            if r.race_date >= cutoff and r.actual_winner > 0 and (not real_only or r.is_real_result)
        ]

        if not recent:
            return BacktestResult(
                period_days=period_days,
                total_races=0,
                winner_correct=0,
                top3_hit_rate=0.0,
                trio_hit_rate=0.0,
                win_roi=0.0,
                place_roi=0.0,
                trio_roi=0.0,
                total_invested=0.0,
                total_returned=0.0,
                net_profit=0.0,
            )

        # no duplicates and race_date ordering
        recent.sort(key=lambda x: x.race_date)
        assert all(recent[i].race_date <= recent[i+1].race_date for i in range(len(recent)-1)), "race_date not sorted"

        winner_correct = sum(1 for r in recent if r.winner_correct)
        top3_hits = [r.top3_hit / 3.0 for r in recent]
        trio_hits = sum(1 for r in recent if r.trio_hit)

        if strategy == "top1_win":
            total_invested = len(recent) * STAKE
            total_returned = sum(r.win_bet_return for r in recent)
        elif strategy == "top1_place":
            total_invested = len(recent) * STAKE
            total_returned = sum(r.place_bet_return for r in recent)
        elif strategy == "top3_place":
            total_invested = len(recent) * STAKE * 3
            # top3_hit counts how many of top3 were in actual top3, there are 3 place bets per race
            total_returned = sum(r.top3_hit * STAKE * 2.0 for r in recent)
        else:  # trio
            total_invested = len(recent) * STAKE
            total_returned = sum(r.trio_bet_return for r in recent)

        net_profit = total_returned - total_invested
        roi = (net_profit / max(total_invested, 1)) * 100

        win_roi = round((sum(r.win_bet_return for r in recent) - len(recent) * STAKE) / max(len(recent) * STAKE, 1) * 100, 2)
        place_roi = round((sum(r.place_bet_return for r in recent) - len(recent) * STAKE) / max(len(recent) * STAKE, 1) * 100, 2)
        trio_roi = round((sum(r.trio_bet_return for r in recent) - len(recent) * STAKE) / max(len(recent) * STAKE, 1) * 100, 2)

        return BacktestResult(
            period_days=period_days,
            total_races=len(recent),
            winner_correct=winner_correct,
            top3_hit_rate=round(float(np.mean(top3_hits)), 4) if top3_hits else 0.0,
            trio_hit_rate=round(trio_hits / max(len(recent), 1), 4),
            win_roi=round(roi, 2) if strategy == "top1_win" else win_roi,
            place_roi=round(roi, 2) if strategy == "top1_place" else place_roi,
            trio_roi=round(roi, 2) if strategy == "trio" else trio_roi,
            total_invested=round(total_invested, 2),
            total_returned=round(total_returned, 2),
            net_profit=round(net_profit, 2),
        )

    def get_risk_guard_status(
        self,
        target_date: Optional[str] = None,
        consecutive_loss_limit: int = 3,
        daily_loss_limit: float = 30.0,
        stake: float = 10.0,
    ) -> Dict:
        """
        Compute whether risk guard should halt further betting-style notifications.

        A loss is defined against the top1 win strategy using settled records only.
        """
        target_date = target_date or date.today().strftime("%Y-%m-%d")
        _parse_date_str(target_date)

        settled = [r for r in self._records.values() if r.actual_winner > 0]
        settled.sort(key=lambda r: (r.race_date, r.race_number))

        daily_records = [r for r in settled if r.race_date == target_date]
        daily_net_profit = round(sum(r.win_bet_return - stake for r in daily_records), 2)

        consecutive_losses = 0
        for rec in reversed(settled):
            if rec.win_bet_return > 0:
                break
            consecutive_losses += 1

        halt_due_to_streak = consecutive_losses >= consecutive_loss_limit
        halt_due_to_daily_loss = daily_net_profit <= -abs(daily_loss_limit)

        return {
            "target_date": target_date,
            "settled_races_today": len(daily_records),
            "daily_net_profit": daily_net_profit,
            "consecutive_losses": consecutive_losses,
            "halt_due_to_streak": halt_due_to_streak,
            "halt_due_to_daily_loss": halt_due_to_daily_loss,
            "halt": halt_due_to_streak or halt_due_to_daily_loss,
        }

    def get_settled_records(self, real_only: bool = False) -> List[RacePredictionRecord]:
        """Return settled race records, optionally filtering to real-world results only."""
        return [
            r for r in self._records.values()
            if r.actual_winner > 0 and (not real_only or r.is_real_result)
        ]

    def get_summary_stats(self, real_only: bool = False) -> Dict:
        """
        Return summary statistics for the dashboard.

        Returns:
            Dict with win_rate, place_rate, roi_7d, roi_30d, etc.
        """
        result_records = self.get_settled_records(real_only=real_only)
        total = len(result_records)

        if total == 0:
            return _empty_summary()

        r7 = self.calculate_roi("top1_win", 7, real_only=real_only)
        r30 = self.calculate_roi("top1_win", 30, real_only=real_only)

        recent_records = sorted(result_records, key=lambda r: r.race_date, reverse=True)[:20]

        return {
            "total_predictions": total,
            "winner_accuracy": round(sum(1 for r in result_records if r.winner_correct) / total, 4),
            "top3_hit_rate": round(
                sum(r.top3_hit / 3.0 for r in result_records) / total, 4
            ),
            "trio_hit_rate": round(sum(1 for r in result_records if r.trio_hit) / total, 4),
            "roi_7d": r7.win_roi,
            "roi_30d": r30.win_roi,
            "place_roi_30d": r30.place_roi,
            "trio_roi_30d": r30.trio_roi,
            "net_profit_30d": r30.net_profit,
            "races_7d": r7.total_races,
            "races_30d": r30.total_races,
            "recent_records": [
                {
                    "race_id": r.race_id,
                    "date": r.race_date,
                    "race": f"第{r.race_number}場",
                    "venue": r.venue,
                    "predicted": r.predicted_top3,
                    "actual_top3": r.actual_result[:3],
                    "winner_correct": r.winner_correct,
                    "top3_hits": r.top3_hit,
                    "confidence": r.model_confidence,
                    "is_real_result": r.is_real_result,
                }
                for r in recent_records
            ],
        }

    def save_history(self) -> None:
        """Save prediction history to JSON."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        data = {k: asdict(v) for k, v in self._records.items()}
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_history(self) -> None:
        """Load prediction history from JSON."""
        from config import config

        if not os.path.exists(self.history_file):
            logger.info("No prediction history found; starting fresh")
            self._records = {}
            if config.DEMO_MODE and config.SEED_DEMO_BACKTEST_HISTORY:
                self._seed_demo_history()
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            allowed_keys = {f.name for f in fields(RacePredictionRecord)}
            records: Dict[str, RacePredictionRecord] = {}
            dropped = 0
            for k, v in data.items():
                if not isinstance(v, dict):
                    dropped += 1
                    continue
                filtered = {kk: vv for kk, vv in v.items() if kk in allowed_keys}
                try:
                    records[k] = RacePredictionRecord(**filtered)
                except Exception:
                    dropped += 1
            self._records = records
            if dropped:
                logger.warning("Dropped %s invalid history entries while loading", dropped)
            logger.info(f"Loaded {len(self._records)} prediction records")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self._records = {}
            if config.DEMO_MODE and config.SEED_DEMO_BACKTEST_HISTORY:
                self._seed_demo_history()

    def _seed_demo_history(self) -> None:
        """Seed with realistic demo history for UI testing."""
        import random
        rng = random.Random(12345)
        venues = ["Sha Tin", "Happy Valley"]
        today = date.today()

        for i in range(40):
            race_date = (today - timedelta(days=i * 3 + rng.randint(0, 2))).strftime("%Y-%m-%d")
            venue = rng.choice(venues)
            race_num = rng.randint(1, 10)
            race_id = f"{race_date}_{venue[:2].upper()}_{race_num}"

            n_runners = rng.randint(8, 14)
            predicted = rng.sample(range(1, n_runners + 1), 3)
            actual = rng.sample(range(1, n_runners + 1), min(n_runners, 10))
            winner_correct = predicted[0] == actual[0]
            actual_top3 = set(actual[:3])
            top3_hit = len(set(predicted) & actual_top3)
            trio_hit = (set(predicted) == actual_top3)

            win_odds = rng.uniform(2.5, 20.0)
            win_return = win_odds * 10.0 if winner_correct else 0.0
            place_return = 10.0 * rng.uniform(1.5, 2.5) if predicted[0] in actual_top3 else 0.0

            rec = RacePredictionRecord(
                race_id=race_id,
                race_date=race_date,
                race_number=race_num,
                venue=venue,
                distance=rng.choice([1200, 1400, 1600, 2000]),
                race_class=rng.choice(["Class 2", "Class 3", "Class 4"]),
                going=rng.choice(["GOOD", "GOOD", "GOOD TO YIELDING"]),
                predicted_top3=predicted,
                predicted_winner=predicted[0],
                predicted_winner_odds=round(win_odds, 1),
                predicted_top3_odds=[round(rng.uniform(2, 20), 1) for _ in range(3)],
                actual_result=actual,
                actual_winner=actual[0],
                winner_correct=winner_correct,
                top3_hit=top3_hit,
                trio_hit=trio_hit,
                win_bet_return=round(win_return, 2),
                place_bet_return=round(place_return, 2),
                trio_bet_return=round(rng.uniform(80, 300) * 10 if trio_hit else 0.0, 2),
                model_confidence=round(rng.uniform(0.3, 0.8), 3),
            )
            self._records[race_id] = rec

        self.save_history()
        logger.info(f"Seeded {len(self._records)} demo history records")


def _empty_summary() -> Dict:
    return {
        "total_predictions": 0,
        "winner_accuracy": 0.0,
        "top3_hit_rate": 0.0,
        "trio_hit_rate": 0.0,
        "roi_7d": 0.0,
        "roi_30d": 0.0,
        "place_roi_30d": 0.0,
        "trio_roi_30d": 0.0,
        "net_profit_30d": 0.0,
        "races_7d": 0,
        "races_30d": 0,
        "recent_records": [],
    }
