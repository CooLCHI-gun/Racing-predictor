"""
HKJC 賽馬預測系統 - 策略自我優化器
Tune recommendation thresholds using settled real-data backtest records.

Improvements over v1:
- Confidence calibration (bucketed actual vs predicted win rate)
- Top-3 finish rate and place ROI evaluation alongside winner metrics
- Expanded grid (9 confidence × 6 odds = 54 combos vs 30)
- Optimisation history tracking
- Removed redundant _optimize_simple code path
- All weights configurable via StrategyProfile
"""
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from model.backtest import RacePredictionRecord

logger = logging.getLogger(__name__)

# Calibration bucket boundaries (model confidence ranges)
CALIBRATION_BUCKETS = [
    (0.0, 0.3, "very_low"),
    (0.3, 0.4, "low"),
    (0.4, 0.5, "medium_low"),
    (0.5, 0.6, "medium"),
    (0.6, 0.7, "medium_high"),
    (0.7, 0.8, "high"),
    (0.8, 0.9, "very_high"),
    (0.9, 1.0, "extreme"),
]

# Grid search ranges
CONFIDENCE_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MAX_ODDS_GRID = (4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 25.0, 35.0, 50.0)

HISTORY_FILE = "data/models/optimization_history.json"


def _distance_bucket(distance: int) -> str:
    """Map race distance (m) into strategy buckets."""
    d = int(distance or 0)
    if d <= 1200:
        return "sprint"
    if d <= 1600:
        return "mile"
    if d <= 2000:
        return "middle"
    return "long"


def _venue_code_from_text(venue: str) -> str:
    txt = str(venue or "").strip().lower()
    if txt in {"st", "sha tin"}:
        return "ST"
    if txt in {"hv", "happy valley"}:
        return "HV"
    return "UNK"


@dataclass
class StrategyProfile:
    min_confidence: float = 0.55
    max_win_odds: float = 18.0
    min_samples: int = 24
    objective_weight_win_rate: float = 0.40
    objective_weight_roi: float = 0.30
    objective_weight_top3: float = 0.15
    objective_weight_place_roi: float = 0.15
    selected_samples: int = 0
    selected_win_rate: float = 0.0
    selected_roi: float = 0.0
    selected_top3_rate: float = 0.0
    selected_place_roi: float = 0.0
    selected_score: float = 0.0
    optimized_at: str = ""


@dataclass
class CalibrationBucket:
    label: str
    count: int
    model_avg_conf: float
    actual_win_rate: float
    actual_place_rate: float
    actual_top3_rate: float


@dataclass
class OptimizationEntry:
    timestamp: str
    params: dict
    score: float
    samples: int
    win_rate: float
    roi: float


class StrategySelfOptimizer:
    """Grid-search optimizer for win-rate/ROI balanced recommendation filters."""

    def __init__(self, strategy_file: str = "data/models/strategy_profile.json"):
        self.strategy_file = strategy_file
        os.makedirs(os.path.dirname(strategy_file), exist_ok=True)
        self.segment_strategy_file = os.path.join(
            os.path.dirname(strategy_file),
            "strategy_profile_segments.json",
        )

    # ── Load / Save ──────────────────────────────────────────────────────────

    def load(self) -> StrategyProfile:
        if not os.path.exists(self.strategy_file):
            return StrategyProfile()
        try:
            with open(self.strategy_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # Backward compat: fill missing new fields
            profile = StrategyProfile(**payload)
            return profile
        except Exception as e:
            logger.warning("Failed to load strategy profile, using defaults: %s", e)
            return StrategyProfile()

    def save(self, profile: StrategyProfile) -> None:
        with open(self.strategy_file, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    # ── Calibration ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_calibration(records: List[RacePredictionRecord]) -> List[CalibrationBucket]:
        """
        Compute calibration buckets: for each confidence range, compare
        the model's average confidence against the actual win/place/top3 rates.

        A well-calibrated model has actual_win_rate ≈ model_avg_conf in each bucket.
        """
        if not records:
            return []

        buckets: Dict[str, List[RacePredictionRecord]] = {}
        bucket_labels: Dict[str, Tuple[float, float]] = {}
        for lo, hi, label in CALIBRATION_BUCKETS:
            buckets[label] = []
            bucket_labels[label] = (lo, hi)

        for r in records:
            conf = getattr(r, "model_confidence", 0.0) or 0.0
            for lo, hi, label in CALIBRATION_BUCKETS:
                if lo <= conf < hi:
                    buckets[label].append(r)
                    break

        results: List[CalibrationBucket] = []
        for lo, hi, label in CALIBRATION_BUCKETS:
            group = buckets.get(label, [])
            if not group:
                continue
            n = len(group)
            conf_avg = sum(getattr(r, "model_confidence", 0.0) or 0.0 for r in group) / n
            wins = sum(1 for r in group if getattr(r, "winner_correct", False))
            places = sum(1 for r in group if getattr(r, "place_bet_return", 0.0) > 0)
            top3_hits = sum(getattr(r, "top3_hit", 0) for r in group) / 3.0  # normalise
            actual_top3_rate = top3_hits / max(n, 1)
            results.append(CalibrationBucket(
                label=label,
                count=n,
                model_avg_conf=round(conf_avg, 4),
                actual_win_rate=round(wins / n, 4),
                actual_place_rate=round(places / n, 4),
                actual_top3_rate=round(actual_top3_rate, 4),
            ))

        return results

    def print_calibration_report(self, records: List[RacePredictionRecord]) -> None:
        """Log calibration analysis for human review."""
        buckets = self.compute_calibration(records)
        if not buckets:
            return
        logger.info("─" * 60)
        logger.info("Confidence Calibration Report")
        logger.info("─" * 60)
        logger.info(f"{'Bucket':<18} {'Count':>6} {'Avg Conf':>9} {'Win Rate':>9} {'Place':>7} {'Top3':>6}")
        logger.info("-" * 56)
        for b in buckets:
            logger.info(
                f"{b.label:<18} {b.count:>6} {b.model_avg_conf:>8.3f} "
                f"{b.actual_win_rate:>8.3f} {b.actual_place_rate:>6.3f} {b.actual_top3_rate:>5.3f}"
            )
        logger.info("─" * 60)

    # ── Evaluation ───────────────────────────────────────────────────────────

    @staticmethod
    def _evaluate_thresholds(
        records: List[RacePredictionRecord],
        min_confidence: float,
        max_win_odds: float,
        stake: float = 10.0,
    ) -> Dict[str, float]:
        """
        Evaluate a threshold combo across win, place, and top-3.

        Returns:
            samples, win_rate, roi, place_roi, top3_rate, top3_full_hits
        """
        selected = [
            r for r in records
            if r.model_confidence >= min_confidence and 0 < r.predicted_winner_odds <= max_win_odds
        ]

        if not selected:
            return {
                "samples": 0, "win_rate": 0.0, "roi": 0.0,
                "place_roi": 0.0, "top3_rate": 0.0, "top3_full_hits": 0,
            }

        n = len(selected)

        # Win stats
        wins = sum(1 for r in selected if r.winner_correct)
        win_rate = wins / n
        total_invested = n * stake
        total_returned_w = sum(r.win_bet_return for r in selected)
        roi = ((total_returned_w - total_invested) / max(total_invested, 1.0)) * 100.0

        # Place stats
        total_returned_p = sum(r.place_bet_return for r in selected)
        place_roi = ((total_returned_p - total_invested) / max(total_invested, 1.0)) * 100.0

        # Top-3 hit rate
        top3_full = sum(1 for r in selected if r.top3_hit >= 3)
        top3_rate = top3_full / n

        return {
            "samples": n,
            "win_rate": win_rate,
            "roi": roi,
            "place_roi": place_roi,
            "top3_rate": top3_rate,
            "top3_full_hits": top3_full,
        }

    @staticmethod
    def _blend_score(
        stats: Dict[str, float],
        w_win: float = 0.40,
        w_roi: float = 0.30,
        w_top3: float = 0.15,
        w_place_roi: float = 0.15,
    ) -> float:
        """Multi-objective blend: win rate + ROI + top-3 + place ROI."""
        # Clamp ROI to a stable range to avoid overfitting to outliers
        roi_norm = max(0.0, min(1.0, (stats["roi"] + 30.0) / 60.0))
        place_roi_norm = max(0.0, min(1.0, (stats["place_roi"] + 20.0) / 50.0))
        return (
            w_win * stats["win_rate"]
            + w_roi * roi_norm
            + w_top3 * stats["top3_rate"]
            + w_place_roi * place_roi_norm
        )

    # ── Core Optimisation ───────────────────────────────────────────────────

    def _optimize_records(
        self,
        records: List[RacePredictionRecord],
        min_samples: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[StrategyProfile]:
        """
        Grid-search confidence × max_odds and return best profile.

        Uses chronological holdout (last 30% as validation) + stability penalty.
        """
        settled = [r for r in records if r.actual_winner > 0]
        if len(settled) < min_samples:
            return None

        settled.sort(key=lambda r: (str(r.race_date), int(getattr(r, "race_number", 0) or 0), str(r.race_id)))
        valid_size = max(12, int(len(settled) * 0.30))
        valid_size = min(valid_size, len(settled) - 1)
        calib = settled[:-valid_size]
        valid = settled[-valid_size:]

        min_samples_calib = max(min_samples, 24)
        min_samples_valid = max(15, min_samples // 2)

        w = weights or {}

        best_profile: Optional[StrategyProfile] = None
        best_score = -1.0

        for min_conf in CONFIDENCE_GRID:
            for max_odds in MAX_ODDS_GRID:
                stats_calib = self._evaluate_thresholds(calib, min_confidence=min_conf, max_win_odds=max_odds)
                stats_valid = self._evaluate_thresholds(valid, min_confidence=min_conf, max_win_odds=max_odds)

                if stats_calib["samples"] < min_samples_calib:
                    continue
                if stats_valid["samples"] < min_samples_valid:
                    continue

                score_calib = self._blend_score(stats_calib, **w)
                score_valid = self._blend_score(stats_valid, **w)

                win_gap = abs(stats_calib["win_rate"] - stats_valid["win_rate"])
                roi_gap = abs(stats_calib["roi"] - stats_valid["roi"]) / 100.0
                stability_penalty = (0.50 * win_gap) + (0.30 * roi_gap)

                score = (0.35 * score_calib) + (0.65 * score_valid) - (0.15 * stability_penalty)

                if score > best_score:
                    best_score = score
                    best_profile = StrategyProfile(
                        min_confidence=min_conf,
                        max_win_odds=max_odds,
                        min_samples=min_samples,
                        objective_weight_win_rate=w.get("w_win", 0.40),
                        objective_weight_roi=w.get("w_roi", 0.30),
                        objective_weight_top3=w.get("w_top3", 0.15),
                        objective_weight_place_roi=w.get("w_place_roi", 0.15),
                        selected_samples=int(stats_valid["samples"]),
                        selected_win_rate=round(stats_valid["win_rate"], 4),
                        selected_roi=round(stats_valid["roi"], 2),
                        selected_top3_rate=round(stats_valid["top3_rate"], 4),
                        selected_place_roi=round(stats_valid["place_roi"], 2),
                        selected_score=round(score, 4),
                        optimized_at=datetime.now().isoformat(),
                    )

        return best_profile

    def optimize(
        self,
        records: List[RacePredictionRecord],
        min_samples: int = 24,
        weights: Optional[Dict[str, float]] = None,
    ) -> StrategyProfile:
        """
        Search threshold combinations and persist best profile.

        Args:
            records: Settled backtest records
            min_samples: Minimum settled records required
            weights: Optional dict with keys w_win, w_roi, w_top3, w_place_roi
                     to override default blend weights
        """
        settled = [r for r in records if r.actual_winner > 0]
        if len(settled) < min_samples:
            logger.info("Skip optimization: only %s settled records (need >= %s)", len(settled), min_samples)
            return self.load()

        # Log calibration before optimising
        self.print_calibration_report(settled)

        best_profile = self._optimize_records(settled, min_samples=min_samples, weights=weights)

        if best_profile is None:
            logger.info("Optimization found no candidate meeting minimum samples")
            return self.load()

        self.save(best_profile)
        self._save_optimization_entry(best_profile)

        logger.info(
            "Strategy optimized: min_conf=%.2f max_odds=%.1f samples=%s "
            "win_rate=%.2f%% roi=%.2f%% top3=%.2f%% place_roi=%.2f%% score=%.4f",
            best_profile.min_confidence,
            best_profile.max_win_odds,
            best_profile.selected_samples,
            best_profile.selected_win_rate * 100,
            best_profile.selected_roi,
            best_profile.selected_top3_rate * 100,
            best_profile.selected_place_roi,
            best_profile.selected_score,
        )
        return best_profile

    # ── Segmented Optimisation ───────────────────────────────────────────────

    def optimize_segmented(
        self,
        records: List[RacePredictionRecord],
        min_samples: int = 12,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict]:
        """
        Optimize strategy per segment: ST/HV x sprint/mile/middle/long.
        """
        settled = [r for r in records if r.actual_winner > 0]
        if len(settled) < min_samples:
            logger.info("Skip segmented optimization: only %s settled records", len(settled))
            return self.load_segmented()

        global_profile = self.load()

        by_segment: Dict[str, List[RacePredictionRecord]] = {}
        for r in settled:
            venue_code = _venue_code_from_text(getattr(r, "venue", ""))
            if venue_code not in {"ST", "HV"}:
                continue
            bucket = _distance_bucket(int(getattr(r, "distance", 0) or 0))
            key = f"{venue_code}_{bucket}"
            by_segment.setdefault(key, []).append(r)

        out = {
            "updated_at": datetime.now().isoformat(),
            "min_samples": int(min_samples),
            "segments": {},
        }

        all_segment_keys = [
            f"{venue}_{bucket}"
            for venue in ("ST", "HV")
            for bucket in ("sprint", "mile", "middle", "long")
        ]

        for key in all_segment_keys:
            seg_records = list(by_segment.get(key, []))

            # Fallback to global if too few records
            if len(seg_records) < max(8, min_samples):
                out["segments"][key] = {
                    "min_confidence": global_profile.min_confidence,
                    "max_win_odds": global_profile.max_win_odds,
                    "selected_samples": len(seg_records),
                    "selected_win_rate": 0.0,
                    "selected_roi": 0.0,
                    "selected_top3_rate": 0.0,
                    "selected_place_roi": 0.0,
                    "selected_score": 0.0,
                    "optimized_at": datetime.now().isoformat(),
                    "mode": "fallback_global",
                }
                continue

            # Try full holdout validation path
            seg_profile = self._optimize_records(seg_records, min_samples=max(8, min_samples), weights=weights)
            if seg_profile is not None:
                out["segments"][key] = {
                    "min_confidence": seg_profile.min_confidence,
                    "max_win_odds": seg_profile.max_win_odds,
                    "selected_samples": seg_profile.selected_samples,
                    "selected_win_rate": seg_profile.selected_win_rate,
                    "selected_roi": seg_profile.selected_roi,
                    "selected_top3_rate": seg_profile.selected_top3_rate,
                    "selected_place_roi": seg_profile.selected_place_roi,
                    "selected_score": seg_profile.selected_score,
                    "optimized_at": seg_profile.optimized_at,
                    "mode": "segment_validated",
                }
                continue

            # Fallback: evaluate full segment (no holdout) for small segments
            best_seq = None
            best_score = -1.0
            for min_conf in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
                for max_odds in (8.0, 12.0, 18.0, 25.0, 40.0):
                    stats = self._evaluate_thresholds(seg_records, min_confidence=min_conf, max_win_odds=max_odds)
                    if stats["samples"] < max(8, min_samples):
                        continue
                    score = self._blend_score(stats, **(weights or {}))
                    if score > best_score:
                        best_score = score
                        best_seq = (min_conf, max_odds, stats)

            if best_seq is None:
                out["segments"][key] = {
                    "min_confidence": global_profile.min_confidence,
                    "max_win_odds": global_profile.max_win_odds,
                    "selected_samples": len(seg_records),
                    "selected_win_rate": 0.0,
                    "selected_roi": 0.0,
                    "selected_top3_rate": 0.0,
                    "selected_place_roi": 0.0,
                    "selected_score": 0.0,
                    "optimized_at": datetime.now().isoformat(),
                    "mode": "fallback_global",
                }
                continue

            c, o, stats = best_seq
            out["segments"][key] = {
                "min_confidence": c,
                "max_win_odds": o,
                "selected_samples": int(stats["samples"]),
                "selected_win_rate": round(stats["win_rate"], 4),
                "selected_roi": round(stats["roi"], 2),
                "selected_top3_rate": round(stats["top3_rate"], 4),
                "selected_place_roi": round(stats["place_roi"], 2),
                "selected_score": round(self._blend_score(stats, **(weights or {})), 4),
                "optimized_at": datetime.now().isoformat(),
                "mode": "segment_simple",
            }

        self.save_segmented(out)
        logger.info("Segmented strategy optimized: segments=%s", len(out.get("segments", {})))
        return out

    # ── Segmented Load/Save ─────────────────────────────────────────────────

    def load_segmented(self) -> Dict:
        if not os.path.exists(self.segment_strategy_file):
            return {"updated_at": "", "min_samples": 0, "segments": {}}
        try:
            with open(self.segment_strategy_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload.setdefault("segments", {})
                return payload
        except Exception as e:
            logger.warning("Failed to load segmented strategy profile: %s", e)
        return {"updated_at": "", "min_samples": 0, "segments": {}}

    def save_segmented(self, payload: Dict) -> None:
        with open(self.segment_strategy_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── Optimisation History ────────────────────────────────────────────────

    def _load_history(self) -> List[Dict]:
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_history(self, history: List[Dict]) -> None:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[-50:], f, ensure_ascii=False, indent=2)

    def _save_optimization_entry(self, profile: StrategyProfile) -> None:
        """Append an optimisation result to history (keep last 50)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "min_confidence": profile.min_confidence,
            "max_win_odds": profile.max_win_odds,
            "samples": profile.selected_samples,
            "win_rate": profile.selected_win_rate,
            "roi": profile.selected_roi,
            "top3_rate": profile.selected_top3_rate,
            "place_roi": profile.selected_place_roi,
            "score": profile.selected_score,
        }
        history = self._load_history()
        history.append(entry)
        self._save_history(history)

    def get_optimization_history(self, limit: int = 10) -> List[Dict]:
        """Return the most recent optimisation entries."""
        history = self._load_history()
        return history[-limit:]

    def print_optimization_summary(self, records: Optional[List[RacePredictionRecord]] = None) -> None:
        """Log current strategy profile and recent history."""
        profile = self.load()
        history = self.get_optimization_history()

        logger.info("=" * 60)
        logger.info("Current Strategy Profile")
        logger.info("=" * 60)
        logger.info(f"  Min Confidence:    {profile.min_confidence:.2f}")
        logger.info(f"  Max Win Odds:      {profile.max_win_odds:.1f}")
        logger.info(f"  Min Samples:       {profile.min_samples}")
        logger.info(f"  Weights:           win={profile.objective_weight_win_rate:.2f} "
                    f"roi={profile.objective_weight_roi:.2f} "
                    f"top3={profile.objective_weight_top3:.2f} "
                    f"place={profile.objective_weight_place_roi:.2f}")
        if profile.selected_samples > 0:
            logger.info(f"  Last Opt Results:  samples={profile.selected_samples} "
                        f"win_rate={profile.selected_win_rate:.2%} "
                        f"roi={profile.selected_roi:.1f}% "
                        f"top3={profile.selected_top3_rate:.2%} "
                        f"place_roi={profile.selected_place_roi:.1f}%")

        if history:
            logger.info("-" * 60)
            logger.info("Recent Optimisation History (last %d entries):", len(history))
            logger.info(f"{'#':<3} {'Date':<20} {'Conf':>5} {'Odds':>5} {'Samples':>7} "
                        f"{'Win%':>6} {'ROI%':>7} {'Top3%':>6} {'Place%':>7} {'Score':>6}")
            logger.info("-" * 70)
            for i, h in enumerate(history[::-1], 1):
                logger.info(
                    f"{i:<3} {h['timestamp'][:19]:<20} {h['min_confidence']:>5.2f} {h['max_win_odds']:>5.1f} "
                    f"{h['samples']:>7} {h['win_rate']*100:>5.1f}% {h['roi']:>6.1f}% "
                    f"{h['top3_rate']*100:>5.1f}% {h['place_roi']:>6.1f}% {h['score']:>6.4f}"
                )
        logger.info("=" * 60)

        if records:
            self.print_calibration_report(records)
