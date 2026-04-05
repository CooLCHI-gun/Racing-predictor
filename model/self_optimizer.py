"""
HKJC 賽馬預測系統 - 策略自我優化器
Tune recommendation thresholds using settled real-data backtest records.
"""
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from model.backtest import RacePredictionRecord

logger = logging.getLogger(__name__)


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
    objective_weight_win_rate: float = 0.55
    objective_weight_roi: float = 0.45
    selected_samples: int = 0
    selected_win_rate: float = 0.0
    selected_roi: float = 0.0
    selected_score: float = 0.0
    optimized_at: str = ""


class StrategySelfOptimizer:
    """Grid-search optimizer for win-rate/ROI balanced recommendation filters."""

    def __init__(self, strategy_file: str = "data/models/strategy_profile.json"):
        self.strategy_file = strategy_file
        os.makedirs(os.path.dirname(strategy_file), exist_ok=True)
        self.segment_strategy_file = os.path.join(
            os.path.dirname(strategy_file),
            "strategy_profile_segments.json",
        )

    def load(self) -> StrategyProfile:
        if not os.path.exists(self.strategy_file):
            return StrategyProfile()

        try:
            with open(self.strategy_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return StrategyProfile(**payload)
        except Exception as e:
            logger.warning("Failed to load strategy profile, using defaults: %s", e)
            return StrategyProfile()

    def save(self, profile: StrategyProfile) -> None:
        with open(self.strategy_file, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    def _optimize_records(self, records: List[RacePredictionRecord], min_samples: int) -> Optional[StrategyProfile]:
        """Return best profile for a record set without persisting to disk."""
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

        best_profile: Optional[StrategyProfile] = None
        best_score = -1.0

        for min_conf in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            for max_odds in (8.0, 12.0, 18.0, 25.0, 40.0):
                stats_calib = self._evaluate_thresholds(calib, min_confidence=min_conf, max_win_odds=max_odds)
                stats_valid = self._evaluate_thresholds(valid, min_confidence=min_conf, max_win_odds=max_odds)

                if stats_calib["samples"] < min_samples_calib:
                    continue
                if stats_valid["samples"] < min_samples_valid:
                    continue

                score_calib = self._blend_score(stats_calib["win_rate"], stats_calib["roi"])
                score_valid = self._blend_score(stats_valid["win_rate"], stats_valid["roi"])

                win_gap = abs(stats_calib["win_rate"] - stats_valid["win_rate"])
                roi_gap = abs(stats_calib["roi"] - stats_valid["roi"]) / 100.0
                stability_penalty = (0.65 * win_gap) + (0.35 * roi_gap)

                # Prefer parameters that hold up on recent races.
                score = (0.35 * score_calib) + (0.65 * score_valid) - (0.15 * stability_penalty)

                if score > best_score:
                    best_score = score
                    best_profile = StrategyProfile(
                        min_confidence=min_conf,
                        max_win_odds=max_odds,
                        min_samples=min_samples,
                        objective_weight_win_rate=0.55,
                        objective_weight_roi=0.45,
                        selected_samples=int(stats_valid["samples"]),
                        selected_win_rate=round(stats_valid["win_rate"], 4),
                        selected_roi=round(stats_valid["roi"], 2),
                        selected_score=round(score, 4),
                        optimized_at=datetime.now().isoformat(),
                    )

        return best_profile

    def optimize(self, records: List[RacePredictionRecord], min_samples: int = 24) -> StrategyProfile:
        """
        Search threshold combinations and persist best profile.
        Uses chronological holdout validation + stability penalty to reduce overfitting.
        """
        settled = [r for r in records if r.actual_winner > 0]
        if len(settled) < min_samples:
            logger.info("Skip optimization: only %s settled records (need >= %s)", len(settled), min_samples)
            return self.load()
        best_profile = self._optimize_records(settled, min_samples=min_samples)

        if best_profile is None:
            logger.info("Optimization found no candidate meeting minimum samples")
            return self.load()

        self.save(best_profile)
        logger.info(
            "Strategy optimized: min_conf=%.2f max_odds=%.1f samples=%s win_rate=%.2f%% roi=%.2f%% score=%.4f",
            best_profile.min_confidence,
            best_profile.max_win_odds,
            best_profile.selected_samples,
            best_profile.selected_win_rate * 100,
            best_profile.selected_roi,
            best_profile.selected_score,
        )
        return best_profile

    def optimize_segmented(
        self,
        records: List[RacePredictionRecord],
        min_samples: int = 12,
    ) -> Dict[str, Dict]:
        """
        Optimize strategy per segment: ST/HV x sprint/mile/middle/long.

        Returns saved profile payload.
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

        def _optimize_simple(segment_records: List[RacePredictionRecord]) -> Optional[Tuple[float, float, Dict[str, float]]]:
            best = None
            best_score = -1.0
            for c in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
                for o in (8.0, 12.0, 18.0, 25.0, 40.0):
                    stats = self._evaluate_thresholds(segment_records, min_confidence=c, max_win_odds=o)
                    if stats["samples"] < max(8, min_samples):
                        continue
                    score = self._blend_score(stats["win_rate"], stats["roi"])
                    if score > best_score:
                        best_score = score
                        best = (c, o, stats)
            return best

        for key in all_segment_keys:
            seg_records = list(by_segment.get(key, []))
            if len(seg_records) < max(8, min_samples):
                out["segments"][key] = {
                    "min_confidence": global_profile.min_confidence,
                    "max_win_odds": global_profile.max_win_odds,
                    "selected_samples": len(seg_records),
                    "selected_win_rate": 0.0,
                    "selected_roi": 0.0,
                    "selected_score": 0.0,
                    "optimized_at": datetime.now().isoformat(),
                    "mode": "fallback_global",
                }
                continue

            seg_profile = self._optimize_records(seg_records, min_samples=max(8, min_samples))
            if seg_profile is None:
                simple = _optimize_simple(seg_records)
                if simple is None:
                    out["segments"][key] = {
                        "min_confidence": global_profile.min_confidence,
                        "max_win_odds": global_profile.max_win_odds,
                        "selected_samples": len(seg_records),
                        "selected_win_rate": 0.0,
                        "selected_roi": 0.0,
                        "selected_score": 0.0,
                        "optimized_at": datetime.now().isoformat(),
                        "mode": "fallback_global",
                    }
                    continue

                c, o, stats = simple
                out["segments"][key] = {
                    "min_confidence": c,
                    "max_win_odds": o,
                    "selected_samples": int(stats["samples"]),
                    "selected_win_rate": round(stats["win_rate"], 4),
                    "selected_roi": round(stats["roi"], 2),
                    "selected_score": round(self._blend_score(stats["win_rate"], stats["roi"]), 4),
                    "optimized_at": datetime.now().isoformat(),
                    "mode": "segment_simple",
                }
                continue

            out["segments"][key] = {
                "min_confidence": seg_profile.min_confidence,
                "max_win_odds": seg_profile.max_win_odds,
                "selected_samples": seg_profile.selected_samples,
                "selected_win_rate": seg_profile.selected_win_rate,
                "selected_roi": seg_profile.selected_roi,
                "selected_score": seg_profile.selected_score,
                "optimized_at": seg_profile.optimized_at,
                "mode": "segment_validated",
            }

        self.save_segmented(out)
        logger.info("Segmented strategy optimized: segments=%s", len(out.get("segments", {})))
        return out

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

    @staticmethod
    def _evaluate_thresholds(
        records: List[RacePredictionRecord],
        min_confidence: float,
        max_win_odds: float,
        stake: float = 10.0,
    ) -> Dict[str, float]:
        selected = [
            r for r in records
            if r.model_confidence >= min_confidence and 0 < r.predicted_winner_odds <= max_win_odds
        ]

        if not selected:
            return {"samples": 0, "win_rate": 0.0, "roi": 0.0}

        wins = sum(1 for r in selected if r.winner_correct)
        win_rate = wins / len(selected)
        total_invested = len(selected) * stake
        total_returned = sum(r.win_bet_return for r in selected)
        roi = ((total_returned - total_invested) / max(total_invested, 1.0)) * 100.0

        return {
            "samples": len(selected),
            "win_rate": win_rate,
            "roi": roi,
        }

    @staticmethod
    def _blend_score(win_rate: float, roi: float, w_win: float = 0.55, w_roi: float = 0.45) -> float:
        # Clamp ROI to a stable optimization range to avoid overfitting to outliers.
        roi_norm = max(0.0, min(1.0, (roi + 30.0) / 60.0))
        return (w_win * win_rate) + (w_roi * roi_norm)
