"""
HKJC 賽馬預測系統 - 策略自我優化器
Tune recommendation thresholds using settled real-data backtest records.
"""
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

from model.backtest import RacePredictionRecord

logger = logging.getLogger(__name__)


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

    def optimize(self, records: List[RacePredictionRecord], min_samples: int = 24) -> StrategyProfile:
        """
        Search threshold combinations and persist best profile.
        Uses a blended objective: score = w1 * win_rate + w2 * roi_norm.
        """
        settled = [r for r in records if r.actual_winner > 0]
        if len(settled) < min_samples:
            logger.info("Skip optimization: only %s settled records (need >= %s)", len(settled), min_samples)
            return self.load()

        best_profile: Optional[StrategyProfile] = None
        best_score = -1.0

        for min_conf in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            for max_odds in (8.0, 12.0, 18.0, 25.0, 40.0):
                stats = self._evaluate_thresholds(settled, min_confidence=min_conf, max_win_odds=max_odds)
                if stats["samples"] < min_samples:
                    continue

                score = self._blend_score(stats["win_rate"], stats["roi"])
                if score > best_score:
                    best_score = score
                    best_profile = StrategyProfile(
                        min_confidence=min_conf,
                        max_win_odds=max_odds,
                        min_samples=min_samples,
                        objective_weight_win_rate=0.55,
                        objective_weight_roi=0.45,
                        selected_samples=int(stats["samples"]),
                        selected_win_rate=round(stats["win_rate"], 4),
                        selected_roi=round(stats["roi"], 2),
                        selected_score=round(score, 4),
                        optimized_at=datetime.now().isoformat(),
                    )

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
