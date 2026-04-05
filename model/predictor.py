"""
HKJC 賽馬預測系統 - 預測模組
Generates top-3 predictions, value bets, Trio suggestions, and parlay picks.
"""
import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scraper.odds import calculate_implied_probability, calculate_expected_value, calculate_overlay
from features.builder import MODEL_FEATURE_COLS

logger = logging.getLogger(__name__)


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def _preferred_trad_name(primary: str, secondary: str) -> str:
    """Prefer Traditional-Chinese display name when available."""
    p = str(primary or "").strip()
    s = str(secondary or "").strip()
    if s and _has_cjk(s):
        return s
    if p and _has_cjk(p):
        return p
    return s or p


def _distance_bucket(distance: int) -> str:
    d = int(distance or 0)
    if d <= 1200:
        return "sprint"
    if d <= 1600:
        return "mile"
    if d <= 2000:
        return "middle"
    return "long"


@dataclass
class HorsePrediction:
    """Prediction details for a single horse."""
    horse_number: int
    horse_name: str
    horse_name_ch: str
    jockey: str
    trainer: str
    draw: int
    model_prob: float           # Ensemble model's estimated top-3 probability
    win_prob: float             # Estimated win probability (scaled from model_prob)
    implied_win_prob: float     # Market's implied win probability
    win_odds: float
    place_odds: float
    confidence: float           # Normalised confidence score [0, 1]
    rank: int                   # Rank within this race's predictions
    is_value_bet: bool          # True if model_prob > implied_prob * 1.1
    expected_value: float       # EV per $10 bet
    overlay_pct: float          # (model_prob - implied) / implied
    elo_rating: float
    recent_form_score: float
    draw_advantage: float


@dataclass
class PredictionResult:
    """Complete prediction result for a single race."""
    race_number: int
    race_date: str
    venue: str
    venue_code: str
    distance: int
    race_class: str
    going: str
    start_time: str

    # Ranked horses (all)
    ranked_horses: List[HorsePrediction] = field(default_factory=list)

    # Top N picks (default top 5)
    top5: List[HorsePrediction] = field(default_factory=list)

    # Value bets: model_prob > implied_prob * 1.1
    value_bets: List[HorsePrediction] = field(default_factory=list)

    # Trio suggestions: list of (horse1, horse2, horse3) tuples
    trio_suggestions: List[Tuple[int, int, int]] = field(default_factory=list)

    # Mixed parlay picks: 1-2 high confidence picks
    parlay_picks: List[HorsePrediction] = field(default_factory=list)

    # Aggregate confidence for this race
    confidence: float = 0.0

    # Best value bet (highest EV)
    best_ev_horse: Optional[HorsePrediction] = None

    # Recommendation selected by adaptive strategy
    recommended_winner: Optional[HorsePrediction] = None

    # Full feature snapshot captured at prediction time for incremental retraining
    feature_rows: List[Dict] = field(default_factory=list)


class RacePredictor:
    """
    Generates race predictions using the trained ensemble model.

    Uses the EnsembleTrainer's predict_proba method and enriches the output
    with market comparison, value bets, Trio combinations, and parlay picks.
    """

    def __init__(self, trainer=None, top_n: int = 5, value_edge_threshold: float = 0.10):
        """
        Initialise predictor.

        Args:
            trainer: EnsembleTrainer instance. If None, creates a new one.
            top_n: Number of top horses to include in predictions
            value_edge_threshold: Minimum overlay to qualify as a value bet (0.10 = 10%)
        """
        if trainer is None:
            from model.trainer import EnsembleTrainer
            trainer = EnsembleTrainer()
        self.trainer = trainer
        self.top_n = top_n
        self.value_edge_threshold = value_edge_threshold

        from config import config
        self.min_confidence_for_recommendation = 0.55
        self.max_win_odds_for_recommendation = 18.0
        self.segment_profiles: Dict[str, Dict] = {}
        if getattr(config, "ENABLE_SELF_OPTIMIZATION", True):
            try:
                from model.self_optimizer import StrategySelfOptimizer
                optimizer = StrategySelfOptimizer(
                    getattr(config, "STRATEGY_PROFILE_PATH", "data/models/strategy_profile.json")
                )
                profile = optimizer.load()
                self.min_confidence_for_recommendation = profile.min_confidence
                self.max_win_odds_for_recommendation = profile.max_win_odds
                segmented = optimizer.load_segmented()
                self.segment_profiles = dict(segmented.get("segments", {}) or {})
            except Exception as e:
                logger.warning("Failed to load adaptive strategy profile: %s", e)

    def predict_top3(
        self,
        features_df: pd.DataFrame,
        race_info: Optional[Dict] = None,
    ) -> PredictionResult:
        """
        Generate top-3 (and beyond) predictions for a race.

        Args:
            features_df: Feature DataFrame from builder.build_features()
                         Must contain metadata columns: horse_number, horse_name, etc.
            race_info: Optional dict with race metadata (race_number, venue, etc.)

        Returns:
            PredictionResult with ranked horses, value bets, Trio, and parlay picks
        """
        if features_df.empty:
            logger.warning("Empty features_df; returning empty prediction")
            return PredictionResult(
                race_number=race_info.get("race_number", 0) if race_info else 0,
                race_date="",
                venue="",
                venue_code="",
                distance=0,
                race_class="",
                going="",
                start_time="",
            )

        # Validate features_df for common issues
        expected_cols = ["horse_number", "win_odds", "implied_win_probability", "elo_rating"]
        missing_cols = [c for c in expected_cols if c not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        if features_df["horse_number"].duplicated().any():
            raise ValueError("Duplicate horse_number in features_df")

        if features_df.index.is_monotonic_increasing is False:
            logger.warning("features_df index not sorted; sorting by default")
            features_df = features_df.sort_index()

        if features_df.isna().any().any():
            logger.warning("features_df contains missing values\n%s", features_df.isna().sum())

        # Get model probabilities
        model_probs = self.trainer.predict_proba(features_df)

        # Normalise to sum=1 within race (softmax-like)
        model_probs = np.clip(model_probs, 1e-6, 1.0)
        model_probs = model_probs / model_probs.sum()

        # Scale to get approximate win probabilities (model outputs top-3 prob)
        # Rough heuristic: win prob ≈ model_prob / 3 * n_runners adjustment
        n_runners = len(features_df)
        win_probs = model_probs / 3.0
        win_probs = win_probs / win_probs.sum()  # re-normalise

        # Build HorsePrediction objects
        horses: List[HorsePrediction] = []
        max_model_prob = model_probs.max()

        # Capture a serializable snapshot of full model features at prediction time.
        feature_rows: List[Dict] = []
        for _, row in features_df.iterrows():
            horse_number = int(row.get("horse_number", 0) or 0)
            if horse_number <= 0:
                continue
            snapshot = {
                "horse_number": horse_number,
            }
            for col in MODEL_FEATURE_COLS:
                value = row.get(col, 0.0)
                try:
                    snapshot[col] = float(value)
                except (TypeError, ValueError):
                    snapshot[col] = 0.0
            feature_rows.append(snapshot)

        for i, (_, row) in enumerate(features_df.iterrows()):
            mp = model_probs[i]
            wp = win_probs[i]
            imp_win = row.get("implied_win_probability", 1.0 / n_runners)
            imp_place = row.get("implied_place_probability", 3.0 / n_runners)
            win_odds = row.get("win_odds", 10.0)
            place_odds = row.get("place_odds", 3.0)

            # Confidence normalised [0, 1] based on model_prob rank
            confidence = mp / max_model_prob if max_model_prob > 0 else 0.0

            # Value bet: model probability > implied * (1 + threshold)
            is_value = (wp > imp_win * (1 + self.value_edge_threshold)) if imp_win > 0 else False
            overlay = calculate_overlay(wp, imp_win)
            ev = calculate_expected_value(wp, win_odds, stake=10.0)

            horse_name = str(row.get("horse_name", f"Horse {i+1}"))
            horse_name_ch = str(row.get("horse_name_ch", ""))
            display_name = _preferred_trad_name(horse_name, horse_name_ch)

            horses.append(HorsePrediction(
                horse_number=int(row.get("horse_number", i + 1)),
                horse_name=display_name,
                horse_name_ch=display_name,
                jockey=str(row.get("jockey", "")),
                trainer=str(row.get("trainer", "")),
                draw=int(row.get("draw", 0)),
                model_prob=round(float(mp), 4),
                win_prob=round(float(wp), 4),
                implied_win_prob=round(float(imp_win), 4),
                win_odds=float(win_odds),
                place_odds=float(place_odds),
                confidence=round(float(confidence), 4),
                rank=0,  # will be set after sorting
                is_value_bet=is_value,
                expected_value=round(ev, 2),
                overlay_pct=round(overlay, 4),
                elo_rating=float(row.get("elo_rating", 1500)),
                recent_form_score=float(row.get("recent_form_score", 0.3)),
                draw_advantage=float(row.get("draw_advantage_score", 0.0)),
            ))

        # Rank by model_prob (descending)
        horses.sort(key=lambda h: h.model_prob, reverse=True)
        for rank, h in enumerate(horses, start=1):
            h.rank = rank

        top5 = horses[:self.top_n]
        value_bets = [h for h in horses if h.is_value_bet]
        value_bets.sort(key=lambda h: h.overlay_pct, reverse=True)

        # Trio suggestions: combinations of top-5 sorted by combined confidence
        trio_suggestions = _generate_trio_suggestions(top5)

        # Parlay: top 1-2 high confidence picks (confidence > 0.7)
        parlay_picks = [h for h in top5[:3] if h.confidence >= 0.65][:2]

        # Overall race confidence (entropy-based)
        confidence = _race_confidence(horses)

        # Best EV horse among value bets
        best_ev = max(value_bets, key=lambda h: h.expected_value) if value_bets else (
            top5[0] if top5 else None
        )

        # Support both Race dataclass and plain dict
        if hasattr(race_info, 'race_number'):
            ri_race_number = race_info.race_number
            ri_race_date = getattr(race_info, 'race_date', '')
            ri_venue = race_info.venue
            ri_venue_code = race_info.venue_code
            ri_distance = race_info.distance
            ri_race_class = race_info.race_class
            ri_going = race_info.going
            ri_start_time = race_info.start_time
        else:
            ri = race_info or {}
            ri_race_number = int(ri.get("race_number", 0))
            ri_race_date = str(ri.get("race_date", ""))
            ri_venue = str(ri.get("venue", ""))
            ri_venue_code = str(ri.get("venue_code", ""))
            ri_distance = int(ri.get("distance", 0))
            ri_race_class = str(ri.get("race_class", ""))
            ri_going = str(ri.get("going", ""))
            ri_start_time = str(ri.get("start_time", ""))

        # Recommendation policy optimized from real-data backtest history.
        min_conf = self.min_confidence_for_recommendation
        max_odds = self.max_win_odds_for_recommendation
        seg_key = f"{ri_venue_code}_{_distance_bucket(ri_distance)}" if ri_venue_code else ""
        seg = self.segment_profiles.get(seg_key)
        if isinstance(seg, dict):
            min_conf = float(seg.get("min_confidence", min_conf) or min_conf)
            max_odds = float(seg.get("max_win_odds", max_odds) or max_odds)

        eligible = [
            h for h in top5
            if h.confidence >= min_conf
            and h.win_odds <= max_odds
        ]
        recommended = eligible[0] if eligible else (top5[0] if top5 else None)

        return PredictionResult(
            race_number=ri_race_number,
            race_date=ri_race_date,
            venue=ri_venue,
            venue_code=ri_venue_code,
            distance=ri_distance,
            race_class=ri_race_class,
            going=ri_going,
            start_time=ri_start_time,
            ranked_horses=horses,
            top5=top5,
            value_bets=value_bets,
            trio_suggestions=trio_suggestions,
            parlay_picks=parlay_picks,
            confidence=confidence,
            best_ev_horse=best_ev,
            recommended_winner=recommended,
            feature_rows=feature_rows,
        )

    def predict_race_from_components(
        self,
        race,
        horse_profiles: Dict,
        odds: Dict,
        elo,
        jt_stats,
        draw_stats,
    ) -> PredictionResult:
        """
        Convenience method: build features and predict in one call.

        Args:
            race: Race object
            horse_profiles: Dict of horse_code -> HorseProfile
            odds: Odds dict
            elo: ELOSystem
            jt_stats: JockeyTrainerStats
            draw_stats: DrawStats

        Returns:
            PredictionResult
        """
        from features.builder import build_features

        features_df = build_features(race, horse_profiles, odds, elo, jt_stats, draw_stats)

        race_info = {
            "race_number": race.race_number,
            "race_date": race.race_date,
            "venue": race.venue,
            "venue_code": race.venue_code,
            "distance": race.distance,
            "race_class": race.race_class,
            "going": race.going,
            "start_time": race.start_time,
        }

        return self.predict_top3(features_df, race_info)


# ── Helper functions ───────────────────────────────────────────────────────────

def _generate_trio_suggestions(
    top_horses: List[HorsePrediction], n_suggestions: int = 5
) -> List[Tuple[int, int, int]]:
    """
    Generate Trio bet suggestions from top horses.

    HKJC Trio requires selecting 3 horses to finish 1st/2nd/3rd in any order.

    Args:
        top_horses: Ranked list of top horses
        n_suggestions: Maximum number of Trio combinations to return

    Returns:
        List of (horse_number, horse_number, horse_number) tuples
    """
    if len(top_horses) < 3:
        return []

    candidates = top_horses[:min(6, len(top_horses))]
    combos = list(combinations(candidates, 3))

    # Rank by combined model probability
    combos.sort(key=lambda c: sum(h.model_prob for h in c), reverse=True)

    return [
        (c[0].horse_number, c[1].horse_number, c[2].horse_number)
        for c in combos[:n_suggestions]
    ]


def _race_confidence(horses: List[HorsePrediction]) -> float:
    """
    Compute race-level prediction confidence using entropy.

    High confidence = model has a clear favourite.
    Low confidence = probabilities evenly spread.

    Returns:
        Confidence in [0, 1]
    """
    probs = [h.model_prob for h in horses]
    if not probs:
        return 0.0
    probs = np.array(probs)
    probs = probs / probs.sum()

    # Shannon entropy: lower entropy = higher confidence
    import math
    entropy = -sum(p * math.log(p + 1e-9) for p in probs)
    max_entropy = math.log(len(probs))  # maximum possible entropy

    if max_entropy == 0:
        return 1.0

    normalised_entropy = entropy / max_entropy
    return round(1.0 - normalised_entropy, 4)
