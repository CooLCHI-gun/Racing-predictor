"""
HKJC 賽馬預測系統 - ELO 評分系統
ELO rating system for horses.

The ELO system tracks the relative strength of each horse.
After each race, ratings are updated based on finishing order vs expected order.
"""
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ELOSystem:
    """
    ELO rating system adapted for horse racing.

    In horse racing, we treat each race as a series of pairwise comparisons:
    every horse that finishes ahead of another "beats" that horse.
    Ratings are updated using the standard ELO formula across all pairs.
    """

    def __init__(self, k_factor: int = 32, base_rating: int = 1500):
        """
        Initialise ELO system.

        Args:
            k_factor: Sensitivity of rating updates (higher = more volatile)
            base_rating: Default rating for new horses
        """
        self.k_factor = k_factor
        self.base_rating = base_rating
        self._ratings: Dict[str, float] = {}      # horse_code -> rating
        self._history: Dict[str, List[float]] = {}  # horse_code -> last N ratings

    def get_rating(self, horse_code: str) -> float:
        """
        Get current ELO rating for a horse.

        Args:
            horse_code: HKJC horse code

        Returns:
            ELO rating (default base_rating for new horses)
        """
        return self._ratings.get(horse_code, float(self.base_rating))

    def get_all_ratings(self) -> Dict[str, float]:
        """Return copy of all ratings."""
        return dict(self._ratings)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score (win probability) of A against B.

        Args:
            rating_a: Rating of horse A
            rating_b: Rating of horse B

        Returns:
            Expected score in [0, 1]
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, race_result: List[str]) -> Dict[str, float]:
        """
        Update ELO ratings after a race.

        Uses a pairwise comparison approach: for every pair of horses,
        the one that finished ahead gets a win credit.

        Args:
            race_result: List of horse_codes in finishing order (1st, 2nd, 3rd...)
                         Retired/disqualified horses are placed at the end.

        Returns:
            Dict of horse_code -> new_rating for all participants
        """
        n = len(race_result)
        if n < 2:
            return {}

        # Get current ratings
        current = {code: self.get_rating(code) for code in race_result}

        # Accumulate deltas via pairwise comparisons
        deltas: Dict[str, float] = {code: 0.0 for code in race_result}

        for i in range(n):
            for j in range(i + 1, n):
                winner_code = race_result[i]   # finished ahead
                loser_code = race_result[j]

                r_w = current[winner_code]
                r_l = current[loser_code]

                # Expected scores
                e_w = self.expected_score(r_w, r_l)
                e_l = 1.0 - e_w

                # Actual scores: winner = 1, loser = 0
                delta_w = self.k_factor * (1.0 - e_w)
                delta_l = self.k_factor * (0.0 - e_l)

                deltas[winner_code] += delta_w
                deltas[loser_code] += delta_l

        # Normalise by number of comparisons each horse participated in
        # Each horse competes against (n-1) others
        comparisons_per_horse = n - 1

        new_ratings = {}
        for code in race_result:
            normalised_delta = deltas[code] / comparisons_per_horse
            new_rating = current[code] + normalised_delta
            new_rating = max(500.0, min(3000.0, new_rating))  # clamp to sane range
            self._ratings[code] = new_rating
            new_ratings[code] = new_rating

            # Store history
            if code not in self._history:
                self._history[code] = []
            self._history[code].append(new_rating)
            # Keep last 20 ratings
            self._history[code] = self._history[code][-20:]

        logger.debug(f"Updated ELO for {n} horses. Deltas: {deltas}")
        return new_ratings

    def get_rating_delta(self, horse_code: str, last_n: int = 3) -> float:
        """
        Calculate rating trend over last N races.

        Args:
            horse_code: HKJC horse code
            last_n: Number of recent races to consider

        Returns:
            Average rating change per race (positive = improving)
        """
        history = self._history.get(horse_code, [])
        if len(history) < 2:
            return 0.0
        recent = history[-min(last_n + 1, len(history)):]
        if len(recent) < 2:
            return 0.0
        delta = (recent[-1] - recent[0]) / (len(recent) - 1)
        return round(delta, 2)

    def save_ratings(self, path: str) -> None:
        """
        Persist ratings to JSON file.

        Args:
            path: File path (directory must exist)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "ratings": self._ratings,
            "history": self._history,
            "k_factor": self.k_factor,
            "base_rating": self.base_rating,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"ELO ratings saved to {path} ({len(self._ratings)} horses)")

    def load_ratings(self, path: str) -> bool:
        """
        Load ratings from JSON file.

        Args:
            path: File path

        Returns:
            True if loaded successfully, False if file not found or invalid
        """
        if not os.path.exists(path):
            logger.info(f"ELO ratings file not found at {path}; starting fresh")
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ratings = {k: float(v) for k, v in data.get("ratings", {}).items()}
            self._history = data.get("history", {})
            self.k_factor = data.get("k_factor", self.k_factor)
            self.base_rating = data.get("base_rating", self.base_rating)
            logger.info(f"Loaded ELO ratings for {len(self._ratings)} horses from {path}")
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to load ELO ratings from {path}: {e}")
            return False

    def initialise_from_history(self, race_history: List[Dict]) -> None:
        """
        Bootstrap ELO ratings from historical race results.

        Args:
            race_history: List of dicts with keys:
                {"race_id": str, "results": ["horse_code_1st", "horse_code_2nd", ...]}
        """
        logger.info(f"Bootstrapping ELO from {len(race_history)} historical races...")
        for race in race_history:
            results = race.get("results", [])
            if results:
                self.update_ratings(results)
        logger.info(f"ELO bootstrap complete. {len(self._ratings)} horses rated.")

    def get_field_average_rating(self, horse_codes: List[str]) -> float:
        """
        Get the average ELO rating of a field.

        Args:
            horse_codes: List of horse codes in the race

        Returns:
            Average ELO rating
        """
        if not horse_codes:
            return float(self.base_rating)
        ratings = [self.get_rating(c) for c in horse_codes]
        return sum(ratings) / len(ratings)

    def get_relative_rating(self, horse_code: str, field_codes: List[str]) -> float:
        """
        Get a horse's rating relative to the field average.

        Args:
            horse_code: Target horse
            field_codes: All horses in the race

        Returns:
            Rating difference from field average
        """
        return self.get_rating(horse_code) - self.get_field_average_rating(field_codes)
