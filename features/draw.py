"""
HKJC 賽馬預測系統 - 檔位統計模組
Draw (gate/barrier) statistics by distance, venue, and track configuration.

In Hong Kong racing, draw (gate) position has a significant impact on outcomes,
particularly at short distances and certain track configurations.
"""
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"

# Known draw biases for HKJC tracks (based on historical analysis)
# Key: (venue, distance) -> {draw_position: bias_score}
# bias_score: positive = advantage, negative = disadvantage
KNOWN_BIASES = {
    # Happy Valley 1000m - low draws (inside) strongly advantaged
    ("HV", 1000): {i: (6 - i) * 0.12 for i in range(1, 13)},
    # Happy Valley 1200m - slight low draw advantage
    ("HV", 1200): {i: max(-0.3, (5 - i) * 0.06) for i in range(1, 13)},
    # Happy Valley 1650m - more neutral
    ("HV", 1650): {i: (7 - i) * 0.02 for i in range(1, 13)},
    # Sha Tin 1000m - low draws very important (short run to first corner)
    ("ST", 1000): {i: (7 - i) * 0.09 for i in range(1, 15)},
    # Sha Tin 1200m - low to mid draws preferred
    ("ST", 1200): {i: max(-0.2, (6 - i) * 0.05) for i in range(1, 15)},
    # Sha Tin 1400m - more neutral, slight inner advantage
    ("ST", 1400): {i: max(-0.15, (7 - i) * 0.03) for i in range(1, 15)},
    # Sha Tin 1600m - wide gate can be neutral or slight disadvantage
    ("ST", 1600): {i: max(-0.1, (8 - i) * 0.02) for i in range(1, 15)},
    # Sha Tin 1800m - more neutral
    ("ST", 1800): {i: (8 - i) * 0.01 for i in range(1, 15)},
    # Sha Tin 2000m - high draws can be slight advantage (rails run)
    ("ST", 2000): {i: (i - 8) * 0.01 for i in range(1, 15)},
    # Sha Tin 2400m - draw less important in marathon
    ("ST", 2400): {i: (i - 8) * 0.005 for i in range(1, 15)},
}


class DrawStats:
    """
    Tracks and computes draw (gate) statistics.

    Maintains win/place rates by draw position for each venue/distance combination.
    Returns a normalised draw advantage score in [-1.0, 1.0].
    """

    def __init__(self, data_dir: str = PROCESSED_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Stats: {(venue, distance, draw): {"wins": int, "places": int, "rides": int}}
        self._stats: Dict[str, Dict] = {}

    def _key(self, venue: str, distance: int, draw: int) -> str:
        """Generate storage key."""
        dist_bucket = round(distance / 100) * 100  # 100m buckets
        return f"{venue}_{dist_bucket}_{draw}"

    def record_result(self, venue: str, distance: int, draw: int,
                      finish_position: int, runners: int) -> None:
        """
        Record a race result for draw statistics.

        Args:
            venue: "ST" | "HV"
            distance: Race distance in metres
            draw: Gate number
            finish_position: Finish position (1 = first)
            runners: Total runners
        """
        key = self._key(venue, distance, draw)
        if key not in self._stats:
            self._stats[key] = {"wins": 0, "places": 0, "rides": 0}
        self._stats[key]["rides"] += 1
        if finish_position == 1:
            self._stats[key]["wins"] += 1
        if finish_position <= 3:
            self._stats[key]["places"] += 1

    def get_draw_win_rate(self, venue: str, distance: int, draw: int) -> float:
        """Get win rate for a specific draw position."""
        key = self._key(venue, distance, draw)
        entry = self._stats.get(key)
        if not entry or entry["rides"] == 0:
            return _theoretical_win_rate(draw, venue, distance)
        return entry["wins"] / entry["rides"]

    def get_draw_place_rate(self, venue: str, distance: int, draw: int) -> float:
        """Get place (top-3) rate for a specific draw position."""
        key = self._key(venue, distance, draw)
        entry = self._stats.get(key)
        if not entry or entry["rides"] == 0:
            return _theoretical_place_rate(draw, venue, distance)
        return entry["places"] / entry["rides"]

    def get_draw_advantage_score(
        self, venue: str, distance: int, draw: int, n_runners: int = 12
    ) -> float:
        """
        Calculate draw advantage score normalised to [-1.0, 1.0].

        Combines statistical win rate and theoretical bias.

        Args:
            venue: "ST" | "HV"
            distance: Race distance in metres
            draw: Gate number
            n_runners: Number of runners in the race

        Returns:
            Score in [-1.0, 1.0]. Positive = advantage, negative = disadvantage.
        """
        # Get all draw positions in this race
        all_wins = []
        for d in range(1, n_runners + 1):
            all_wins.append(self.get_draw_win_rate(venue, distance, d))

        if not all_wins or all(w == 0 for w in all_wins):
            return 0.0

        # Target draw win rate
        target_wr = self.get_draw_win_rate(venue, distance, draw)

        # Expected win rate if no draw bias = 1/n_runners
        expected_wr = 1.0 / max(n_runners, 1)

        # Normalise: deviation from expected, scaled by field variance
        max_wr = max(all_wins)
        min_wr = min(all_wins)
        spread = max_wr - min_wr

        if spread < 1e-6:
            return 0.0

        # Score: where does this draw fall in the distribution?
        score = (target_wr - expected_wr) / (spread / 2)
        score = max(-1.0, min(1.0, score))
        return round(score, 4)

    def save(self, filename: str = "draw_stats.json") -> None:
        """Save statistics to JSON."""
        path = os.path.join(self.data_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved draw stats to {path}")

    def load(self, filename: str = "draw_stats.json") -> bool:
        """Load statistics from JSON."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            logger.info("Draw stats not found; using theoretical biases")
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._stats = json.load(f)
            logger.info(f"Loaded draw stats from {path}")
            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load draw stats: {e}")
            return False

    def bootstrap_from_mock(self) -> None:
        """Bootstrap stats from simulated historical data."""
        rng = random.Random(999)
        venues = ["ST", "HV"]
        distances = {
            "ST": [1000, 1200, 1400, 1600, 1800, 2000, 2400],
            "HV": [1000, 1200, 1650, 2200],
        }
        for venue in venues:
            for distance in distances[venue]:
                n_runners = 14 if venue == "ST" else 12
                for race_sim in range(50):  # 50 simulated races per config
                    # Simulate finish using known biases
                    probs = []
                    for draw in range(1, n_runners + 1):
                        base_prob = 1.0 / n_runners
                        bias = _get_known_bias(venue, distance, draw)
                        prob = max(0.01, base_prob * (1 + bias))
                        probs.append(prob)
                    total = sum(probs)
                    probs = [p / total for p in probs]

                    # Generate a result
                    draws = list(range(1, n_runners + 1))
                    finishing_order = rng.choices(
                        draws, weights=probs, k=min(3, n_runners)
                    )
                    used = set(finishing_order)
                    rest = [d for d in draws if d not in used]
                    rng.shuffle(rest)
                    full_result = finishing_order + rest

                    for pos, draw in enumerate(full_result, start=1):
                        self.record_result(venue, distance, draw, pos, n_runners)


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_known_bias(venue: str, distance: int, draw: int) -> float:
    """Look up known bias from KNOWN_BIASES dict (with distance rounding)."""
    # Try exact match first
    for (v, d), biases in KNOWN_BIASES.items():
        if v == venue and abs(d - distance) <= 100:
            return biases.get(draw, 0.0)
    return 0.0


def _theoretical_win_rate(draw: int, venue: str, distance: int) -> float:
    """
    Compute theoretical win rate from known biases.

    For positions without historical data, we use the known bias tables.
    """
    base = 1.0 / 12  # assume 12 runners
    bias = _get_known_bias(venue, distance, draw)
    return max(0.01, base * (1.0 + bias))


def _theoretical_place_rate(draw: int, venue: str, distance: int) -> float:
    """Theoretical top-3 rate (~3x win rate for equal-field, adjusted for bias)."""
    return min(0.95, _theoretical_win_rate(draw, venue, distance) * 3.2)
