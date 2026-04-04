"""
HKJC 賽馬預測系統 - 騎師及練馬師統計模組
Jockey and trainer statistics tracking.

Tracks win rates, place rates, and combination statistics
across different time windows, distances, and venues.
"""
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"


@dataclass
class RideRecord:
    """A single ride record for statistics tracking."""
    date: str           # "YYYY-MM-DD"
    jockey: str
    jockey_code: str
    trainer: str
    trainer_code: str
    venue: str          # "ST" | "HV"
    distance: int
    track_type: str
    going: str
    finish_position: int
    runners: int
    race_class: str
    horse_code: str


@dataclass
class StrikeRate:
    """Win/place strike rate statistics."""
    wins: int = 0
    places: int = 0    # Top-3
    rides: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.rides if self.rides > 0 else 0.0

    @property
    def place_rate(self) -> float:
        return self.places / self.rides if self.rides > 0 else 0.0


class JockeyTrainerStats:
    """
    Tracks and computes jockey, trainer, and combination statistics.

    Statistics are computed for:
    - Overall season
    - Last 30 days
    - By venue
    - By distance range
    - Jockey-Trainer combinations
    """

    def __init__(self, data_dir: str = PROCESSED_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Raw records
        self._records: List[RideRecord] = []

        # Cached stats (rebuilt on load/update)
        self._jockey_stats: Dict[str, Dict] = {}
        self._trainer_stats: Dict[str, Dict] = {}
        self._combo_stats: Dict[str, Dict] = {}

    def add_record(self, record: RideRecord) -> None:
        """Add a ride record."""
        self._records.append(record)

    def add_records(self, records: List[RideRecord]) -> None:
        """Add multiple ride records."""
        self._records.extend(records)
        self._rebuild_stats()

    def _rebuild_stats(self) -> None:
        """Rebuild all cached statistics from raw records."""
        today = date.today()
        last_30 = today - timedelta(days=30)
        last_30_str = last_30.strftime("%Y-%m-%d")

        # ── Jockey stats ──────────────────────────────────────────────────
        jk: Dict[str, Dict] = defaultdict(lambda: {
            "overall": StrikeRate(),
            "last_30d": StrikeRate(),
            "by_venue": defaultdict(StrikeRate),
            "by_distance": defaultdict(StrikeRate),
        })

        # ── Trainer stats ─────────────────────────────────────────────────
        tr: Dict[str, Dict] = defaultdict(lambda: {
            "overall": StrikeRate(),
            "last_30d": StrikeRate(),
            "by_venue": defaultdict(StrikeRate),
            "by_distance": defaultdict(StrikeRate),
        })

        # ── Combo stats ───────────────────────────────────────────────────
        cb: Dict[str, StrikeRate] = defaultdict(StrikeRate)

        for rec in self._records:
            is_win = rec.finish_position == 1
            is_place = rec.finish_position <= 3
            is_recent = rec.date >= last_30_str
            dist_bucket = str(round(rec.distance / 200) * 200)

            # Jockey
            j_stats = jk[rec.jockey_code]
            _update_strike(j_stats["overall"], is_win, is_place)
            if is_recent:
                _update_strike(j_stats["last_30d"], is_win, is_place)
            _update_strike(j_stats["by_venue"][rec.venue], is_win, is_place)
            _update_strike(j_stats["by_distance"][dist_bucket], is_win, is_place)

            # Trainer
            t_stats = tr[rec.trainer_code]
            _update_strike(t_stats["overall"], is_win, is_place)
            if is_recent:
                _update_strike(t_stats["last_30d"], is_win, is_place)
            _update_strike(t_stats["by_venue"][rec.venue], is_win, is_place)
            _update_strike(t_stats["by_distance"][dist_bucket], is_win, is_place)

            # Combo
            combo_key = f"{rec.jockey_code}_{rec.trainer_code}"
            _update_strike(cb[combo_key], is_win, is_place)

        self._jockey_stats = {k: _serialise_stats(v) for k, v in jk.items()}
        self._trainer_stats = {k: _serialise_stats(v) for k, v in tr.items()}
        self._combo_stats = {k: _serialise_strike(v) for k, v in cb.items()}

    def get_jockey_win_rate(self, jockey_code: str, period: str = "last_30d") -> float:
        """
        Get jockey win rate for a given period.

        Args:
            jockey_code: Jockey code
            period: "overall" | "last_30d"

        Returns:
            Win rate [0, 1]
        """
        stats = self._jockey_stats.get(jockey_code, {})
        return stats.get(period, {}).get("win_rate", 0.0)

    def get_jockey_place_rate(self, jockey_code: str, period: str = "last_30d") -> float:
        """Get jockey top-3 place rate."""
        stats = self._jockey_stats.get(jockey_code, {})
        return stats.get(period, {}).get("place_rate", 0.0)

    def get_trainer_win_rate(self, trainer_code: str, period: str = "last_30d") -> float:
        """Get trainer win rate."""
        stats = self._trainer_stats.get(trainer_code, {})
        return stats.get(period, {}).get("win_rate", 0.0)

    def get_trainer_place_rate(self, trainer_code: str, period: str = "last_30d") -> float:
        """Get trainer top-3 place rate."""
        stats = self._trainer_stats.get(trainer_code, {})
        return stats.get(period, {}).get("place_rate", 0.0)

    def get_combo_win_rate(self, jockey_code: str, trainer_code: str) -> float:
        """Get jockey-trainer combination win rate."""
        key = f"{jockey_code}_{trainer_code}"
        return self._combo_stats.get(key, {}).get("win_rate", 0.0)

    def get_combo_place_rate(self, jockey_code: str, trainer_code: str) -> float:
        """Get jockey-trainer combination place rate."""
        key = f"{jockey_code}_{trainer_code}"
        return self._combo_stats.get(key, {}).get("place_rate", 0.0)

    def get_jockey_venue_win_rate(self, jockey_code: str, venue: str) -> float:
        """Get jockey win rate at specific venue."""
        stats = self._jockey_stats.get(jockey_code, {})
        return stats.get("by_venue", {}).get(venue, {}).get("win_rate", 0.0)

    def get_jockey_distance_win_rate(self, jockey_code: str, distance: int) -> float:
        """Get jockey win rate at specific distance (±200m bucket)."""
        bucket = str(round(distance / 200) * 200)
        stats = self._jockey_stats.get(jockey_code, {})
        return stats.get("by_distance", {}).get(bucket, {}).get("win_rate", 0.0)

    def save(self, filename: str = "jockey_trainer_stats.json") -> None:
        """Save statistics to JSON file."""
        path = os.path.join(self.data_dir, filename)
        data = {
            "jockey": self._jockey_stats,
            "trainer": self._trainer_stats,
            "combo": self._combo_stats,
            "records_count": len(self._records),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved jockey/trainer stats to {path}")

    def load(self, filename: str = "jockey_trainer_stats.json") -> bool:
        """Load statistics from JSON file."""
        from config import config

        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            if config.REAL_DATA_ONLY and not config.DEMO_MODE:
                logger.error(f"Stats file not found at {path}; REAL_DATA_ONLY=True so mock stats are disabled")
                self._jockey_stats = {}
                self._trainer_stats = {}
                self._combo_stats = {}
                return False
            logger.info(f"Stats file not found at {path}; using mock data")
            self._load_mock_stats()
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._jockey_stats = data.get("jockey", {})
            self._trainer_stats = data.get("trainer", {})
            self._combo_stats = data.get("combo", {})
            logger.info(f"Loaded stats from {path}")
            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load stats: {e}")
            if config.REAL_DATA_ONLY and not config.DEMO_MODE:
                self._jockey_stats = {}
                self._trainer_stats = {}
                self._combo_stats = {}
                return False
            self._load_mock_stats()
            return False

    def _load_mock_stats(self) -> None:
        """Load mock statistics for offline/demo mode."""
        mock_data = generate_mock_stats()
        self._jockey_stats = mock_data["jockey"]
        self._trainer_stats = mock_data["trainer"]
        self._combo_stats = mock_data["combo"]
        logger.info("Loaded mock jockey/trainer stats")


# ── Helper functions ──────────────────────────────────────────────────────────

def _update_strike(sr: StrikeRate, is_win: bool, is_place: bool) -> None:
    """Mutate a StrikeRate in place."""
    sr.rides += 1
    if is_win:
        sr.wins += 1
    if is_place:
        sr.places += 1


def _serialise_strike(sr: StrikeRate) -> Dict:
    return {
        "wins": sr.wins,
        "places": sr.places,
        "rides": sr.rides,
        "win_rate": round(sr.win_rate, 4),
        "place_rate": round(sr.place_rate, 4),
    }


def _serialise_stats(stats: Dict) -> Dict:
    result = {}
    for k, v in stats.items():
        if isinstance(v, StrikeRate):
            result[k] = _serialise_strike(v)
        elif isinstance(v, defaultdict):
            result[k] = {sub_k: _serialise_strike(sub_v) for sub_k, sub_v in v.items()}
        else:
            result[k] = v
    return result


def generate_mock_stats() -> Dict:
    """
    Generate mock jockey and trainer statistics.

    Returns:
        Dict with "jockey", "trainer", "combo" keys
    """
    rng = random.Random(42)

    jockeys = {
        "ZP": ("Z Purton", 0.28, 0.52),
        "JM": ("J Moreira", 0.26, 0.50),
        "VB": ("V Borges", 0.20, 0.45),
        "KL": ("K Leung", 0.15, 0.38),
        "MC": ("M Chadwick", 0.18, 0.42),
        "BP": ("B Prebble", 0.16, 0.40),
        "CYH": ("C Y Ho", 0.14, 0.36),
        "HTM": ("H T Mo", 0.12, 0.33),
        "NC": ("N Callan", 0.13, 0.35),
        "VD": ("V Duric", 0.11, 0.30),
        "DW2": ("D Whyte", 0.17, 0.39),
        "AH": ("A Hamelin", 0.12, 0.32),
    }

    trainers = {
        "FCL": ("F C Lor", 0.22, 0.48),
        "DW": ("D Whyte", 0.20, 0.45),
        "CF": ("C Fownes", 0.18, 0.42),
        "JS": ("J Size", 0.19, 0.44),
        "RG": ("R Gibson", 0.16, 0.38),
        "ATM": ("A T Millard", 0.15, 0.37),
        "PFY": ("P F Yiu", 0.14, 0.35),
        "KWL": ("K W Lui", 0.13, 0.33),
        "LH": ("L Ho", 0.12, 0.30),
        "PO": ("P O'Sullivan", 0.17, 0.40),
        "DH": ("D Hall", 0.15, 0.36),
        "CHY": ("C H Yip", 0.11, 0.28),
    }

    def _make_rate_stats(code: str, base_win: float, base_place: float, rides_base: int = 80) -> Dict:
        rides = rng.randint(rides_base, rides_base * 2)
        rides_30d = rng.randint(10, 40)
        wins = int(rides * base_win * rng.uniform(0.8, 1.2))
        places = int(rides * base_place * rng.uniform(0.8, 1.2))
        wins_30d = int(rides_30d * base_win * rng.uniform(0.8, 1.2))
        places_30d = int(rides_30d * base_place * rng.uniform(0.8, 1.2))

        venues = ["ST", "HV"]
        distances = ["1000", "1200", "1400", "1600", "1800", "2000"]

        return {
            "overall": {
                "wins": wins, "places": places, "rides": rides,
                "win_rate": round(wins / max(rides, 1), 4),
                "place_rate": round(places / max(rides, 1), 4),
            },
            "last_30d": {
                "wins": wins_30d, "places": places_30d, "rides": rides_30d,
                "win_rate": round(wins_30d / max(rides_30d, 1), 4),
                "place_rate": round(places_30d / max(rides_30d, 1), 4),
            },
            "by_venue": {
                v: {
                    "wins": int(rides * 0.5 * base_win * rng.uniform(0.7, 1.3)),
                    "places": int(rides * 0.5 * base_place * rng.uniform(0.7, 1.3)),
                    "rides": int(rides * 0.5),
                    "win_rate": round(base_win * rng.uniform(0.7, 1.3), 4),
                    "place_rate": round(base_place * rng.uniform(0.7, 1.3), 4),
                }
                for v in venues
            },
            "by_distance": {
                d: {
                    "wins": rng.randint(0, 15),
                    "places": rng.randint(0, 25),
                    "rides": rng.randint(10, 50),
                    "win_rate": round(base_win * rng.uniform(0.6, 1.4), 4),
                    "place_rate": round(base_place * rng.uniform(0.6, 1.4), 4),
                }
                for d in distances
            },
        }

    jockey_stats = {
        code: _make_rate_stats(code, base_win, base_place)
        for code, (name, base_win, base_place) in jockeys.items()
    }

    trainer_stats = {
        code: _make_rate_stats(code, base_win, base_place, rides_base=100)
        for code, (name, base_win, base_place) in trainers.items()
    }

    # Combos
    combo_stats = {}
    for jk_code, (_, jk_win, jk_place) in jockeys.items():
        for tr_code, (_, tr_win, tr_place) in trainers.items():
            combo_win = min(jk_win, tr_win) * rng.uniform(0.9, 1.3)
            combo_place = min(jk_place, tr_place) * rng.uniform(0.9, 1.2)
            rides = rng.randint(2, 30)
            combo_stats[f"{jk_code}_{tr_code}"] = {
                "wins": int(rides * combo_win),
                "places": int(rides * combo_place),
                "rides": rides,
                "win_rate": round(combo_win, 4),
                "place_rate": round(combo_place, 4),
            }

    return {"jockey": jockey_stats, "trainer": trainer_stats, "combo": combo_stats}
