"""
HKJC 賽馬預測系統 - 系統設定
All configuration in one place.
"""
import os
from dataclasses import dataclass, field
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Config:
    # ── HKJC URLs (confirmed working 2026-04-02) ──────────────────────────
    HKJC_BASE_URL: str = "https://racing.hkjc.com"
    # Racecard: ?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=1
    RACECARD_URL: str = "https://racing.hkjc.com/en-us/local/information/racecard"
    # Horse profile: ?horseid=HK_2025_L126&Option=1
    HORSE_PROFILE_URL: str = "https://racing.hkjc.com/en-us/local/information/horse"
    # Results: ?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=1
    RESULTS_URL: str = "https://racing.hkjc.com/en-us/local/racing/results"
    # Odds: JS-rendered, need Playwright → bet.hkjc.com/en/racing/wp/{date}/{venue}/{race_no}
    ODDS_WP_URL: str = "https://bet.hkjc.com/en/racing/wp"
    ODDS_TRIO_URL: str = "https://bet.hkjc.com/en/racing/trio"
    # Going/weather
    WINDTRACKER_URL: str = "https://racing.hkjc.com/en-us/local/info/windtracker"
    HKO_WEATHER_URL: str = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"

    # ── Telegram ────────────────────────────────────────────────────────────
    TELEGRAM_TOKEN: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # ── Model ───────────────────────────────────────────────────────────────
    MODEL_PATH: str = "data/models/"
    TOP_N: int = 5          # Top candidates shown per race
    ENSEMBLE_WEIGHTS: tuple = (0.5, 0.5)  # (XGBoost, LightGBM)

    # ── Web ─────────────────────────────────────────────────────────────────
    FLASK_PORT: int = field(default_factory=lambda: _env_int("PORT", _env_int("FLASK_PORT", 5000)))
    FLASK_DEBUG: bool = field(default_factory=lambda: _env_bool("FLASK_DEBUG", False))
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "hkjc-predictor-dev-secret"))

    # ── Scheduler (HKT = UTC+8) ─────────────────────────────────────────────
    RACE_DAY_FETCH_TIME: str = "09:00"    # Fetch racecard at 9am HKT
    ODDS_FETCH_INTERVAL_MINS: int = 5     # Fetch odds every 5 min
    PRE_RACE_NOTIFY_MINS: int = 30        # Notify 30 min before each race
    PRE_RACE_REFRESH_MINS: int = 120      # Refresh and snapshot prediction 2h before race
    RESULTS_FETCH_DELAY_MINS: int = 30    # Fetch results 30 min after last race
    TIMEZONE: str = "Asia/Hong_Kong"

    # ── Feature settings ────────────────────────────────────────────────────
    FORM_RACES: int = 6       # Look back N races for form
    ELO_K_FACTOR: int = 32
    ELO_BASE: int = 1500
    JOCKEY_LOOKBACK_DAYS: int = 30
    TRAINER_LOOKBACK_DAYS: int = 30
    DISTANCE_TOLERANCE_M: int = 200       # ±200m for distance preference

    # ── Demo / Testing ──────────────────────────────────────────────────────
    DEMO_MODE: bool = field(default_factory=lambda: _env_bool("DEMO_MODE", False))
    REAL_DATA_ONLY: bool = field(default_factory=lambda: _env_bool("REAL_DATA_ONLY", True))
    ALLOW_SYNTHETIC_TRAINING: bool = field(default_factory=lambda: _env_bool("ALLOW_SYNTHETIC_TRAINING", False))
    SEED_DEMO_BACKTEST_HISTORY: bool = field(default_factory=lambda: _env_bool("SEED_DEMO_BACKTEST_HISTORY", False))

    # ── Live update / self-optimization ────────────────────────────────────
    ENABLE_LIVE_REPREDICT: bool = True  # Recompute predictions when fresh odds arrive
    ENABLE_SELF_OPTIMIZATION: bool = True  # Auto-tune recommendation thresholds from backtest
    STRATEGY_PROFILE_PATH: str = "data/models/strategy_profile.json"
    OPTIMIZATION_LOOKBACK_DAYS: int = 120

    # ── Risk guard ──────────────────────────────────────────────────────────
    ENABLE_RISK_GUARD: bool = True
    MAX_CONSECUTIVE_LOSSES: int = 3
    MAX_DAILY_LOSS: float = 30.0  # stop after losing 3 x $10 win bets in a day

    # ── Logging ─────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "data/hkjc_predictor.log"


config = Config()
