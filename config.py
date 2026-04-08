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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


@dataclass
class Config:
    # ── HKJC URLs (confirmed working 2026-04-02) ──────────────────────────
    HKJC_BASE_URL: str = "https://racing.hkjc.com"
    # Racecard: ?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=1
    RACECARD_URL: str = "https://racing.hkjc.com/en-us/local/information/racecard"
    # Horse profile: ?horseid=HK_2025_L126&Option=1
    HORSE_PROFILE_URL: str = "https://racing.hkjc.com/en-us/local/information/horse"
    # Results: ?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=1
    RESULTS_URL: str = "https://racing.hkjc.com/zh-hk/local/information/localresults"
    # Shared GraphQL endpoint for odds/results APIs
    HKJC_GRAPHQL_URL: str = "https://info.cld.hkjc.com/graphql/base/"
    # Odds: JS-rendered, need Playwright → bet.hkjc.com/en/racing/wp/{date}/{venue}/{race_no}
    ODDS_WP_URL: str = "https://bet.hkjc.com/en/racing/wp"
    ODDS_TRIO_URL: str = "https://bet.hkjc.com/en/racing/trio"
    # Going/weather
    WINDTRACKER_URL: str = "https://racing.hkjc.com/en-us/local/info/windtracker"
    HKO_WEATHER_URL: str = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"

    # ── Telegram ────────────────────────────────────────────────────────────
    TELEGRAM_TOKEN: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
    )
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    MESSAGE_STYLE: str = field(default_factory=lambda: os.getenv("MESSAGE_STYLE", "pro").strip().lower())
    TELEGRAM_LANGUAGE_GUARD: bool = field(
        default_factory=lambda: _env_bool("TELEGRAM_LANGUAGE_GUARD", True)
    )
    TELEGRAM_ALERT_ON_FAILURE: bool = field(
        default_factory=lambda: _env_bool("TELEGRAM_ALERT_ON_FAILURE", True)
    )
    TELEGRAM_ALERT_COOLDOWN_MINS: int = field(
        default_factory=lambda: _env_int("TELEGRAM_ALERT_COOLDOWN_MINS", 30)
    )
    TELEGRAM_ALERT_STATE_FILE: str = field(
        default_factory=lambda: _env_str("TELEGRAM_ALERT_STATE_FILE", "data/predictions/alert_state.json")
    )
    ENABLE_DAILY_ALERT_NOISE_SUMMARY: bool = field(
        default_factory=lambda: _env_bool("ENABLE_DAILY_ALERT_NOISE_SUMMARY", True)
    )
    ALERT_NOISE_TOP_N: int = field(
        default_factory=lambda: _env_int("ALERT_NOISE_TOP_N", 5)
    )

    # ── Model ───────────────────────────────────────────────────────────────
    MODEL_PATH: str = "data/models/"
    TOP_N: int = 5          # Top candidates shown per race
    ENSEMBLE_WEIGHTS: tuple = (0.5, 0.5)  # (XGBoost, LightGBM)

    # ── Web ─────────────────────────────────────────────────────────────────
    FLASK_PORT: int = field(default_factory=lambda: _env_int("PORT", _env_int("FLASK_PORT", 5000)))
    FLASK_DEBUG: bool = field(default_factory=lambda: _env_bool("FLASK_DEBUG", False))
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "racing-predictor-dev-secret"))

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
    TICK_SENT_STATE_FILE: str = field(
        default_factory=lambda: os.getenv("TICK_SENT_STATE_FILE", "data/predictions/tick_notified.json")
    )
    CRON_STATE_FILE: str = field(
        default_factory=lambda: _env_str("CRON_STATE_FILE", "data/predictions/cron_state.json")
    )
    CRON_MAINTENANCE_HOUR: int = field(default_factory=lambda: _env_int("CRON_MAINTENANCE_HOUR", 21))
    CRON_MAINTENANCE_MINUTE: int = field(default_factory=lambda: _env_int("CRON_MAINTENANCE_MINUTE", 50))
    CRON_MAINTENANCE_WINDOW_MINS: int = field(
        default_factory=lambda: _env_int("CRON_MAINTENANCE_WINDOW_MINS", 10)
    )
    HEARTBEAT_ENABLED: bool = field(
        default_factory=lambda: _env_bool("HEARTBEAT_ENABLED", True)
    )
    HEARTBEAT_HOUR: int = field(default_factory=lambda: _env_int("HEARTBEAT_HOUR", 12))
    HEARTBEAT_MINUTE: int = field(default_factory=lambda: _env_int("HEARTBEAT_MINUTE", 0))
    HEARTBEAT_WINDOW_MINS: int = field(
        default_factory=lambda: _env_int("HEARTBEAT_WINDOW_MINS", 10)
    )
    MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED: bool = field(
        default_factory=lambda: _env_bool("MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED", True)
    )
    MAINTENANCE_REPORT_DIR: str = field(
        default_factory=lambda: _env_str("MAINTENANCE_REPORT_DIR", "data/reports")
    )
    ENABLE_RETENTION_CLEANUP: bool = field(
        default_factory=lambda: _env_bool("ENABLE_RETENTION_CLEANUP", True)
    )
    RETENTION_POLICY_FILE: str = field(
        default_factory=lambda: _env_str("RETENTION_POLICY_FILE", "data/retention_policy.json")
    )
    TRAINING_RETENTION_MANIFEST_FILE: str = field(
        default_factory=lambda: _env_str("TRAINING_RETENTION_MANIFEST_FILE", "data/training_retention_manifest.json")
    )
    DAILY_RETRAIN_ENABLED: bool = field(default_factory=lambda: _env_bool("DAILY_RETRAIN_ENABLED", True))
    DAILY_RETRAIN_MIN_NEW_SETTLED: int = field(
        default_factory=lambda: _env_int("DAILY_RETRAIN_MIN_NEW_SETTLED", 1)
    )
    DAILY_RETRAIN_SYNTHETIC_RACES_BASE: int = field(
        default_factory=lambda: _env_int("DAILY_RETRAIN_SYNTHETIC_RACES_BASE", 600)
    )
    DAILY_RETRAIN_SYNTHETIC_RACES_SCALE: int = field(
        default_factory=lambda: _env_int("DAILY_RETRAIN_SYNTHETIC_RACES_SCALE", 8)
    )

    # ── Risk guard ──────────────────────────────────────────────────────────
    ENABLE_RISK_GUARD: bool = True
    MAX_CONSECUTIVE_LOSSES: int = 3
    MAX_DAILY_LOSS: float = 30.0  # stop after losing 3 x $10 win bets in a day

    # ── Logging ─────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "data/hkjc_predictor.log"


config = Config()
