#!/usr/bin/env python3
"""
HKJC 賽馬預測系統 - 主程式入口
Main entry point for the HKJC Horse Racing Prediction System.

Usage:
    python run.py                    # Start Flask + scheduler (default)
    python run.py --mode web         # Flask web dashboard only
    python run.py --mode debug       # One-shot diagnostics for scheduled runs
    python run.py --mode cron        # Single-command cron: tick + timed maintenance
    python run.py --mode tick        # One-shot fetch + predict + Telegram send
    python run.py --mode maintenance # Run backtest + optimize + Telegram summary
    python run.py --mode retention-cleanup # Apply retention policy and prune old artifacts
    python run.py --mode authenticity-audit # Export provenance authenticity audit report
    python run.py --mode migrate-provenance # One-time migrate legacy source provenance (C/D)
    python run.py --mode send-all-previews-now # Force send pre-race preview for all races
    python run.py --mode predict     # Run predictions for today and print
    python run.py --mode backtest    # Show backtest stats
    python run.py --mode fetch       # Fetch racecard and odds only
    python run.py --mode train       # Train/retrain model on synthetic data
    python run.py --mode demo        # Run all in demo mode (no real data)
    python run.py --mode optimize    # Optimize recommendation thresholds using real backtest
    python run.py --mode telegram-test  # Send 3 Telegram template test messages

    Options:
      --port PORT     Override Flask port (default: 5000)
      --debug         Enable Flask debug mode
      --no-scheduler  Start Flask without scheduler
"""
import argparse
import glob
import hashlib
import html
import json
import logging
import os
import sys
import traceback
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# ── Ensure project root is on Python path ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Load .env file if present ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not required

from config import config


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with colourised output."""
    os.makedirs(os.path.dirname(config.LOG_FILE) if os.path.dirname(config.LOG_FILE) else ".", exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
    ]
    try:
        handlers.append(logging.FileHandler(config.LOG_FILE, encoding="utf-8"))
    except OSError:
        pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    # Silence noisy third-party loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
HKT = ZoneInfo("Asia/Hong_Kong")


# ── Mode Handlers ──────────────────────────────────────────────────────────────

def run_web(port: Optional[int] = None, debug: Optional[bool] = None, with_scheduler: bool = True) -> None:
    """Start the Flask web dashboard, optionally with the scheduler."""
    from web.app import create_app

    app = create_app()
    _port = port or config.FLASK_PORT
    _debug = debug if debug is not None else config.FLASK_DEBUG

    scheduler_manager = None
    if with_scheduler:
        try:
            from scheduler.tasks import SchedulerManager
            # We pass None for app_state; scheduler updates shared state
            scheduler_manager = SchedulerManager()
            scheduler_manager.start()
            logger.info("Scheduler started alongside Flask")
        except Exception as e:
            logger.warning(f"Scheduler could not start: {e}")

    logger.info(f"Starting Flask on http://0.0.0.0:{_port}")
    logger.info(f"Demo mode: {'ON' if config.DEMO_MODE else 'OFF'}")

    try:
        app.run(host="0.0.0.0", port=_port, debug=_debug, use_reloader=False)
    finally:
        if scheduler_manager:
            scheduler_manager.stop()


def run_predict() -> None:
    """Run prediction pipeline and print results to stdout."""
    logger.info("=== Prediction Mode ===")
    today = datetime.now(HKT).strftime("%Y-%m-%d")

    from scraper.racecard import fetch_racecard
    from scraper.horse_profile import fetch_horse_profile
    from scraper.odds import fetch_odds
    from features.elo import ELOSystem
    from features.jockey_trainer import JockeyTrainerStats
    from features.draw import DrawStats
    from model.trainer import EnsembleTrainer
    from model.predictor import RacePredictor
    from model.backtest import Backtester

    logger.info(f"Fetching races for {today}...")
    races = fetch_racecard(today)
    logger.info(f"Found {len(races)} races")
    if config.REAL_DATA_ONLY and not config.DEMO_MODE and not races:
        raise RuntimeError("No real races available while REAL_DATA_ONLY=True; aborting predict run")

    elo = ELOSystem(k_factor=config.ELO_K_FACTOR, base_rating=config.ELO_BASE)
    elo.load_ratings("data/processed/elo_ratings.json")

    jt_stats = JockeyTrainerStats()
    jt_stats.load()

    draw_stats = DrawStats()
    if not draw_stats.load():
        draw_stats.bootstrap_from_mock()

    trainer = EnsembleTrainer()
    if not trainer.load():
        if config.REAL_DATA_ONLY and not config.ALLOW_SYNTHETIC_TRAINING:
            raise RuntimeError(
                "Model not found and synthetic training is disabled (REAL_DATA_ONLY=True). "
                "Please train with real data or explicitly enable ALLOW_SYNTHETIC_TRAINING."
            )
        logger.info("Training model on synthetic data...")
        trainer.train()
        trainer.save()

    predictor = RacePredictor(trainer=trainer, top_n=config.TOP_N)
    backtester = Backtester(repair_missing=True)

    print("\n" + "="*70)
    print(f"  HKJC 賽馬預測系統 - {today}")
    print("="*70)

    for race in races:
        horse_profiles = {h.horse_code: fetch_horse_profile(h.horse_code) for h in race.horses}
        odds = fetch_odds(race.race_date, race.venue_code, race.race_number,
                          [h.horse_number for h in race.horses])

        pred = predictor.predict_race_from_components(
            race=race,
            horse_profiles=horse_profiles,
            odds=odds,
            elo=elo,
            jt_stats=jt_stats,
            draw_stats=draw_stats,
        )

        race_id = f"{race.race_date}_{race.venue_code}_{race.race_number}"
        backtester.record_prediction(race_id, pred, race.race_date)

        print(f"\n第{race.race_number}場 | {race.venue} | {race.distance}m | "
              f"{race.race_class} | {race.going} | {race.start_time}")
        print(f"  信心度: {pred.confidence:.0%}")
        print(f"  頭{len(pred.top5)}名預測:")

        for h in pred.top5:
            value_tag = " ★值博" if h.is_value_bet else ""
            print(f"    {h.rank}. #{h.horse_number} {h.horse_name[:20]:<22} "
                  f"模型:{h.model_prob:.1%}  賠率:{h.win_odds}  EV:${h.expected_value:+.2f}{value_tag}")

        if pred.trio_suggestions:
            t = pred.trio_suggestions[0]
            print(f"  三T首選: #{t[0]}/{t[1]}/{t[2]}")

    print("\n" + "="*70)
    print("  ⚠️  本系統僅供研究參考，不構成投注建議")
    print("="*70 + "\n")


def run_debug() -> None:
    """Run one-shot diagnostics and print categorized reasons for quick troubleshooting."""
    logger.info("=== Debug Mode ===")
    now_hkt = datetime.now(HKT)
    today = now_hkt.strftime("%Y-%m-%d")
    race_weekdays = {2, 5, 6}  # Wed/Sat/Sun

    print("\n" + "=" * 78)
    print(f"  HKJC 系統診斷報告 | {now_hkt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 78)

    # 1) Race-day heuristic
    if now_hkt.weekday() in race_weekdays:
        print("[賽日檢查] ✅ 可能是賽日（週三/六/日）")
    else:
        print("[賽日檢查] ⚠️ 非常見賽日（週三/六/日之外），可能無場次")

    # 2) Network and HKJC page availability check
    page_flags: dict[str, Optional[bool]] = {"ST": None, "HV": None}
    network_ok = False
    try:
        import requests

        for venue in ("ST", "HV"):
            params = {
                "racedate": today.replace("-", "/"),
                "Racecourse": venue,
                "RaceNo": "1",
            }
            resp = requests.get(config.RACECARD_URL, params=params, timeout=15)
            resp.raise_for_status()
            network_ok = True
            html = resp.text or ""
            no_info = ("No information" in html) or ("no information" in html.lower())
            page_flags[venue] = not no_info

        print("[網絡檢查] ✅ 可連線 HKJC racecard")
    except Exception as e:
        print(f"[網絡檢查] ❌ 網絡抓取失敗: {e}")

    if network_ok:
        if page_flags["ST"] or page_flags["HV"]:
            venues = [v for v, ok in page_flags.items() if ok]
            print(f"[場次頁面] ✅ HKJC 有可用頁面（venue={','.join(venues)}）")
        else:
            print("[場次頁面] ⚠️ HKJC 顯示今日無場次（ST/HV Race 1 皆 No information）")

    # 3) Pipeline fetch check
    races = []
    fetch_error = None
    try:
        from scraper.racecard import fetch_racecard

        races = fetch_racecard(today)
    except Exception as e:
        fetch_error = e

    if fetch_error:
        print(f"[抓取流程] ❌ fetch_racecard 發生錯誤: {fetch_error}")
    elif races:
        print(f"[抓取流程] ✅ 解析成功，共 {len(races)} 場")
    else:
        if not network_ok:
            print("[抓取流程] ❌ 無法抓取，主因偏向網絡/連線")
        elif page_flags["ST"] is False and page_flags["HV"] is False:
            print("[抓取流程] ⚠️ 無場次資料（較可能是非賽日或尚未開放）")
            try:
                from scraper.racecard import find_next_meeting_date

                next_meeting = find_next_meeting_date(today, max_days=7)
                if next_meeting:
                    print(f"[下一賽日] ℹ️ 下一個可用賽事：{next_meeting[0]} ({next_meeting[1]})")
            except Exception:
                pass
        else:
            print("[抓取流程] ⚠️ 有頁面但抓取為 0 場，可能網站結構變更或解析規則需更新")

    # 4) Telegram config check
    from notifier.telegram import TelegramNotifier

    notifier = TelegramNotifier()
    if notifier.is_configured():
        print(f"[Telegram] ✅ 已設定（風格={getattr(config, 'MESSAGE_STYLE', 'pro')})")
    else:
        print("[Telegram] ❌ 未設定 TELEGRAM_TOKEN / TELEGRAM_CHAT_ID")

    # 5) Model load check
    try:
        from model.trainer import EnsembleTrainer

        trainer = EnsembleTrainer()
        model_ok = trainer.load()
        if model_ok:
            print("[模型檢查] ✅ 模型可載入")
        else:
            print("[模型檢查] ⚠️ 模型未載入（可能不存在或 provenance 條件不符）")
    except Exception as e:
        print(f"[模型檢查] ❌ 載入失敗: {e}")

    print("=" * 78 + "\n")


def _load_tick_sent_state(path: str, target_date: str) -> set[str]:
    """Load per-day sent race IDs for tick de-duplication."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sent = payload.get(target_date, [])
        if isinstance(sent, list):
            return {str(x) for x in sent}
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not load tick sent state: {e}")
    return set()


def _save_tick_sent_state(path: str, target_date: str, sent_ids: set[str]) -> None:
    """Persist per-day sent race IDs for tick de-duplication."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {target_date: sorted(sent_ids)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Could not save tick sent state: {e}")


def _load_json_state(path: str) -> dict:
    """Load state dictionary from disk."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not load cron state: {e}")
    return {}


def _save_json_state(path: str, payload: dict) -> None:
    """Save state dictionary to disk."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Could not save cron state: {e}")


def _apply_retention_policy() -> dict:
    """Apply retention policy to generated artifacts and return cleanup summary."""
    policy_path = getattr(config, "RETENTION_POLICY_FILE", "data/retention_policy.json")
    default_policy = {
        "version": "1.0",
        "rules": [
            {
                "name": "artifact_reports_90d",
                "enabled": True,
                "retention_days": 90,
                "patterns": ["data/reports/*.json", "data/processed/_debug_*.html"],
            }
        ],
    }

    policy = default_policy
    if os.path.exists(policy_path):
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                policy = loaded
        except Exception as e:
            logger.warning("Retention policy parse failed (%s): %s", policy_path, e)

    now_ts = datetime.now(HKT).timestamp()
    summary = {
        "policy_path": policy_path,
        "policy_version": str(policy.get("version", "")),
        "scanned_files": 0,
        "deleted_files": 0,
        "deleted_paths": [],
        "errors": [],
    }
    protected = [str(p).replace("\\", "/").rstrip("/") for p in list(policy.get("protected_paths", []) or [])]

    def _is_protected(path: str) -> bool:
        p = str(path).replace("\\", "/")
        for pref in protected:
            if not pref:
                continue
            if p == pref or p.startswith(pref + "/"):
                return True
        return False

    for rule in policy.get("rules", []):
        if not isinstance(rule, dict) or not bool(rule.get("enabled", True)):
            continue

        retention_days = int(rule.get("retention_days", 90) or 90)
        if retention_days <= 0:
            continue
        threshold_seconds = retention_days * 86400
        patterns = list(rule.get("patterns", []) or [])

        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                if not os.path.isfile(path):
                    continue
                if _is_protected(path):
                    continue
                summary["scanned_files"] += 1
                try:
                    age_seconds = now_ts - os.path.getmtime(path)
                    if age_seconds >= threshold_seconds:
                        os.remove(path)
                        summary["deleted_files"] += 1
                        if len(summary["deleted_paths"]) < 20:
                            summary["deleted_paths"].append(path)
                except Exception as e:
                    summary["errors"].append(f"{path}: {e}")

    logger.info(
        "Retention cleanup done: scanned=%s deleted=%s errors=%s",
        summary["scanned_files"],
        summary["deleted_files"],
        len(summary["errors"]),
    )
    return summary


def _load_training_retention_manifest() -> dict:
    """Load machine-readable training retention manifest, writing defaults if missing."""
    manifest_path = getattr(config, "TRAINING_RETENTION_MANIFEST_FILE", "data/training_retention_manifest.json")
    default_manifest = {
        "version": "1.0",
        "updated_at": datetime.now(HKT).isoformat(),
        "quality_tiers": {
            "A": {"label": "官方 GraphQL", "allow_training": True, "priority": 1},
            "B": {"label": "官方 localresults", "allow_training": True, "priority": 2},
            "C": {"label": "可追溯回填", "allow_training": False, "priority": 3},
            "D": {"label": "來源未標註", "allow_training": False, "priority": 4},
        },
        "training_policy": {
            "allowed_confidence_tiers": ["A", "B"],
            "minimum_total_records": 24,
            "minimum_by_tier": {"A": 8, "B": 8},
        },
        "required_assets": [
            {"path": "data/predictions/history.json", "must_exist": True},
            {"path": "data/models/meta.json", "must_exist": False},
            {"path": "data/models/strategy_profile.json", "must_exist": False},
        ],
        "retention_protection": {
            "paths": [
                "data/predictions/history.json",
                "data/models/",
                "data/models/meta.json",
            ]
        },
    }

    if not os.path.exists(manifest_path):
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(default_manifest, f, ensure_ascii=False, indent=2)
        return default_manifest

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception as e:
        logger.warning("Failed to parse training retention manifest (%s): %s", manifest_path, e)

    return default_manifest


def _source_quality_tier(rec) -> str:
    tier = str(getattr(rec, "result_source_confidence", "") or "D").strip().upper()[:1]
    return tier if tier in {"A", "B", "C", "D"} else "D"


def _evaluate_training_retention_status(settled_real: List, manifest: dict) -> Tuple[List, dict]:
    """Evaluate quality-tier retention readiness and return eligible records plus report."""
    policy = dict(manifest.get("training_policy", {}) or {})
    allowed_tiers = [str(x).upper()[:1] for x in list(policy.get("allowed_confidence_tiers", ["A", "B"]) or ["A", "B"])]
    allowed_tiers = [x for x in allowed_tiers if x in {"A", "B", "C", "D"}]
    if not allowed_tiers:
        allowed_tiers = ["A", "B"]

    by_tier: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    eligible = []
    for rec in settled_real:
        tier = _source_quality_tier(rec)
        by_tier[tier] += 1
        if tier in allowed_tiers:
            eligible.append(rec)

    min_total = max(1, int(policy.get("minimum_total_records", 24) or 24))
    min_by_tier = dict(policy.get("minimum_by_tier", {}) or {})
    blocking_reasons: List[str] = []

    if len(eligible) < min_total:
        blocking_reasons.append(f"eligible_records_below_min_total({len(eligible)}<{min_total})")

    for tier, min_count in min_by_tier.items():
        t = str(tier).upper()[:1]
        if t not in by_tier:
            continue
        required = max(0, int(min_count or 0))
        if by_tier[t] < required:
            blocking_reasons.append(f"tier_{t}_below_min({by_tier[t]}<{required})")

    asset_checks = []
    for item in list(manifest.get("required_assets", []) or []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "") or "")
        if not path:
            continue
        must_exist = bool(item.get("must_exist", True))
        exists = os.path.exists(path)
        ok = (exists or not must_exist)
        asset_checks.append({"path": path, "must_exist": must_exist, "exists": exists, "ok": ok})
        if not ok:
            blocking_reasons.append(f"missing_required_asset:{path}")

    report = {
        "manifest_path": getattr(config, "TRAINING_RETENTION_MANIFEST_FILE", "data/training_retention_manifest.json"),
        "allowed_tiers": allowed_tiers,
        "counts_by_tier": by_tier,
        "total_settled_real": len(settled_real),
        "eligible_records": len(eligible),
        "minimum_total_records": min_total,
        "minimum_by_tier": min_by_tier,
        "asset_checks": asset_checks,
        "ready_for_retrain": len(blocking_reasons) == 0,
        "blocking_reasons": blocking_reasons,
    }
    return eligible, report


def _send_failure_alert(mode: str, err: Exception) -> None:
    """Best-effort Telegram alert for fatal run failures."""
    if not bool(getattr(config, "TELEGRAM_ALERT_ON_FAILURE", True)):
        return

    try:
        from notifier.telegram import TelegramNotifier

        cooldown_mins = max(1, int(getattr(config, "TELEGRAM_ALERT_COOLDOWN_MINS", 30) or 30))
        state_path = str(getattr(config, "TELEGRAM_ALERT_STATE_FILE", "data/predictions/alert_state.json") or "")

        err_type = type(err).__name__
        err_msg = str(err or "").strip()
        fingerprint_src = f"{mode}|{err_type}|{err_msg}"
        fingerprint = hashlib.sha1(fingerprint_src.encode("utf-8", errors="ignore")).hexdigest()

        now_hkt = datetime.now(HKT)
        now_ts = now_hkt.timestamp()

        state = _load_json_state(state_path)
        alerts = state.get("alerts", {}) if isinstance(state.get("alerts", {}), dict) else {}
        prev = alerts.get(fingerprint, {}) if isinstance(alerts.get(fingerprint, {}), dict) else {}
        last_ts = float(prev.get("last_sent_ts", 0.0) or 0.0)
        elapsed = max(0.0, now_ts - last_ts)

        if last_ts > 0 and elapsed < (cooldown_mins * 60):
            prev["suppressed_count"] = int(prev.get("suppressed_count", 0) or 0) + 1
            day_key = now_hkt.strftime("%Y-%m-%d")
            by_day = prev.get("suppressed_by_day", {}) if isinstance(prev.get("suppressed_by_day", {}), dict) else {}
            by_day[day_key] = int(by_day.get(day_key, 0) or 0) + 1
            # Keep recent 30 days only to bound state size.
            if len(by_day) > 30:
                keys = sorted(by_day.keys())
                for k in keys[:-30]:
                    by_day.pop(k, None)
            prev["suppressed_by_day"] = by_day
            prev["last_seen_ts"] = now_ts
            prev["last_seen_at"] = now_hkt.isoformat()
            prev["mode"] = mode
            prev["error_type"] = err_type
            prev["error_message"] = err_msg[:500]
            alerts[fingerprint] = prev
            state["alerts"] = alerts
            _save_json_state(state_path, state)
            logger.info(
                "Suppress duplicate failure alert: mode=%s fingerprint=%s cooldown_mins=%s",
                mode,
                fingerprint[:12],
                cooldown_mins,
            )
            return

        notifier = TelegramNotifier()
        if not notifier.is_configured():
            return

        tb = traceback.format_exc(limit=10)
        suppressed_count = int(prev.get("suppressed_count", 0) or 0)
        last_sent_at = str(prev.get("last_sent_at", "") or "")
        last_seen_at = str(prev.get("last_seen_at", "") or "")
        suppression_lines = []
        if suppressed_count > 0:
            suppression_lines.append(f"suppressed_since_last_sent={suppressed_count}")
            if last_sent_at:
                suppression_lines.append(f"suppression_window_start={last_sent_at}")
            if last_seen_at:
                suppression_lines.append(f"suppression_window_end={last_seen_at}")

        body = (
            f"mode={mode}\n"
            f"error={type(err).__name__}: {err}\n"
            f"fingerprint={fingerprint[:12]}\n"
            + ("\n".join(suppression_lines) + "\n" if suppression_lines else "")
            +
            f"traceback:\n{tb[:3000]}"
        )
        notifier.send_sync(
            "🚨 <b>排程執行失敗</b>\n"
            f"時間: {datetime.now(HKT).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"\n<pre>{body}</pre>"
        )

        alerts[fingerprint] = {
            "mode": mode,
            "error_type": err_type,
            "error_message": err_msg[:500],
            "last_sent_ts": now_ts,
            "last_sent_at": now_hkt.isoformat(),
            "last_seen_ts": now_ts,
            "last_seen_at": now_hkt.isoformat(),
            "cooldown_mins": cooldown_mins,
            "suppressed_count": 0,
            "suppressed_by_day": dict(prev.get("suppressed_by_day", {}) if isinstance(prev.get("suppressed_by_day", {}), dict) else {}),
        }
        state["alerts"] = alerts
        _save_json_state(state_path, state)
    except Exception as notify_err:
        logger.error("Failure alert send failed: %s", notify_err)


def _get_daily_alert_noise_summary(target_date: str) -> dict:
    """Build daily suppression summary from alert dedup state."""
    state_path = str(getattr(config, "TELEGRAM_ALERT_STATE_FILE", "data/predictions/alert_state.json") or "")
    top_n = max(1, int(getattr(config, "ALERT_NOISE_TOP_N", 5) or 5))
    state = _load_json_state(state_path)
    alerts = state.get("alerts", {}) if isinstance(state.get("alerts", {}), dict) else {}

    rows = []
    total = 0
    for fingerprint, meta in alerts.items():
        if not isinstance(meta, dict):
            continue
        by_day = meta.get("suppressed_by_day", {}) if isinstance(meta.get("suppressed_by_day", {}), dict) else {}
        count = int(by_day.get(target_date, 0) or 0)
        if count <= 0:
            continue
        total += count
        rows.append({
            "fingerprint": str(fingerprint)[:12],
            "mode": str(meta.get("mode", "") or ""),
            "error_type": str(meta.get("error_type", "") or ""),
            "error_message": str(meta.get("error_message", "") or "")[:120],
            "suppressed_count": count,
        })

    rows.sort(key=lambda x: int(x.get("suppressed_count", 0)), reverse=True)
    top_rows = rows[:top_n]
    return {
        "date": target_date,
        "state_path": state_path,
        "total_suppressed": total,
        "unique_fingerprints": len(rows),
        "top_noisy": top_rows,
    }


def _should_run_maintenance_now(now_hkt: datetime) -> bool:
    """Return True when current time falls into the configured daily maintenance window."""
    target_minutes = int(config.CRON_MAINTENANCE_HOUR) * 60 + int(config.CRON_MAINTENANCE_MINUTE)
    current_minutes = now_hkt.hour * 60 + now_hkt.minute
    window = max(1, int(config.CRON_MAINTENANCE_WINDOW_MINS))
    return target_minutes <= current_minutes < (target_minutes + window)


def _should_send_heartbeat_now(now_hkt: datetime) -> bool:
    """Return True when current time falls into the configured daily heartbeat window."""
    if not bool(getattr(config, "HEARTBEAT_ENABLED", True)):
        return False
    target_minutes = int(config.HEARTBEAT_HOUR) * 60 + int(config.HEARTBEAT_MINUTE)
    current_minutes = now_hkt.hour * 60 + now_hkt.minute
    window = max(1, int(getattr(config, "HEARTBEAT_WINDOW_MINS", 10) or 10))
    return target_minutes <= current_minutes < (target_minutes + window)


def _send_cron_heartbeat(now_hkt: datetime, state: dict) -> bool:
    """Send one daily heartbeat message in the configured window."""
    from notifier.telegram import TelegramNotifier

    today = now_hkt.strftime("%Y-%m-%d")
    heartbeat_map = state.get("heartbeat_sent", {})
    if not isinstance(heartbeat_map, dict):
        heartbeat_map = {}

    if heartbeat_map.get(today):
        return False

    notifier = TelegramNotifier()
    if not notifier.is_configured():
        logger.warning("Telegram not configured; heartbeat not sent")
        return False

    is_pro_style = (config.MESSAGE_STYLE or "pro").strip().lower() != "casual"
    maintenance_map = state.get("maintenance_sent", {}) if isinstance(state.get("maintenance_sent", {}), dict) else {}
    maintenance_done = bool(maintenance_map.get(today))
    last_maintenance = str(state.get("maintenance_last_run_at", "") or "N/A")

    lines = [
        f"{'💓 <b>排程健康檢查</b>' if is_pro_style else '💓 <b>系統心跳</b>'}",
        f"時間: {now_hkt.strftime('%Y-%m-%d %H:%M:%S')} HKT",
        "",
        f"• cron mode: 正常執行中",
        f"• 今日 maintenance: {'已完成' if maintenance_done else '未完成/未到時段'}",
        f"• 最近 maintenance 時間: {last_maintenance}",
        f"• heartbeat 時段: {int(config.HEARTBEAT_HOUR):02d}:{int(config.HEARTBEAT_MINUTE):02d} "
        f"(+{max(1, int(getattr(config, 'HEARTBEAT_WINDOW_MINS', 10) or 10))}m)",
    ]
    lines.extend([
        "",
        f"<i>{'⚠️ 僅供研究參考，不構成投注建議' if is_pro_style else '⚠️ 只供研究參考，唔係投注建議'}</i>",
    ])

    ok = notifier.send_sync("\n".join(lines))
    if not ok:
        return False

    heartbeat_map[today] = now_hkt.isoformat()
    keys = sorted(heartbeat_map.keys())
    if len(keys) > 14:
        for k in keys[:-14]:
            heartbeat_map.pop(k, None)
    state["heartbeat_sent"] = heartbeat_map
    logger.info("Cron heartbeat sent")
    return True


def _latest_settled_marker(records: list) -> str:
    """Build a stable marker for the latest settled real-result record."""
    if not records:
        return ""

    latest = max(
        records,
        key=lambda r: (
            str(getattr(r, "result_recorded_at", "") or ""),
            str(getattr(r, "race_date", "") or ""),
            int(getattr(r, "race_number", 0) or 0),
            str(getattr(r, "race_id", "") or ""),
        ),
    )
    return (
        f"{getattr(latest, 'race_id', '')}|"
        f"{getattr(latest, 'result_recorded_at', '')}|"
        f"{getattr(latest, 'actual_winner', 0)}"
    )


def _build_combo_recommendations(day_records: List[Dict]) -> Dict[str, object]:
    """Build Trio and Quinella suggestions from the strongest recent settled record."""
    if not day_records:
        return {}

    anchor = max(
        day_records,
        key=lambda item: (
            float(item.get("model_confidence", 0.0) or 0.0),
            -float(item.get("predicted_winner_odds", 0.0) or 0.0),
            -int(item.get("race_number", 0) or 0),
        ),
    )
    top3 = [int(x) for x in list(anchor.get("predicted_top3", []) or [])[:3] if int(x) > 0]
    if len(top3) < 2:
        return {}

    trio_primary = tuple(top3[:3]) if len(top3) >= 3 else tuple()
    quinella_primary = tuple(top3[:2])
    quinella_backup = tuple((top3[0], top3[2])) if len(top3) >= 3 else tuple()
    return {
        "race_id": str(anchor.get("race_id", "") or ""),
        "confidence": float(anchor.get("model_confidence", 0.0) or 0.0),
        "winner_odds": float(anchor.get("predicted_winner_odds", 0.0) or 0.0),
        "trio_primary": trio_primary,
        "quinella_primary": quinella_primary,
        "quinella_backup": quinella_backup,
    }


def _format_combo(combo: tuple[int, ...]) -> str:
    """Render horse-number combinations for Telegram text."""
    if not combo:
        return "N/A"
    return " / ".join(f"#{number}" for number in combo)


def _write_maintenance_report(report: dict) -> str:
    """Persist maintenance analysis report and return file path."""
    report_dir = config.MAINTENANCE_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    date_str = report.get("date", date.today().strftime("%Y-%m-%d"))
    path = os.path.join(report_dir, f"maintenance_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def _write_authenticity_audit_report(bt) -> str:
    """Persist a per-race provenance audit report and return file path."""
    report_dir = config.MAINTENANCE_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    date_str = datetime.now(HKT).strftime("%Y-%m-%d")
    path = os.path.join(report_dir, f"authenticity_audit_{date_str}.json")

    settled = sorted(
        bt.get_settled_records(real_only=False),
        key=lambda r: (str(getattr(r, "race_date", "")), int(getattr(r, "race_number", 0) or 0)),
    )

    source_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    rows = []
    for r in settled:
        src = str(getattr(r, "result_source", "") or "legacy_unverified")
        conf = str(getattr(r, "result_source_confidence", "") or "D").upper()[:1]
        source_counts[src] = source_counts.get(src, 0) + 1
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        rows.append({
            "race_id": r.race_id,
            "race_date": r.race_date,
            "race_number": r.race_number,
            "venue": r.venue,
            "is_real_result": bool(getattr(r, "is_real_result", False)),
            "actual_winner": int(getattr(r, "actual_winner", 0) or 0),
            "result_source": src,
            "result_source_url": str(getattr(r, "result_source_url", "") or ""),
            "result_source_confidence": conf,
            "result_source_note": str(getattr(r, "result_source_note", "") or ""),
            "result_recorded_at": str(getattr(r, "result_recorded_at", "") or ""),
        })

    report = {
        "generated_at": datetime.now(HKT).isoformat(),
        "total_settled_records": len(settled),
        "source_counts": source_counts,
        "confidence_counts": confidence_counts,
        "confidence_legend": {
            "A": "官方 GraphQL 端點",
            "B": "官方 localresults 網頁解析",
            "C": "腳本/人工回填（可追溯但非即時官方 API）",
            "D": "歷史遺留或來源未標註",
        },
        "records": rows,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def _run_daily_retrain(settled_real_records: list, state: dict) -> dict:
    """Run daily retraining when new settled data is available and allowed by config."""
    retrain = {
        "attempted": False,
        "ran": False,
        "reason": "",
        "new_settled_since_last": 0,
        "target_real_rows": 0,
        "metrics": {},
    }

    settled_real_count = len(settled_real_records or [])

    if not config.DAILY_RETRAIN_ENABLED:
        retrain["reason"] = "disabled_by_config"
        return retrain

    last_settled_count = int(state.get("retrain_last_settled_count", 0) or 0)
    new_settled = max(0, settled_real_count - last_settled_count)
    retrain["new_settled_since_last"] = new_settled

    if new_settled < int(config.DAILY_RETRAIN_MIN_NEW_SETTLED):
        retrain["reason"] = "no_enough_new_settled"
        return retrain

    from model.trainer import EnsembleTrainer

    retrain["attempted"] = True
    trainer = EnsembleTrainer()

    df = trainer.build_real_incremental_data(settled_real_records, min_rows=30)
    if df is None or df.empty:
        retrain["reason"] = "insufficient_real_incremental_data"
        return retrain

    retrain["target_real_rows"] = int(len(df))
    metrics = trainer.train(df=df, training_source="real_incremental")
    trainer.save()

    retrain["ran"] = True
    retrain["metrics"] = metrics
    retrain["reason"] = "ok"
    state["retrain_last_settled_count"] = settled_real_count
    state["retrain_last_run_at"] = datetime.now(HKT).isoformat()
    return retrain


def _sync_recent_real_results(days_back: int = 2) -> dict:
    """
    Backfill recent official results into backtest history for cron-only setups.

    This lets maintenance optimize/retrain on newer settled races even when
    the APScheduler long-running web mode is not used.
    """
    from model.backtest import Backtester
    from scraper.results_fetcher import fetch_day_results

    bt = Backtester(repair_missing=True)
    records = list(bt._records.values())

    today_hkt = datetime.now(HKT).date()
    target_dates = {
        (today_hkt - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(max(1, int(days_back)) + 1)
    }
    unresolved = {
        r.race_id
        for r in records
        if (r.actual_winner <= 0) and (r.race_date in target_dates)
    }

    if not unresolved:
        return {"updated": 0, "checked_dates": sorted(target_dates)}

    updated = 0
    for d in sorted(target_dates):
        for venue in ("ST", "HV"):
            try:
                results = fetch_day_results(d, venue)
            except Exception as e:
                logger.warning(f"Result sync failed for {d} {venue}: {e}")
                continue

            for rr in results:
                if rr.race_id not in unresolved:
                    continue

                try:
                    place_dividends = {
                        int(k): float(v)
                        for k, v in (rr.place_dividends or {}).items()
                        if str(k).isdigit()
                    }
                    rec = bt.record_result(
                        race_id=rr.race_id,
                        actual_result=list(rr.finishing_order or []),
                        trio_dividend=float(rr.trio_dividend or 0.0),
                        place_dividends=place_dividends,
                        win_dividend=float(rr.win_dividend or 0.0),
                        is_real_result=True,
                        result_source=str(getattr(rr, "result_source", "") or "hkjc_unknown"),
                        result_source_url=str(getattr(rr, "result_source_url", "") or ""),
                        result_source_confidence=str(getattr(rr, "result_source_confidence", "") or "B"),
                        result_source_note=str(getattr(rr, "result_source_note", "") or ""),
                    )
                    if rec is not None:
                        updated += 1
                except Exception as e:
                    logger.warning(f"Result sync parse/apply failed for {rr.race_id}: {e}")

    return {"updated": updated, "checked_dates": sorted(target_dates)}


def run_tick() -> None:
    """
    Run one-shot cron cycle: fetch + predict + pre-race send, then exit.
    Designed for external schedulers (e.g. every 5 minutes).
    """
    logger.info("=== Tick Mode ===")
    now_hkt = datetime.now(HKT)
    today = now_hkt.strftime("%Y-%m-%d")

    from scraper.racecard import fetch_racecard
    from scraper.horse_profile import fetch_horse_profile
    from scraper.odds import fetch_odds
    from features.elo import ELOSystem
    from features.jockey_trainer import JockeyTrainerStats
    from features.draw import DrawStats
    from model.trainer import EnsembleTrainer
    from model.predictor import RacePredictor
    from model.backtest import Backtester
    from notifier.telegram import TelegramNotifier

    races = fetch_racecard(today)
    if not races:
        logger.info("No races found for today; tick exits")
        return

    elo = ELOSystem(k_factor=config.ELO_K_FACTOR, base_rating=config.ELO_BASE)
    elo.load_ratings("data/processed/elo_ratings.json")

    jt_stats = JockeyTrainerStats()
    jt_stats.load()

    draw_stats = DrawStats()
    if not draw_stats.load() and not config.REAL_DATA_ONLY:
        draw_stats.bootstrap_from_mock()

    trainer = EnsembleTrainer()
    if not trainer.load():
        if config.REAL_DATA_ONLY and not config.ALLOW_SYNTHETIC_TRAINING:
            logger.error(
                "Model not available (REAL_DATA_ONLY=True, ALLOW_SYNTHETIC_TRAINING=False); "
                "tick run exits without sending notifications"
            )
            return
        logger.info("Model missing; training fallback model for tick mode")
        trainer.train()
        trainer.save()

    predictor = RacePredictor(trainer=trainer, top_n=config.TOP_N)
    notifier = TelegramNotifier()
    backtester = Backtester(repair_missing=True)
    notify_window = int(config.PRE_RACE_NOTIFY_MINS)

    state_path = config.TICK_SENT_STATE_FILE
    sent_ids = _load_tick_sent_state(state_path, today)

    profile_cache = {}
    eligible = 0
    sent = 0

    for race in races:
        try:
            hour, minute = map(int, race.start_time.split(":"))
            race_day = datetime.strptime(race.race_date, "%Y-%m-%d")
            race_dt = datetime(
                race_day.year,
                race_day.month,
                race_day.day,
                hour,
                minute,
                tzinfo=HKT,
            )
            mins_to_start = (race_dt - now_hkt).total_seconds() / 60.0
        except Exception:
            logger.warning(f"Invalid race start time for race {race.race_number}; skipping")
            continue

        # Only notify upcoming races in the configured pre-race window.
        if mins_to_start < 0 or mins_to_start > notify_window:
            continue

        eligible += 1
        race_id = f"{race.race_date}_{race.venue_code}_{race.race_number}"
        if race_id in sent_ids:
            logger.info(f"Race {race.race_number} already sent in tick state; skipping duplicate")
            continue

        horse_profiles = {}
        for h in race.horses:
            if h.horse_code not in profile_cache:
                profile_cache[h.horse_code] = fetch_horse_profile(h.horse_code)
            horse_profiles[h.horse_code] = profile_cache[h.horse_code]

        odds = fetch_odds(
            race.race_date,
            race.venue_code,
            race.race_number,
            [h.horse_number for h in race.horses],
        )
        pred = predictor.predict_race_from_components(
            race=race,
            horse_profiles=horse_profiles,
            odds=odds,
            elo=elo,
            jt_stats=jt_stats,
            draw_stats=draw_stats,
        )

        # Persist prediction snapshot so maintenance can settle results later.
        backtester.record_prediction(race_id, pred, race.race_date)

        ok = notifier._run_async_sync(notifier.send_race_preview(race, pred))
        if ok:
            sent_ids.add(race_id)
            sent += 1

    _save_tick_sent_state(state_path, today, sent_ids)
    logger.info(f"Tick complete: total_races={len(races)} eligible={eligible} sent={sent}")


def run_maintenance() -> None:
    """Run daily backtest + optimize and send one Telegram summary message."""
    logger.info("=== Maintenance Mode ===")

    from model.backtest import Backtester
    from model.self_optimizer import StrategySelfOptimizer
    from notifier.telegram import TelegramNotifier

    state_path = config.CRON_STATE_FILE
    state = _load_json_state(state_path)
    today_str = date.today().strftime("%Y-%m-%d")
    alert_noise_summary = _get_daily_alert_noise_summary(today_str)

    sync_info = _sync_recent_real_results(days_back=2)
    retention_info = _apply_retention_policy() if getattr(config, "ENABLE_RETENTION_CLEANUP", True) else {
        "skipped": True,
        "reason": "disabled_by_config",
    }
    if sync_info.get("updated", 0):
        logger.info(
            "Maintenance synced %s recent results for dates=%s",
            sync_info.get("updated", 0),
            ",".join(sync_info.get("checked_dates", [])),
        )

    bt = Backtester(repair_missing=True)
    summary = bt.get_summary_stats()
    r30 = bt.calculate_roi("top1_win", 30, real_only=True)

    settled_real = bt.get_settled_records(real_only=True)
    training_manifest = _load_training_retention_manifest()
    retrain_candidates, retention_status = _evaluate_training_retention_status(settled_real, training_manifest)
    latest_marker = _latest_settled_marker(settled_real)
    last_notified_marker = str(state.get("maintenance_last_notified_marker", "") or "")
    has_new_settled = bool(latest_marker and latest_marker != last_notified_marker)

    if retention_status.get("ready_for_retrain", False):
        retrain_info = _run_daily_retrain(retrain_candidates, state)
    else:
        retrain_info = {
            "attempted": False,
            "ran": False,
            "reason": "blocked_by_training_retention_policy",
            "new_settled_since_last": 0,
            "target_real_rows": 0,
            "metrics": {},
        }
    retrain_info["quality_gate"] = {
        "ready": bool(retention_status.get("ready_for_retrain", False)),
        "allowed_tiers": list(retention_status.get("allowed_tiers", [])),
        "blocking_reasons": list(retention_status.get("blocking_reasons", [])),
        "counts_by_tier": dict(retention_status.get("counts_by_tier", {})),
        "eligible_records": int(retention_status.get("eligible_records", 0) or 0),
    }

    optimized = False
    profile = None
    segmented_profile = {}
    walk_forward_report = {}
    if settled_real:
        lookback_days = max(30, int(config.OPTIMIZATION_LOOKBACK_DAYS))
        cutoff = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        recent = [r for r in settled_real if r.race_date >= cutoff]
        optimizer = StrategySelfOptimizer(config.STRATEGY_PROFILE_PATH)
        profile = optimizer.optimize(recent, min_samples=24)
        segmented_profile = optimizer.optimize_segmented(recent, min_samples=12)
        optimized = True

    walk_forward_report = bt.build_walk_forward_report(
        lookback_days=max(120, int(config.OPTIMIZATION_LOOKBACK_DAYS)),
        train_days=60,
        test_days=14,
        step_days=7,
        real_only=True,
        min_train_samples=24,
    )

    walk_forward_path = os.path.join(
        config.MAINTENANCE_REPORT_DIR,
        f"walkforward_{date.today().strftime('%Y-%m-%d')}.json",
    )
    os.makedirs(config.MAINTENANCE_REPORT_DIR, exist_ok=True)
    with open(walk_forward_path, "w", encoding="utf-8") as f:
        json.dump(walk_forward_report, f, ensure_ascii=False, indent=2)

    latest_compare = None
    latest_day_comparisons = []
    if settled_real:
        latest_record = max(
            settled_real,
            key=lambda r: (
                str(getattr(r, "result_recorded_at", "") or ""),
                str(getattr(r, "race_date", "") or ""),
                int(getattr(r, "race_number", 0) or 0),
            ),
        )
        latest_compare = {
            "race_id": latest_record.race_id,
            "predicted_top3": latest_record.predicted_top3,
            "actual_top3": latest_record.actual_result[:3],
            "winner_correct": latest_record.winner_correct,
            "top3_hit": latest_record.top3_hit,
            "trio_hit": latest_record.trio_hit,
        }

        latest_day = str(getattr(latest_record, "race_date", "") or "")
        day_records = [r for r in settled_real if str(getattr(r, "race_date", "") or "") == latest_day]
        day_records.sort(key=lambda r: int(getattr(r, "race_number", 0) or 0))
        for r in day_records:
            latest_day_comparisons.append({
                "race_id": r.race_id,
                "race_number": int(getattr(r, "race_number", 0) or 0),
                "predicted_top3": list(getattr(r, "predicted_top3", []) or [])[:3],
                "actual_top3": list(getattr(r, "actual_result", []) or [])[:3],
                "winner_correct": bool(getattr(r, "winner_correct", False)),
                "top3_hit": int(getattr(r, "top3_hit", 0) or 0),
                "trio_hit": bool(getattr(r, "trio_hit", False)),
                "predicted_winner_odds": float(getattr(r, "predicted_winner_odds", 0.0) or 0.0),
                "model_confidence": float(getattr(r, "model_confidence", 0.0) or 0.0),
            })

    report = {
        "date": today_str,
        "generated_at": datetime.now(HKT).isoformat(),
        "result_sync": sync_info,
        "retention_cleanup": retention_info,
        "training_retention": retention_status,
        "alert_noise_summary": alert_noise_summary,
        "has_new_settled": has_new_settled,
        "latest_settled_marker": latest_marker,
        "summary": {
            "total_predictions": summary.get("total_predictions", 0),
            "winner_accuracy": summary.get("winner_accuracy", 0),
            "top3_hit_rate": summary.get("top3_hit_rate", 0),
            "r30_real_roi": r30.win_roi,
            "r30_real_races": r30.total_races,
        },
        "optimization": {
            "ran": optimized,
            "profile": {
                "min_confidence": getattr(profile, "min_confidence", None),
                "max_win_odds": getattr(profile, "max_win_odds", None),
                "selected_samples": getattr(profile, "selected_samples", None),
                "selected_win_rate": getattr(profile, "selected_win_rate", None),
                "selected_roi": getattr(profile, "selected_roi", None),
            },
            "segmented": segmented_profile,
        },
        "walk_forward": {
            "summary": walk_forward_report.get("summary", {}),
            "report_path": walk_forward_path,
        },
        "prediction_vs_actual": {
            "latest_day_count": len(latest_day_comparisons),
            "latest_day_records": latest_day_comparisons,
        },
        "retrain": retrain_info,
    }
    report_path = _write_maintenance_report(report)
    audit_path = _write_authenticity_audit_report(bt)
    report_path_display = report_path.replace("\\", "/")
    audit_path_display = audit_path.replace("\\", "/")
    walk_forward_path_display = walk_forward_path.replace("\\", "/")
    state["maintenance_last_report_path"] = report_path
    state["maintenance_last_run_at"] = datetime.now(HKT).isoformat()

    notifier = TelegramNotifier()
    if not notifier.is_configured():
        logger.warning("Telegram not configured; maintenance summary not sent")
        _save_json_state(state_path, state)
        return

    noisy_count = int(alert_noise_summary.get("total_suppressed", 0) or 0)
    if config.MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED and not has_new_settled and noisy_count <= 0:
        logger.info("No new settled real results; skip maintenance Telegram summary")
        _save_json_state(state_path, state)
        return

    date_str = today_str
    is_pro_style = (config.MESSAGE_STYLE or "pro").strip().lower() != "casual"
    lines = [
        f"{'🛠️✨ <b>每日維護摘要</b>' if is_pro_style else '🛠️ <b>每日更新</b>'} - {date_str}",
        "",
        f"{'<b>📌 回測摘要</b>' if is_pro_style else '<b>📌 今日回測重點</b>'}",
        f"• 🧾 總預測場數: {summary.get('total_predictions', 0)}",
        f"• 🎯 首選準確率: {summary.get('winner_accuracy', 0):.1%}",
        f"• 🏇 頭三到位率: {summary.get('top3_hit_rate', 0):.1%}",
        f"• 📈 30日真實 ROI: {r30.win_roi:+.2f}% ({r30.total_races}場)",
        f"• 🆕 新結算賽果: {'有' if has_new_settled else '無'}",
    ]

    if latest_compare:
        pred_top3 = " / ".join(f"#{x}" for x in latest_compare["predicted_top3"]) or "N/A"
        actual_top3 = " / ".join(f"#{x}" for x in latest_compare["actual_top3"]) or "N/A"
        lines.extend([
            "",
            f"{'<b>🆚 預測與結果對照（最近一場）</b>' if is_pro_style else '<b>🆚 最近一場對比</b>'}",
            f"• 場次: {html.escape(str(latest_compare['race_id']))}",
            f"• 🤖 預測頭三: {pred_top3}",
            f"• 🏁 實際頭三: {actual_top3}",
            f"• ✅ 命中數: {latest_compare['top3_hit']}/3",
            f"• {'🥇 首選命中' if latest_compare['winner_correct'] else '🥈 首選未中'}",
            f"• {'🎉 三T命中' if latest_compare['trio_hit'] else '📍 三T未命中'}",
        ])

    if latest_day_comparisons:
        day_winner_hits = sum(1 for x in latest_day_comparisons if x["winner_correct"])
        day_top3_hits = sum(int(x["top3_hit"]) for x in latest_day_comparisons)
        day_trio_hits = sum(1 for x in latest_day_comparisons if x["trio_hit"])
        lines.extend([
            "",
            f"{'<b>📒 當日整體對照</b>' if is_pro_style else '<b>📒 當日整體比較</b>'}",
            f"• 場數: {len(latest_day_comparisons)}",
            f"• 首選命中: {day_winner_hits}",
            f"• 頭三命中總數: {day_top3_hits}",
            f"• 三T命中場數: {day_trio_hits}",
        ])

    if optimized and profile is not None:
        lines.extend([
            "",
            f"{'<b>🧠 優化摘要</b>' if is_pro_style else '<b>🧠 參數調整結果</b>'}",
            f"• min_confidence: {profile.min_confidence:.2f}",
            f"• max_win_odds: {profile.max_win_odds:.1f}",
            f"• selected_samples: {profile.selected_samples}",
            f"• selected_win_rate: {profile.selected_win_rate:.1%}",
            f"• selected_roi: {profile.selected_roi:+.2f}%",
        ])
    else:
        lines.extend([
            "",
            f"{'<b>🧠 優化摘要</b>' if is_pro_style else '<b>🧠 參數調整結果</b>'}",
            "• 目前沒有足夠已結算真實賽果，已略過優化。",
        ])

    lines.extend([
        "",
        f"{'<b>🔁 再訓練摘要</b>' if is_pro_style else '<b>🔁 今日再訓練</b>'}",
        f"• 是否執行: {'是' if retrain_info.get('ran') else '否'}",
        f"• 原因: {html.escape(str(retrain_info.get('reason', '') or ''))}",
        f"• 新增已結算場數: {retrain_info.get('new_settled_since_last', 0)}",
    ])

    gate = retrain_info.get("quality_gate", {})
    if gate:
        tier_counts = gate.get("counts_by_tier", {})
        lines.extend([
            f"• 可訓練等級: {html.escape(','.join(gate.get('allowed_tiers', [])) or 'N/A')}",
            f"• 高品質樣本: {gate.get('eligible_records', 0)}",
            f"• A/B/C/D: {tier_counts.get('A', 0)}/{tier_counts.get('B', 0)}/{tier_counts.get('C', 0)}/{tier_counts.get('D', 0)}",
        ])
        if gate.get("blocking_reasons"):
            lines.append(
                f"• 阻擋原因: {html.escape('; '.join(gate.get('blocking_reasons', [])[:2]))}"
            )

    if bool(getattr(config, "ENABLE_DAILY_ALERT_NOISE_SUMMARY", True)):
        lines.extend([
            "",
            f"{'<b>🔕 告警抑制噪音摘要</b>' if is_pro_style else '<b>🔕 告警噪音摘要</b>'}",
            f"• 今日抑制總數: {alert_noise_summary.get('total_suppressed', 0)}",
            f"• 指紋數量: {alert_noise_summary.get('unique_fingerprints', 0)}",
        ])
        for row in list(alert_noise_summary.get("top_noisy", []) or [])[:3]:
            lines.append(
                f"• {html.escape(str(row.get('fingerprint', '') or ''))} "
                f"[{html.escape(str(row.get('mode', '') or ''))}] "
                f"{html.escape(str(row.get('error_type', '') or ''))}: "
                f"{row.get('suppressed_count', 0)} 次"
            )

    metrics = retrain_info.get("metrics", {})
    if metrics:
        lines.extend([
            f"• ensemble_auc_train: {metrics.get('ensemble_auc_train', 0):.4f}",
            f"• xgb_auc_train: {metrics.get('xgb_auc_train', 0):.4f}",
            f"• lgb_auc_train: {metrics.get('lgb_auc_train', 0):.4f}",
        ])

    lines.extend([
        "",
        f"{'<b>🗂️ 分析報告</b>' if is_pro_style else '<b>🗂️ 今日報告</b>'}\n• {html.escape(report_path_display)}",
        f"• walk-forward: {html.escape(walk_forward_path_display)}",
        f"• authenticity-audit: {html.escape(audit_path_display)}",
    ])

    lines.extend([
        "",
        f"<i>{'⚠️ 僅供研究參考，不構成投注建議' if is_pro_style else '⚠️ 只供研究參考，唔係投注建議'}</i>",
    ])

    ok = notifier.send_sync("\n".join(lines))
    if not ok:
        logger.warning("Maintenance summary failed to send to Telegram")
        _save_json_state(state_path, state)
        return

    if has_new_settled:
        state["maintenance_last_notified_marker"] = latest_marker
    _save_json_state(state_path, state)
    logger.info("Maintenance summary sent to Telegram")


def run_cron() -> None:
    """
    Unified mode for platforms that only allow one cron command.

    Behavior:
      - Always run tick logic.
      - Run maintenance once per day inside configured HKT window.
    """
    logger.info("=== Cron Mode ===")
    now_hkt = datetime.now(HKT)
    today = now_hkt.strftime("%Y-%m-%d")
    state_path = config.CRON_STATE_FILE

    run_tick()

    state = _load_json_state(state_path)
    if _should_send_heartbeat_now(now_hkt):
        heartbeat_sent = _send_cron_heartbeat(now_hkt, state)
        if heartbeat_sent:
            _save_json_state(state_path, state)

    if not _should_run_maintenance_now(now_hkt):
        logger.info("Outside maintenance window; cron exits after tick")
        return

    maint_map = state.get("maintenance_sent", {})
    if not isinstance(maint_map, dict):
        maint_map = {}

    if maint_map.get(today):
        logger.info("Maintenance already executed today; skipping")
        return

    run_maintenance()

    # Reload latest state written by maintenance to avoid overwriting fields.
    state = _load_json_state(state_path)
    maint_map = state.get("maintenance_sent", {})
    if not isinstance(maint_map, dict):
        maint_map = {}
    maint_map[today] = True

    # Keep state compact to recent 14 days.
    keys = sorted(maint_map.keys())
    if len(keys) > 14:
        for k in keys[:-14]:
            maint_map.pop(k, None)

    state["maintenance_sent"] = maint_map
    _save_json_state(state_path, state)
    logger.info("Maintenance marked as completed for today")


def run_retention_cleanup() -> None:
    """Run artifact retention cleanup once and print summary."""
    summary = _apply_retention_policy()
    print("\nRetention cleanup summary")
    print(f"- policy_path: {summary.get('policy_path', '')}")
    print(f"- scanned_files: {summary.get('scanned_files', 0)}")
    print(f"- deleted_files: {summary.get('deleted_files', 0)}")
    print(f"- errors: {len(summary.get('errors', []))}\n")


def _send_mode_report_to_telegram(mode: str, title: str, body_lines: List[str]) -> bool:
    """Best-effort helper to send mode summary to Telegram."""
    from notifier.telegram import TelegramNotifier

    notifier = TelegramNotifier()
    if not notifier.is_configured():
        logger.warning("Telegram not configured; %s summary not sent", mode)
        return False

    is_pro_style = (config.MESSAGE_STYLE or "pro").strip().lower() != "casual"
    footer = (
        "⚠️ 僅供研究參考，不構成投注建議"
        if is_pro_style
        else "⚠️ 只供研究參考，唔係投注建議"
    )
    now_hkt_str = datetime.now(HKT).strftime("%Y-%m-%d %H:%M:%S")
    message = "\n".join([
        f"<b>{title}</b>",
        f"時間: {now_hkt_str} HKT",
        "",
        *body_lines,
        "",
        f"<i>{footer}</i>",
    ])

    ok = notifier.send_sync(message)
    if ok:
        logger.info("%s summary sent to Telegram", mode)
    else:
        logger.warning("Failed to send %s summary to Telegram", mode)
    return ok


def run_backtest() -> None:
    """Print backtest statistics."""
    logger.info("=== Backtest Mode ===")
    from model.backtest import Backtester

    bt = Backtester()
    stats = bt.get_summary_stats()
    real_stats = bt.get_summary_stats(real_only=True)

    r7 = bt.calculate_roi("top1_win", 7)
    r30 = bt.calculate_roi("top1_win", 30)
    r30_trio = bt.calculate_roi("trio", 30)
    real_r30 = bt.calculate_roi("top1_win", 30, real_only=True)

    print("\n" + "="*60)
    print("  HKJC 賽馬預測系統 - 回測成績報告")
    print("="*60)
    print(f"  總預測場數:    {stats['total_predictions']}")
    print(f"  首選準確率:    {stats['winner_accuracy']:.1%}")
    print(f"  頭三到位率:    {stats['top3_hit_rate']:.1%}")
    print(f"  三T命中率:     {stats['trio_hit_rate']:.1%}")
    print(f"  真實數據場數:  {real_stats['total_predictions']}")
    print(f"  真實數據勝率:  {real_stats['winner_accuracy']:.1%}")
    print(f"")
    print(f"  【近7日】")
    print(f"    場數:      {r7.total_races}")
    print(f"    ROI:       {r7.win_roi:+.1f}%")
    print(f"")
    print(f"  【近30日】")
    print(f"    場數:      {r30.total_races}")
    print(f"    勝出注 ROI: {r30.win_roi:+.1f}%")
    print(f"    三T ROI:   {r30_trio.trio_roi:+.1f}%")
    print(f"    淨利潤:    ${r30.net_profit:+.0f}")
    print(f"    真實ROI:   {real_r30.win_roi:+.1f}%")
    print("="*60 + "\n")

    _send_mode_report_to_telegram(
        mode="backtest",
        title="📊 回測模式摘要",
        body_lines=[
            f"• 總預測場數: {stats['total_predictions']}",
            f"• 首選準確率: {stats['winner_accuracy']:.1%}",
            f"• 頭三到位率: {stats['top3_hit_rate']:.1%}",
            f"• 三T命中率: {stats['trio_hit_rate']:.1%}",
            f"• 近7日 ROI: {r7.win_roi:+.1f}% ({r7.total_races}場)",
            f"• 近30日 勝出注 ROI: {r30.win_roi:+.1f}% ({r30.total_races}場)",
            f"• 近30日 三T ROI: {r30_trio.trio_roi:+.1f}%",
            f"• 近30日 淨利潤: ${r30.net_profit:+.0f}",
            f"• 近30日 真實 ROI: {real_r30.win_roi:+.1f}% ({real_r30.total_races}場)",
        ],
    )


def run_optimize() -> None:
    """Run real-data backtest and optimize adaptive recommendation thresholds."""
    logger.info("=== Optimize Mode ===")
    from config import config
    from model.backtest import Backtester
    from model.self_optimizer import StrategySelfOptimizer

    bt = Backtester(repair_missing=True)
    settled_real = bt.get_settled_records(real_only=True)
    if not settled_real:
        print("\n目前沒有已結算的真實賽果，無法進行自我優化。\n")
        _send_mode_report_to_telegram(
            mode="optimize",
            title="🧠 優化模式摘要",
            body_lines=[
                "• 狀態: 未執行",
                "• 原因: 目前沒有已結算真實賽果",
            ],
        )
        return

    lookback_days = max(30, int(config.OPTIMIZATION_LOOKBACK_DAYS))
    cutoff = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent = [r for r in settled_real if r.race_date >= cutoff]

    optimizer = StrategySelfOptimizer(config.STRATEGY_PROFILE_PATH)
    profile = optimizer.optimize(recent, min_samples=24)

    real_30 = bt.calculate_roi("top1_win", 30, real_only=True)
    print("\n" + "="*70)
    print("  HKJC 賽馬預測系統 - 自我優化結果 (真實數據)")
    print("="*70)
    print(f"  近30日真實 ROI:        {real_30.win_roi:+.2f}%")
    print(f"  近30日真實回測場數:    {real_30.total_races}")
    print(f"  優化後最小信心值:      {profile.min_confidence:.2f}")
    print(f"  優化後最大勝出賠率:    {profile.max_win_odds:.1f}")
    print(f"  優化樣本數:            {profile.selected_samples}")
    print(f"  優化樣本勝率:          {profile.selected_win_rate:.1%}")
    print(f"  優化樣本 ROI:          {profile.selected_roi:+.2f}%")
    print(f"  策略檔案:              {config.STRATEGY_PROFILE_PATH}")
    print("="*70 + "\n")

    _send_mode_report_to_telegram(
        mode="optimize",
        title="🧠 優化模式摘要",
        body_lines=[
            f"• 近30日真實 ROI: {real_30.win_roi:+.2f}% ({real_30.total_races}場)",
            f"• 最小信心值: {profile.min_confidence:.2f}",
            f"• 最大勝出賠率: {profile.max_win_odds:.1f}",
            f"• 優化樣本數: {profile.selected_samples}",
            f"• 優化樣本勝率: {profile.selected_win_rate:.1%}",
            f"• 優化樣本 ROI: {profile.selected_roi:+.2f}%",
            f"• 策略檔案: {config.STRATEGY_PROFILE_PATH}",
        ],
    )


def run_fetch() -> None:
    """Fetch and display today's racecard and odds."""
    logger.info("=== Fetch Mode ===")
    today = datetime.now(HKT).strftime("%Y-%m-%d")

    from scraper.racecard import fetch_racecard
    from scraper.odds import fetch_odds
    from scraper.weather import fetch_weather

    races = fetch_racecard(today)
    weather = fetch_weather(races[0].venue_code if races else "ST")

    print(f"\n今日賽事 ({today})")
    print(f"場地狀況: {weather.going} | 溫度: {weather.temperature}°C")
    print(f"共 {len(races)} 場")

    for race in races:
        odds = fetch_odds(race.race_date, race.venue_code, race.race_number,
                          [h.horse_number for h in race.horses])
        print(f"\n第{race.race_number}場 {race.start_time} | {race.distance}m {race.race_class}")
        for horse in race.horses:
            h_odds = odds.get(horse.horse_number, {})
            print(f"  #{horse.horse_number} {horse.horse_name:<24} "
                  f"檔:{horse.draw}  勝賠:{h_odds.get('win_odds', '-')}")


def run_train() -> None:
    """Train / retrain the model."""
    logger.info("=== Training Mode ===")
    from model.trainer import EnsembleTrainer

    trainer = EnsembleTrainer()
    logger.info("Generating synthetic training data...")
    df = trainer.generate_synthetic_data(n_races=1000)
    logger.info(f"Generated {len(df)} samples")

    metrics = trainer.train(df)
    logger.info(f"Training complete: {metrics}")

    trainer.save()
    logger.info("Model saved")

    fi = trainer.get_feature_importance()
    print("\nTop-15 Feature Importances:")
    print(fi.head(15).to_string(index=False))


def run_telegram_test() -> None:
    """Send Telegram template test messages (pre-race, result, daily summary)."""
    logger.info("=== Telegram Test Mode ===")
    import asyncio
    from notifier.telegram import TelegramNotifier

    notifier = TelegramNotifier()
    if not notifier.is_configured():
        print("\nTelegram 未設定，請先在 .env 設定 TELEGRAM_TOKEN 和 TELEGRAM_CHAT_ID。\n")
        return

    class _Horse:
        def __init__(self, horse_number, horse_name, confidence, is_value_bet, win_prob, implied_win_prob):
            self.horse_number = horse_number
            self.horse_name = horse_name
            self.confidence = confidence
            self.is_value_bet = is_value_bet
            self.win_prob = win_prob
            self.implied_win_prob = implied_win_prob

    class _Prediction:
        def __init__(self):
            self.top5 = [
                _Horse(5, "GOLDEN ARROW", 0.72, True, 0.38, 0.18),
                _Horse(2, "SUPER STAR", 0.63, False, 0.27, 0.20),
                _Horse(9, "FLASH KING", 0.58, False, 0.19, 0.11),
                _Horse(1, "SKY DASH", 0.42, False, 0.10, 0.09),
                _Horse(7, "SILVER WIND", 0.39, False, 0.06, 0.07),
            ]
            self.trio_suggestions = [(5, 2, 9)]
            self.best_ev_horse = self.top5[0]
            self.confidence = 0.72

    class _Race:
        def __init__(self):
            self.race_number = 6
            self.venue_code = "ST"
            self.venue = "Sha Tin"
            self.start_time = "15:35"
            self.distance = 1200
            self.race_class = "Class 3"
            self.track_type = "Turf"
            self.going = "GOOD"
            self.horses = list(range(1, 13))

    async def _send_all() -> tuple[bool, bool, bool]:
        race = _Race()
        pred = _Prediction()
        ok_preview = await notifier.send_race_preview(race, pred)
        ok_result = await notifier.send_race_result(race, [5, 2, 9, 1, 7], pred)
        ok_daily = await notifier.send_daily_summary(
            date.today().strftime("%Y-%m-%d"),
            {
                "races_30d": 88,
                "winner_accuracy": 0.364,
                "top3_hit_rate": 0.614,
                "roi_7d": 346.0,
                "roi_30d": 214.9,
                "net_profit_30d": 1891,
            },
        )
        return ok_preview, ok_result, ok_daily

    preview_ok, result_ok, daily_ok = asyncio.run(_send_all())

    print("\n" + "=" * 60)
    print("  Telegram 模板測試結果")
    print("=" * 60)
    print(f"  賽前通知:   {'✅ 成功' if preview_ok else '❌ 失敗'}")
    print(f"  賽後通知:   {'✅ 成功' if result_ok else '❌ 失敗'}")
    print(f"  每日總結:   {'✅ 成功' if daily_ok else '❌ 失敗'}")
    print("=" * 60 + "\n")


def run_authenticity_audit() -> None:
    """Generate authenticity audit report from current settled history."""
    from model.backtest import Backtester

    bt = Backtester(repair_missing=True)
    path = _write_authenticity_audit_report(bt)
    print(f"\n真實性稽核報表已輸出: {path}\n")


def run_migrate_provenance() -> None:
    """One-time migration: backfill legacy source provenance to C/D with rule notes."""
    from model.backtest import Backtester

    bt = Backtester(repair_missing=True)
    changed = 0
    c_count = 0
    d_count = 0
    skipped = 0

    for rec in bt._records.values():
        existing_src = str(getattr(rec, "result_source", "") or "").strip()
        existing_conf = str(getattr(rec, "result_source_confidence", "") or "").strip().upper()
        if existing_src and existing_conf in {"A", "B", "C", "D"}:
            skipped += 1
            continue

        is_settled = bool(getattr(rec, "actual_winner", 0) and getattr(rec, "actual_result", []))
        has_recorded_at = bool(str(getattr(rec, "result_recorded_at", "") or "").strip())
        is_real = bool(getattr(rec, "is_real_result", False))

        if is_real and is_settled and has_recorded_at:
            rec.result_source = "legacy_migrated_settled_record"
            rec.result_source_url = ""
            rec.result_source_confidence = "C"
            rec.result_source_note = (
                "rule:C legacy settled record with result_recorded_at; "
                "source not preserved in historical schema"
            )
            c_count += 1
        else:
            rec.result_source = "legacy_unverified"
            rec.result_source_url = ""
            rec.result_source_confidence = "D"
            rec.result_source_note = (
                "rule:D legacy record lacks sufficient source evidence "
                "for external verification"
            )
            d_count += 1
        changed += 1

    bt.save_history()

    report_dir = config.MAINTENANCE_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"provenance_migration_{datetime.now(HKT).strftime('%Y-%m-%d')}.json")
    payload = {
        "generated_at": datetime.now(HKT).isoformat(),
        "mode": "one_time_provenance_migration",
        "summary": {
            "total_records": len(bt._records),
            "changed": changed,
            "assigned_C": c_count,
            "assigned_D": d_count,
            "skipped_with_existing_provenance": skipped,
        },
        "rules": {
            "C": "is_real_result=true AND settled AND result_recorded_at present",
            "D": "all other legacy records lacking sufficient source evidence",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    audit_path = _write_authenticity_audit_report(bt)
    print("\n來源溯源一次性遷移完成")
    print(f"- 變更筆數: {changed}")
    print(f"- C 級: {c_count}")
    print(f"- D 級: {d_count}")
    print(f"- 保留既有標註: {skipped}")
    print(f"- 遷移報表: {path}")
    print(f"- 稽核報表: {audit_path}\n")


def run_send_all_previews_now() -> None:
    """Force-send pre-race preview Telegram messages for all races in current/next meeting."""
    from scraper.racecard import fetch_racecard
    from scraper.horse_profile import fetch_horse_profile
    from scraper.odds import fetch_odds
    from features.elo import ELOSystem
    from features.jockey_trainer import JockeyTrainerStats
    from features.draw import DrawStats
    from model.trainer import EnsembleTrainer
    from model.predictor import RacePredictor
    from notifier.telegram import TelegramNotifier

    today = datetime.now(HKT).strftime("%Y-%m-%d")
    races = fetch_racecard(today)
    if not races:
        print("\n找不到可用賽事，未發送 Telegram。\n")
        return

    elo = ELOSystem(k_factor=config.ELO_K_FACTOR, base_rating=config.ELO_BASE)
    elo.load_ratings("data/processed/elo_ratings.json")

    jt_stats = JockeyTrainerStats()
    jt_stats.load()

    draw_stats = DrawStats()
    if not draw_stats.load() and not config.REAL_DATA_ONLY:
        draw_stats.bootstrap_from_mock()

    trainer = EnsembleTrainer()
    if not trainer.load():
        raise RuntimeError("模型未載入，請先訓練模型後再發送賽前預測")

    predictor = RacePredictor(trainer=trainer, top_n=config.TOP_N)
    notifier = TelegramNotifier()
    if not notifier.is_configured():
        print("\nTelegram 未設定，未發送。\n")
        return

    total = len(races)
    sent = 0
    blocked = 0
    profile_cache = {}

    for race in races:
        horse_profiles = {}
        for h in race.horses:
            if h.horse_code not in profile_cache:
                profile_cache[h.horse_code] = fetch_horse_profile(h.horse_code)
            horse_profiles[h.horse_code] = profile_cache[h.horse_code]

        odds = fetch_odds(
            race.race_date,
            race.venue_code,
            race.race_number,
            [h.horse_number for h in race.horses],
        )
        pred = predictor.predict_race_from_components(
            race=race,
            horse_profiles=horse_profiles,
            odds=odds,
            elo=elo,
            jt_stats=jt_stats,
            draw_stats=draw_stats,
        )
        ok = notifier._run_async_sync(notifier.send_race_preview(race, pred))
        if ok:
            sent += 1
        else:
            blocked += 1

    print("\n已執行全場賽前推送")
    print(f"- meeting_date: {races[0].race_date}")
    print(f"- total_races: {total}")
    print(f"- sent: {sent}")
    print(f"- blocked_or_failed: {blocked}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HKJC 賽馬預測系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "web",
            "debug",
            "cron",
            "tick",
            "maintenance",
            "retention-cleanup",
            "authenticity-audit",
            "migrate-provenance",
            "send-all-previews-now",
            "predict",
            "backtest",
            "fetch",
            "train",
            "demo",
            "optimize",
            "telegram-test",
        ],
        default="web",
        help="執行模式 (default: web)",
    )
    parser.add_argument("--port", type=int, default=None, help="Flask port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable scheduler")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    logger.info(f"HKJC 賽馬預測系統 starting in [{args.mode.upper()}] mode")
    logger.info(f"Demo mode: {'ON' if config.DEMO_MODE else 'OFF (real data)'}")

    try:
        if args.mode == "demo":
            config.DEMO_MODE = True
            config.REAL_DATA_ONLY = False
            config.ALLOW_SYNTHETIC_TRAINING = True
            logger.info("Forced DEMO_MODE=True, REAL_DATA_ONLY=False, ALLOW_SYNTHETIC_TRAINING=True")
            run_web(port=args.port, debug=args.debug, with_scheduler=not args.no_scheduler)

        elif args.mode == "web":
            run_web(port=args.port, debug=args.debug, with_scheduler=not args.no_scheduler)

        elif args.mode == "debug":
            run_debug()

        elif args.mode == "cron":
            run_cron()

        elif args.mode == "tick":
            run_tick()

        elif args.mode == "maintenance":
            run_maintenance()

        elif args.mode == "retention-cleanup":
            run_retention_cleanup()

        elif args.mode == "authenticity-audit":
            run_authenticity_audit()

        elif args.mode == "migrate-provenance":
            run_migrate_provenance()

        elif args.mode == "send-all-previews-now":
            run_send_all_previews_now()

        elif args.mode == "predict":
            run_predict()

        elif args.mode == "backtest":
            run_backtest()

        elif args.mode == "fetch":
            run_fetch()

        elif args.mode == "train":
            run_train()

        elif args.mode == "optimize":
            run_optimize()

        elif args.mode == "telegram-test":
            run_telegram_test()
    except Exception as e:
        logger.exception("Run failed in mode=%s", args.mode)
        _send_failure_alert(args.mode, e)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
