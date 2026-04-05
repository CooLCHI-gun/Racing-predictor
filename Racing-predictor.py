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
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional
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
    today = date.today().strftime("%Y-%m-%d")

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
    today = date.today().strftime("%Y-%m-%d")
    now_hkt = datetime.now(HKT)
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


def _should_run_maintenance_now(now_hkt: datetime) -> bool:
    """Return True when current time falls into the configured daily maintenance window."""
    target_minutes = int(config.CRON_MAINTENANCE_HOUR) * 60 + int(config.CRON_MAINTENANCE_MINUTE)
    current_minutes = now_hkt.hour * 60 + now_hkt.minute
    window = max(1, int(config.CRON_MAINTENANCE_WINDOW_MINS))
    return target_minutes <= current_minutes < (target_minutes + window)


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


def _write_maintenance_report(report: dict) -> str:
    """Persist maintenance analysis report and return file path."""
    report_dir = config.MAINTENANCE_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    date_str = report.get("date", date.today().strftime("%Y-%m-%d"))
    path = os.path.join(report_dir, f"maintenance_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def _run_daily_retrain(settled_real_count: int, state: dict) -> dict:
    """Run daily retraining when new settled data is available and allowed by config."""
    retrain = {
        "attempted": False,
        "ran": False,
        "reason": "",
        "new_settled_since_last": 0,
        "target_synthetic_races": 0,
        "metrics": {},
    }

    if not config.DAILY_RETRAIN_ENABLED:
        retrain["reason"] = "disabled_by_config"
        return retrain

    last_settled_count = int(state.get("retrain_last_settled_count", 0) or 0)
    new_settled = max(0, settled_real_count - last_settled_count)
    retrain["new_settled_since_last"] = new_settled

    if new_settled < int(config.DAILY_RETRAIN_MIN_NEW_SETTLED):
        retrain["reason"] = "no_enough_new_settled"
        return retrain

    if config.REAL_DATA_ONLY and not config.ALLOW_SYNTHETIC_TRAINING:
        retrain["reason"] = "REAL_DATA_ONLY blocks synthetic retrain"
        return retrain

    from model.trainer import EnsembleTrainer

    retrain["attempted"] = True
    trainer = EnsembleTrainer()
    n_races = int(config.DAILY_RETRAIN_SYNTHETIC_RACES_BASE) + (
        settled_real_count * int(config.DAILY_RETRAIN_SYNTHETIC_RACES_SCALE)
    )
    n_races = max(300, min(2000, n_races))
    retrain["target_synthetic_races"] = n_races

    df = trainer.generate_synthetic_data(n_races=n_races)
    metrics = trainer.train(df=df, training_source="synthetic")
    trainer.save()

    retrain["ran"] = True
    retrain["metrics"] = metrics
    retrain["reason"] = "ok"
    state["retrain_last_settled_count"] = settled_real_count
    state["retrain_last_run_at"] = datetime.now(HKT).isoformat()
    return retrain


def run_tick() -> None:
    """
    Run one-shot cron cycle: fetch + predict + pre-race send, then exit.
    Designed for external schedulers (e.g. every 5 minutes).
    """
    logger.info("=== Tick Mode ===")
    today = date.today().strftime("%Y-%m-%d")
    now_hkt = datetime.now(HKT)

    from scraper.racecard import fetch_racecard
    from scraper.horse_profile import fetch_horse_profile
    from scraper.odds import fetch_odds
    from features.elo import ELOSystem
    from features.jockey_trainer import JockeyTrainerStats
    from features.draw import DrawStats
    from model.trainer import EnsembleTrainer
    from model.predictor import RacePredictor
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
            raise RuntimeError(
                "Model not found and synthetic training is disabled (REAL_DATA_ONLY=True)."
            )
        logger.info("Model missing; training fallback model for tick mode")
        trainer.train()
        trainer.save()

    predictor = RacePredictor(trainer=trainer, top_n=config.TOP_N)
    notifier = TelegramNotifier()
    notify_window = int(config.PRE_RACE_NOTIFY_MINS)

    state_path = config.TICK_SENT_STATE_FILE
    sent_ids = _load_tick_sent_state(state_path, today)

    profile_cache = {}
    eligible = 0
    sent = 0

    for race in races:
        try:
            hour, minute = map(int, race.start_time.split(":"))
            race_dt = datetime(now_hkt.year, now_hkt.month, now_hkt.day, hour, minute, tzinfo=HKT)
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

    bt = Backtester(repair_missing=True)
    summary = bt.get_summary_stats()
    r30 = bt.calculate_roi("top1_win", 30, real_only=True)

    settled_real = bt.get_settled_records(real_only=True)
    latest_marker = _latest_settled_marker(settled_real)
    last_notified_marker = str(state.get("maintenance_last_notified_marker", "") or "")
    has_new_settled = bool(latest_marker and latest_marker != last_notified_marker)

    retrain_info = _run_daily_retrain(len(settled_real), state)

    optimized = False
    profile = None
    if settled_real:
        lookback_days = max(30, int(config.OPTIMIZATION_LOOKBACK_DAYS))
        cutoff = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        recent = [r for r in settled_real if r.race_date >= cutoff]
        optimizer = StrategySelfOptimizer(config.STRATEGY_PROFILE_PATH)
        profile = optimizer.optimize(recent, min_samples=24)
        optimized = True

    report = {
        "date": date.today().strftime("%Y-%m-%d"),
        "generated_at": datetime.now(HKT).isoformat(),
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
        },
        "retrain": retrain_info,
    }
    report_path = _write_maintenance_report(report)
    report_path_display = report_path.replace("\\", "/")
    state["maintenance_last_report_path"] = report_path
    state["maintenance_last_run_at"] = datetime.now(HKT).isoformat()

    notifier = TelegramNotifier()
    if not notifier.is_configured():
        logger.warning("Telegram not configured; maintenance summary not sent")
        _save_json_state(state_path, state)
        return

    if config.MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED and not has_new_settled:
        logger.info("No new settled real results; skip maintenance Telegram summary")
        _save_json_state(state_path, state)
        return

    date_str = date.today().strftime("%Y-%m-%d")
    is_pro_style = (config.MESSAGE_STYLE or "pro").strip().lower() != "casual"
    latest_compare = None
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
            f"• 場次: {latest_compare['race_id']}",
            f"• 🤖 預測頭三: {pred_top3}",
            f"• 🏁 實際頭三: {actual_top3}",
            f"• ✅ 命中數: {latest_compare['top3_hit']}/3",
            f"• {'🥇 首選命中' if latest_compare['winner_correct'] else '🥈 首選未中'}",
            f"• {'🎉 三T命中' if latest_compare['trio_hit'] else '📍 三T未命中'}",
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
        f"• 原因: {retrain_info.get('reason', '')}",
        f"• 新增已結算場數: {retrain_info.get('new_settled_since_last', 0)}",
    ])

    metrics = retrain_info.get("metrics", {})
    if metrics:
        lines.extend([
            f"• ensemble_auc_train: {metrics.get('ensemble_auc_train', 0):.4f}",
            f"• xgb_auc_train: {metrics.get('xgb_auc_train', 0):.4f}",
            f"• lgb_auc_train: {metrics.get('lgb_auc_train', 0):.4f}",
        ])

    lines.extend([
        "",
        f"{'<b>🗂️ 分析報告</b>' if is_pro_style else '<b>🗂️ 今日報告</b>'}\n• {report_path_display}",
    ])

    lines.extend([
        "",
        f"<i>{'⚠️ 僅供研究參考，不構成投注建議' if is_pro_style else '⚠️ 只供研究參考，唔係投注建議'}</i>",
    ])

    notifier.send_sync("\n".join(lines))
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

    run_tick()

    if not _should_run_maintenance_now(now_hkt):
        logger.info("Outside maintenance window; cron exits after tick")
        return

    state_path = config.CRON_STATE_FILE
    state = _load_json_state(state_path)
    maint_map = state.get("maintenance_sent", {})
    if not isinstance(maint_map, dict):
        maint_map = {}

    if maint_map.get(today):
        logger.info("Maintenance already executed today; skipping")
        return

    run_maintenance()
    maint_map[today] = True

    # Keep state compact to recent 14 days.
    keys = sorted(maint_map.keys())
    if len(keys) > 14:
        for k in keys[:-14]:
            maint_map.pop(k, None)

    state["maintenance_sent"] = maint_map
    _save_json_state(state_path, state)
    logger.info("Maintenance marked as completed for today")


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


def run_fetch() -> None:
    """Fetch and display today's racecard and odds."""
    logger.info("=== Fetch Mode ===")
    today = date.today().strftime("%Y-%m-%d")

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


if __name__ == "__main__":
    main()
