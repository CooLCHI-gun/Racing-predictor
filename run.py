#!/usr/bin/env python3
"""
HKJC 賽馬預測系統 - 主程式入口
Main entry point for the HKJC Horse Racing Prediction System.

Usage:
    python run.py                    # Start Flask + scheduler (default)
    python run.py --mode web         # Flask web dashboard only
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
import logging
import os
import sys
from datetime import date, timedelta

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


# ── Mode Handlers ──────────────────────────────────────────────────────────────

def run_web(port: int = None, debug: bool = None, with_scheduler: bool = True) -> None:
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
        choices=["web", "predict", "backtest", "fetch", "train", "demo", "optimize", "telegram-test"],
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
