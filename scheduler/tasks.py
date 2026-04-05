"""
HKJC 賽馬預測系統 - 排程任務
APScheduler jobs for automated data fetching, prediction, and notification.

Timezone: Asia/Hong_Kong (HKT = UTC+8)

Schedule:
  09:00 HKT - Fetch racecard (on race days)
  T-60min  - Start fetching odds every 5 min
  T-30min  - Send pre-race notification
  T+30min  - Fetch results and record
  21:00 HKT - Send daily summary
"""
import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import List, Optional

import pytz

logger = logging.getLogger(__name__)

HKT = pytz.timezone("Asia/Hong_Kong")

# HKJC race days (Wednesday evening + Saturday or Sunday)
# 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
RACE_WEEKDAYS = {2, 5, 6}  # Wed, Sat, Sun (approximate)


class SchedulerManager:
    """
    Manages APScheduler background jobs for the prediction system.

    Usage:
        manager = SchedulerManager(app_state)
        manager.start()
        # ... app runs ...
        manager.stop()
    """

    def __init__(self, app_state=None):
        """
        Initialise scheduler manager.

        Args:
            app_state: Flask app state object (_AppState from web/app.py)
        """
        self.app_state = app_state
        self.scheduler = None
        self._races = []
        self._predictions = {}

    def start(self) -> None:
        """Start the APScheduler with all configured jobs."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            self.scheduler = BackgroundScheduler(timezone=HKT)

            # ── Daily racecard fetch at 09:00 HKT ──────────────────────────
            self.scheduler.add_job(
                func=self.fetch_racecard_job,
                trigger=CronTrigger(hour=9, minute=0, timezone=HKT),
                id="fetch_racecard",
                name="Fetch Today's Racecard",
                replace_existing=True,
            )

            # ── Odds fetch every 5 minutes from 11:30 to 21:00 ────────────
            self.scheduler.add_job(
                func=self.fetch_odds_job,
                trigger=CronTrigger(
                    hour="11-21", minute="*/5",
                    timezone=HKT,
                ),
                id="fetch_odds",
                name="Fetch Live Odds",
                replace_existing=True,
            )

            # ── Pre-race notifications (30 min before each race) ───────────
            self.scheduler.add_job(
                func=self.schedule_pre_race_notifications,
                trigger=CronTrigger(hour=9, minute=30, timezone=HKT),
                id="schedule_notifications",
                name="Schedule Pre-Race Notifications",
                replace_existing=True,
            )

            # ── Daily summary at 21:00 HKT ─────────────────────────────────
            self.scheduler.add_job(
                func=self.send_daily_summary_job,
                trigger=CronTrigger(hour=21, minute=0, timezone=HKT),
                id="daily_summary",
                name="Send Daily Summary",
                replace_existing=True,
            )

            # ── Results fetch at 20:30 HKT ─────────────────────────────────
            self.scheduler.add_job(
                func=self.record_results_job,
                trigger=CronTrigger(hour=20, minute=30, timezone=HKT),
                id="fetch_results",
                name="Fetch Race Results",
                replace_existing=True,
            )

            # ── ELO auto-update at 21:30 HKT (after results settled) ─────────
            self.scheduler.add_job(
                func=self.elo_update_job,
                trigger=CronTrigger(hour=21, minute=30, timezone=HKT),
                id="elo_update",
                name="ELO Auto-Update (Post-Race)",
                replace_existing=True,
            )

            # ── Auto-retrain model weekly (Sunday 02:00 HKT) ─────────────────
            self.scheduler.add_job(
                func=self.auto_retrain_job,
                trigger=CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=HKT),
                id="auto_retrain",
                name="Weekly Model Retrain",
                replace_existing=True,
            )


            self.scheduler.start()
            logger.info("APScheduler started with all jobs")
            self._log_scheduled_jobs()

        except ImportError:
            logger.error("APScheduler not installed: pip install apscheduler==3.10.4")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}", exc_info=True)

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("APScheduler stopped")

    def _log_scheduled_jobs(self) -> None:
        """Log all scheduled jobs."""
        if not self.scheduler:
            return
        jobs = self.scheduler.get_jobs()
        logger.info(f"Scheduled {len(jobs)} jobs:")
        for job in jobs:
            logger.info(f"  - {job.name}: next run {job.next_run_time}")

    def _risk_guard_status(self) -> Optional[dict]:
        """Return risk guard status for today, or None if disabled/unavailable."""
        try:
            from config import config
            from model.backtest import Backtester

            if not config.ENABLE_RISK_GUARD:
                return None

            backtester = Backtester(repair_missing=True)
            return backtester.get_risk_guard_status(
                target_date=date.today().strftime("%Y-%m-%d"),
                consecutive_loss_limit=config.MAX_CONSECUTIVE_LOSSES,
                daily_loss_limit=config.MAX_DAILY_LOSS,
            )
        except Exception as e:
            logger.error(f"Failed to compute risk guard status: {e}")
            return None

    def _risk_guard_allows_notifications(self, race_number: int) -> bool:
        """Stop further recommendation-style actions after repeated losses."""
        status = self._risk_guard_status()
        if not status or not status.get("halt"):
            return True

        logger.warning(
            "Risk guard halted race %s actions: consecutive_losses=%s daily_net_profit=%s",
            race_number,
            status.get("consecutive_losses"),
            status.get("daily_net_profit"),
        )
        _send_alert(
            (
                f"風險控制已啟動，暫停第{race_number}場建議。"
                f"連輸={status.get('consecutive_losses')}，"
                f"今日淨損={status.get('daily_net_profit')}"
            ),
            "WARNING",
        )
        return False

    # ── Job Functions ──────────────────────────────────────────────────────

    def fetch_racecard_job(self) -> None:
        """
        Fetch today's racecard and generate predictions.
        Runs at 09:00 HKT on race days.
        """
        logger.info("=== fetch_racecard_job started ===")
        today = date.today()

        # Only run on race days
        if today.weekday() not in RACE_WEEKDAYS:
            logger.info(f"Not a race day ({today.strftime('%A')}); skipping")
            return

        try:
            from scraper.racecard import fetch_racecard
            self._races = fetch_racecard(today.strftime("%Y-%m-%d"))
            logger.info(f"Fetched {len(self._races)} races for {today}")

            # Trigger prediction generation
            self.run_predictions_job()

            # Schedule per-race notifications
            self.schedule_pre_race_notifications()

        except Exception as e:
            logger.error(f"fetch_racecard_job failed: {e}", exc_info=True)
            _send_alert(f"賽事表抓取失敗: {e}", "ERROR")

    def fetch_odds_job(self) -> None:
        """
        Fetch latest odds for all today's races.
        Runs every 5 min from 11:30-21:00 HKT on race days.
        """
        if not self._races:
            return

        logger.debug("=== fetch_odds_job ===")
        try:
            from config import config
            from scraper.odds import fetch_odds
            for race in self._races:
                new_odds = fetch_odds(
                    race.race_date, race.venue_code, race.race_number,
                    [h.horse_number for h in race.horses]
                )
                if self.app_state:
                    self.app_state.odds[race.race_number] = new_odds

            logger.debug(f"Odds updated for {len(self._races)} races")

            if config.ENABLE_LIVE_REPREDICT:
                logger.info("Fresh odds received; recomputing predictions")
                self.run_predictions_job()
        except Exception as e:
            logger.warning(f"fetch_odds_job error: {e}")

    def run_predictions_job(self) -> None:
        """
        Run prediction pipeline for all today's races.
        Called after racecard fetch and odds update.
        """
        if not self._races:
            logger.info("No races to predict")
            return

        logger.info("=== run_predictions_job ===")
        try:
            from features.elo import ELOSystem
            from features.jockey_trainer import JockeyTrainerStats
            from features.draw import DrawStats
            from model.trainer import EnsembleTrainer
            from model.predictor import RacePredictor
            from scraper.horse_profile import fetch_horse_profile
            from scraper.odds import fetch_odds
            from config import config

            elo = ELOSystem(k_factor=config.ELO_K_FACTOR, base_rating=config.ELO_BASE)
            elo.load_ratings("data/processed/elo_ratings.json")

            jt_stats = JockeyTrainerStats()
            jt_stats.load()

            draw_stats = DrawStats()
            if not draw_stats.load():
                if not config.REAL_DATA_ONLY:
                    draw_stats.bootstrap_from_mock()

            trainer = EnsembleTrainer()
            if not trainer.load():
                if config.REAL_DATA_ONLY and not config.ALLOW_SYNTHETIC_TRAINING:
                    raise RuntimeError(
                        "Model not found and synthetic training is disabled (REAL_DATA_ONLY=True)."
                    )
                trainer.train()
                trainer.save()

            predictor = RacePredictor(trainer=trainer, top_n=config.TOP_N)
            profile_cache = {}

            self._predictions = {}
            for race in self._races:
                horse_profiles = {}
                for h in race.horses:
                    if h.horse_code not in profile_cache:
                        profile_cache[h.horse_code] = fetch_horse_profile(h.horse_code)
                    horse_profiles[h.horse_code] = profile_cache[h.horse_code]
                odds = fetch_odds(
                    race.race_date, race.venue_code, race.race_number,
                    [h.horse_number for h in race.horses]
                )
                pred = predictor.predict_race_from_components(
                    race=race,
                    horse_profiles=horse_profiles,
                    odds=odds,
                    elo=elo,
                    jt_stats=jt_stats,
                    draw_stats=draw_stats,
                )
                self._predictions[race.race_number] = pred

                # Update app state if available
                if self.app_state:
                    self.app_state.predictions[race.race_number] = pred
                    self.app_state.odds[race.race_number] = odds

            logger.info(f"Generated {len(self._predictions)} predictions")

        except Exception as e:
            logger.error(f"run_predictions_job failed: {e}", exc_info=True)

    def schedule_pre_race_notifications(self) -> None:
        """
        Schedule pre-race refresh and notifications for each race.
        Should be called once per race day morning.
        """
        if not self._races or not self.scheduler:
            return

        from config import config
        notify_minutes = config.PRE_RACE_NOTIFY_MINS
        refresh_minutes = config.PRE_RACE_REFRESH_MINS
        today = date.today()

        for race in self._races:
            try:
                # Parse race start time
                hour, minute = map(int, race.start_time.split(":"))
                notify_dt = datetime(
                    today.year, today.month, today.day, hour, minute, 0,
                    tzinfo=HKT
                ) - timedelta(minutes=notify_minutes)

                if notify_dt <= datetime.now(HKT):
                    logger.debug(f"Skipping notification for race {race.race_number} (already past)")
                    continue

                race_num = race.race_number

                refresh_dt = datetime(
                    today.year, today.month, today.day, hour, minute, 0,
                    tzinfo=HKT
                ) - timedelta(minutes=refresh_minutes)

                if refresh_dt > datetime.now(HKT):
                    self.scheduler.add_job(
                        func=lambda rn=race_num: self.refresh_pre_race_data_job(rn),
                        trigger="date",
                        run_date=refresh_dt,
                        id=f"refresh_race_{race_num}",
                        name=f"Pre-Race Refresh #{race_num}",
                        replace_existing=True,
                    )
                    logger.info(
                        f"Scheduled pre-race refresh for race {race_num} at {refresh_dt.strftime('%H:%M')}"
                    )

                # Schedule the notification job
                self.scheduler.add_job(
                    func=lambda rn=race_num: self.notify_pre_race_job(rn),
                    trigger="date",
                    run_date=notify_dt,
                    id=f"notify_race_{race_num}",
                    name=f"Pre-Race Notify #{race_num}",
                    replace_existing=True,
                )
                logger.info(f"Scheduled pre-race notification for race {race_num} at {notify_dt.strftime('%H:%M')}")

            except Exception as e:
                logger.warning(f"Failed to schedule notification for race {race.race_number}: {e}")

    def refresh_pre_race_data_job(self, race_number: int) -> None:
        """
        Refresh odds and regenerate prediction at T-120min, then snapshot to backtest history.
        """
        logger.info(f"=== refresh_pre_race_data_job race {race_number} ===")
        race = next((r for r in self._races if r.race_number == race_number), None)
        if not race:
            logger.warning(f"No race found for race {race_number}")
            return
        if not self._risk_guard_allows_notifications(race_number):
            return

        try:
            from config import config
            from model.backtest import Backtester
            from scraper.horse_profile import fetch_horse_profile
            from scraper.odds import fetch_odds
            from model.predictor import RacePredictor
            from model.trainer import EnsembleTrainer
            from features.elo import ELOSystem
            from features.jockey_trainer import JockeyTrainerStats
            from features.draw import DrawStats

            elo = ELOSystem(k_factor=config.ELO_K_FACTOR, base_rating=config.ELO_BASE)
            elo.load_ratings("data/processed/elo_ratings.json")

            jt_stats = JockeyTrainerStats()
            jt_stats.load()

            draw_stats = DrawStats()
            draw_stats.load()

            trainer = EnsembleTrainer()
            if not trainer.load():
                logger.warning("Skip refresh_pre_race_data_job: model not available")
                return

            predictor = RacePredictor(trainer=trainer, top_n=config.TOP_N)
            horse_profiles = {h.horse_code: fetch_horse_profile(h.horse_code) for h in race.horses}
            odds = fetch_odds(
                race.race_date, race.venue_code, race.race_number,
                [h.horse_number for h in race.horses]
            )
            pred = predictor.predict_race_from_components(
                race=race,
                horse_profiles=horse_profiles,
                odds=odds,
                elo=elo,
                jt_stats=jt_stats,
                draw_stats=draw_stats,
            )

            self._predictions[race.race_number] = pred
            if self.app_state:
                self.app_state.predictions[race.race_number] = pred
                self.app_state.odds[race.race_number] = odds

            race_id = f"{race.race_date}_{race.venue_code}_{race.race_number}"
            backtester = Backtester(repair_missing=True)
            backtester.record_prediction(race_id, pred, race.race_date)
            logger.info(f"Snapshot prediction recorded for {race_id}")

        except Exception as e:
            logger.error(f"refresh_pre_race_data_job failed: {e}", exc_info=True)

    def notify_pre_race_job(self, race_number: int) -> None:
        """
        Send pre-race notification for a specific race.

        Args:
            race_number: Race number to notify
        """
        logger.info(f"=== notify_pre_race_job race {race_number} ===")
        race = next((r for r in self._races if r.race_number == race_number), None)
        pred = self._predictions.get(race_number)

        if not race or not pred:
            logger.warning(f"No race/prediction for race {race_number}")
            return
        if not self._risk_guard_allows_notifications(race_number):
            return

        try:
            from notifier.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            asyncio.run(notifier.send_race_preview(race, pred))
        except Exception as e:
            logger.error(f"Failed to send pre-race notification: {e}")

    def record_results_job(self) -> None:
        """
        Fetch actual race results and record prediction accuracy.
        Runs at 20:30 HKT (after last race typically finishes).
        """
        logger.info("=== record_results_job ===")
        if not self._races:
            return

        try:
            from model.backtest import Backtester
            from config import config
            from scraper.racecard import fetch_results

            backtester = Backtester(repair_missing=True)

            results_map = {}
            if self._races and not config.DEMO_MODE:
                race_date = self._races[0].race_date
                venue_code = self._races[0].venue_code
                results_map = fetch_results(race_date, venue_code)

            for race in self._races:
                race_id = f"{race.race_date}_{race.venue_code}_{race.race_number}"
                pred = self._predictions.get(race.race_number)
                if not pred:
                    continue

                # Record prediction
                backtester.record_prediction(race_id, pred, race.race_date)

                # Fetch actual results (DEMO MODE: simulate)
                if config.DEMO_MODE:
                    import random
                    n = len(race.horses)
                    result = random.sample([h.horse_number for h in race.horses], min(n, 10))
                    backtester.record_result(
                        race_id,
                        result,
                        is_real_result=False,
                        result_source="demo_simulation",
                        result_source_confidence="C",
                        result_source_note="scheduler_demo_mode",
                    )
                    logger.info(f"Recorded demo result for {race_id}: {result[:3]}")
                else:
                    actual_result = results_map.get(race.race_number)
                    if not actual_result:
                        logger.warning(f"No real results found for {race_id}; skipping result record")
                        continue
                    backtester.record_result(
                        race_id,
                        actual_result,
                        is_real_result=True,
                        result_source="hkjc_racecard_results",
                        result_source_url="https://racing.hkjc.com/zh-hk/local/information/localresults",
                        result_source_confidence="B",
                        result_source_note="scheduler_fetch_results_map",
                    )
                    logger.info(f"Recorded real result for {race_id}: {actual_result[:3]}")

            backtester.save_history()
            self.optimize_strategy_job(backtester)

        except Exception as e:
            logger.error(f"record_results_job failed: {e}", exc_info=True)

    def optimize_strategy_job(self, backtester=None) -> None:
        """Optimize recommendation thresholds using settled real-data records."""
        try:
            from config import config
            from model.backtest import Backtester
            from model.self_optimizer import StrategySelfOptimizer

            bt = backtester or Backtester(repair_missing=True)
            settled = bt.get_settled_records(real_only=True)
            if not settled:
                logger.info("No settled real-data records yet; skip strategy optimization")
                return

            lookback_days = max(30, int(config.OPTIMIZATION_LOOKBACK_DAYS))
            cutoff = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            recent = [r for r in settled if r.race_date >= cutoff]

            optimizer = StrategySelfOptimizer(config.STRATEGY_PROFILE_PATH)
            profile = optimizer.optimize(recent, min_samples=24)
            logger.info(
                "Strategy profile active: min_conf=%.2f max_odds=%.1f samples=%s win_rate=%.2f%% roi=%.2f%%",
                profile.min_confidence,
                profile.max_win_odds,
                profile.selected_samples,
                profile.selected_win_rate * 100,
                profile.selected_roi,
            )
        except Exception as e:
            logger.error(f"optimize_strategy_job failed: {e}", exc_info=True)

    def send_daily_summary_job(self) -> None:
        """
        Send end-of-day Telegram summary with performance stats.
        Runs at 21:00 HKT.
        """
        logger.info("=== send_daily_summary_job ===")
        try:
            from model.backtest import Backtester
            from notifier.telegram import TelegramNotifier

            backtester = Backtester()
            stats = backtester.get_summary_stats()
            today = date.today().strftime("%Y-%m-%d")

            notifier = TelegramNotifier()
            asyncio.run(notifier.send_daily_summary(today, stats))

        except Exception as e:
            logger.error(f"send_daily_summary_job failed: {e}")

    def elo_update_job(self) -> None:
        """
        Post-race ELO auto-update.
        Runs at 21:30 HKT on race days — after all results are officially settled.
        """
        today = date.today()
        if today.weekday() not in RACE_WEEKDAYS:
            return

        logger.info("=== elo_update_job started ===")
        try:
            from features.elo_updater import run_post_race_elo_update

            racecard_horses = {}
            if self._races:
                for race in self._races:
                    racecard_horses[race.race_number] = {
                        str(h.horse_number): h.horse_code
                        for h in race.horses
                    }

            venue_code = self._races[0].venue_code if self._races else "ST"
            summary = run_post_race_elo_update(
                race_date=today.strftime("%Y-%m-%d"),
                venue_code=venue_code,
                racecard_horses=racecard_horses or None,
            )

            msg = (
                f"✅ ELO 更新完成 {today}\n"
                f"更新賽事: {summary.get('races_updated', 0)} 場\n"
                f"更新馬匹: {summary.get('horses_updated', 0)} 匹"
            )
            logger.info(msg)

            if self.app_state:
                self.app_state.last_elo_update = summary

        except Exception as e:
            logger.error(f"elo_update_job failed: {e}", exc_info=True)

    def auto_retrain_job(self) -> None:
        """
        Weekly model retrain using accumulated historical data.
        Runs every Sunday at 02:00 HKT.
        """
        logger.info("=== auto_retrain_job started ===")
        try:
            from model.trainer import ModelTrainer
            from scraper.results_fetcher import load_results_history

            history = load_results_history()
            if len(history) < 20:
                logger.info(f"Insufficient history ({len(history)} races) for retrain; skipping")
                return

            trainer = ModelTrainer()
            meta_path = "data/models/meta.json"
            last_train_races = 0
            if os.path.exists(meta_path):
                import json as _json
                try:
                    with open(meta_path) as f:
                        meta = _json.load(f)
                    last_train_races = meta.get("training_samples", 0)
                except Exception:
                    pass

            if len(history) - last_train_races < 20:
                logger.info(f"Only {len(history) - last_train_races} new races since last train; skipping")
                return

            logger.info(f"Retraining with {len(history)} historical races...")
            trainer.train()
            logger.info("Weekly retrain complete")

        except Exception as e:
            logger.error(f"auto_retrain_job failed: {e}", exc_info=True)


# ── Utility ────────────────────────────────────────────────────────────────────

def _send_alert(message: str, level: str = "WARNING") -> None:
    """Send a system alert via Telegram (fire and forget)."""
    try:
        from notifier.telegram import TelegramNotifier
        notifier = TelegramNotifier()
        asyncio.run(notifier.send_system_alert(message, level))
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
