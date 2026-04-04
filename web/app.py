"""
HKJC 賽馬預測系統 - Flask Web 應用程式
Dashboard and API for the horse racing prediction system.
"""
import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Dict

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS

logger = logging.getLogger(__name__)

# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """Create and configure the Flask application."""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    CORS(app)

    from config import config
    app.secret_key = config.SECRET_KEY

    # ── Application state ──────────────────────────────────────────────────
    app_state = _AppState()
    app.config["APP_STATE"] = app_state

    # ── Routes ─────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        """Today's races dashboard."""
        state = app.config["APP_STATE"]
        state.ensure_loaded()

        races = state.races
        predictions = state.predictions
        weather = state.weather
        today = date.today().strftime("%Y年%m月%d日")
        venue = races[0].venue if races else "沙田"

        # Build race cards for template
        race_cards = []
        for race in races:
            pred = predictions.get(race.race_number)
            top3 = pred.top5[:3] if pred and pred.top5 else []
            value_count = len(pred.value_bets) if pred else 0
            confidence = pred.confidence if pred else 0.0

            race_cards.append({
                "race": race,
                "top3": top3,
                "value_count": value_count,
                "confidence": confidence,
                "confidence_pct": int(confidence * 100),
            })

        return render_template(
            "index.html",
            race_cards=race_cards,
            today=today,
            venue=venue,
            weather=weather,
            total_races=len(races),
            demo_mode=config.DEMO_MODE,
            load_error=state.load_error,
        )

    @app.route("/race/<int:race_number>")
    def race_detail(race_number: int):
        """Per-race prediction detail page."""
        state = app.config["APP_STATE"]
        state.ensure_loaded()

        race = next((r for r in state.races if r.race_number == race_number), None)
        if not race:
            return jsonify({"error": f"Race {race_number} not found"}), 404

        pred = state.predictions.get(race_number)
        odds = state.odds.get(race_number, {})

        # Build runners table
        runners = []
        if pred:
            for h in pred.ranked_horses:
                runners.append({
                    "horse": h,
                    "rank": h.rank,
                    "confidence_pct": int(h.confidence * 100),
                    "model_prob_pct": round(h.model_prob * 100, 1),
                    "implied_prob_pct": round(h.implied_win_prob * 100, 1),
                    "overlay_pct": round(h.overlay_pct * 100, 1),
                    "ev_per_10": h.expected_value,
                })

        # Chart data for Chart.js
        chart_labels = [f"#{h['horse'].horse_number} {h['horse'].horse_name[:12]}" for h in runners[:8]]
        chart_model = [round(h["horse"].model_prob * 100, 1) for h in runners[:8]]
        chart_implied = [round(h["horse"].implied_win_prob * 100, 1) for h in runners[:8]]

        return render_template(
            "race_detail.html",
            race=race,
            prediction=pred,
            runners=runners,
            odds=odds,
            chart_labels=json.dumps(chart_labels),
            chart_model=json.dumps(chart_model),
            chart_implied=json.dumps(chart_implied),
            total_races=len(state.races),
        )

    @app.route("/backtest")
    def backtest():
        """Historical performance / backtesting page."""
        state = app.config["APP_STATE"]
        state.ensure_loaded()

        if state.backtester is None:
            from model.backtest import Backtester
            state.backtester = Backtester()

        summary = state.backtester.get_summary_stats()
        r7 = state.backtester.calculate_roi("top1_win", 7)
        r30 = state.backtester.calculate_roi("top1_win", 30)
        r30_place = state.backtester.calculate_roi("top1_place", 30)
        r30_trio = state.backtester.calculate_roi("trio", 30)

        # ROI chart data (last 30 records)
        records = summary.get("recent_records", [])
        roi_dates = [r["date"] for r in records[-15:]]
        roi_values = []
        running_net = 0.0
        for r in records[-15:]:
            # Approximate running net profit
            running_net += 10.0 if r["winner_correct"] else -10.0
            roi_values.append(round(running_net, 1))

        return render_template(
            "backtest.html",
            summary=summary,
            r7=r7,
            r30=r30,
            r30_place=r30_place,
            r30_trio=r30_trio,
            records=records,
            roi_dates=json.dumps(roi_dates),
            roi_values=json.dumps(roi_values),
            total_races=len(state.races) if state.races else 0,
            demo_mode=config.DEMO_MODE,
            load_error=state.load_error,
        )

    # ── JSON APIs ──────────────────────────────────────────────────────────

    @app.route("/api/odds/<int:race_number>")
    def api_odds(race_number: int):
        """Live odds JSON endpoint (polled every 60s by frontend)."""
        state = app.config["APP_STATE"]

        # Refresh odds from source (or drift mock odds for demo)
        from config import config
        if config.DEMO_MODE:
            from scraper.odds import generate_mock_odds_update
            current = state.odds.get(race_number, {})
            if current:
                updated = generate_mock_odds_update(current, drift_pct=0.03)
            else:
                from scraper.odds import mock_odds
                race = next((r for r in state.races if r.race_number == race_number), None)
                n = len(race.horses) if race else 12
                updated = mock_odds(race_number, n)
            state.odds[race_number] = updated
        else:
            from scraper.odds import fetch_odds
            race = next((r for r in state.races if r.race_number == race_number), None)
            if race:
                updated = fetch_odds(race.race_date, race.venue_code, race_number,
                                     [h.horse_number for h in race.horses])
                state.odds[race_number] = updated
            else:
                updated = {}

        return jsonify({
            "race_number": race_number,
            "odds": {str(k): v for k, v in state.odds.get(race_number, {}).items()},
            "updated_at": datetime.now().isoformat(),
        })

    @app.route("/api/predict/<int:race_number>")
    def api_predict(race_number: int):
        """Prediction JSON endpoint."""
        state = app.config["APP_STATE"]
        pred = state.predictions.get(race_number)
        if not pred:
            return jsonify({"error": "No prediction available"}), 404

        return jsonify({
            "race_number": race_number,
            "confidence": float(pred.confidence),
            "top5": [
                {
                    "horse_number": int(h.horse_number),
                    "horse_name": str(h.horse_name),
                    "horse_name_ch": str(h.horse_name_ch),
                    "rank": int(h.rank),
                    "model_prob": float(h.model_prob),
                    "win_prob": float(h.win_prob),
                    "implied_win_prob": float(h.implied_win_prob),
                    "win_odds": float(h.win_odds),
                    "confidence": float(h.confidence),
                    "is_value_bet": bool(h.is_value_bet),
                    "expected_value": float(h.expected_value),
                }
                for h in pred.top5
            ],
            "value_bets": [int(h.horse_number) for h in pred.value_bets],
            "trio_suggestions": [[int(x) for x in t] for t in pred.trio_suggestions[:3]],
        })

    @app.route("/api/races")
    def api_races():
        """List today's races as JSON."""
        state = app.config["APP_STATE"]
        state.ensure_loaded()
        races_data = []
        for race in state.races:
            races_data.append({
                "race_number": race.race_number,
                "venue": race.venue,
                "distance": race.distance,
                "track_type": race.track_type,
                "race_class": race.race_class,
                "going": race.going,
                "start_time": race.start_time,
                "n_horses": len(race.horses),
            })
        return jsonify({"races": races_data, "date": date.today().isoformat()})

    @app.route("/api/status")
    def api_status():
        """System status endpoint."""
        from config import config
        state = app.config["APP_STATE"]
        state.ensure_loaded()
        return jsonify({
            "demo_mode": config.DEMO_MODE,
            "races_loaded": len(state.races),
            "predictions_available": len(state.predictions),
            "model_trained": state.trainer.is_trained if state.trainer else False,
            "load_error": state.load_error,
            "timestamp": datetime.now().isoformat(),
        })

    @app.route("/api/refresh")
    def api_refresh():
        """Force refresh of all data."""
        state = app.config["APP_STATE"]
        state.reset()
        loaded_ok = state.ensure_loaded()
        if not loaded_ok:
            return jsonify({"status": "error", "error": state.load_error, "races": 0}), 503
        return jsonify({"status": "ok", "races": len(state.races)})

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    return app


# ── Application State ──────────────────────────────────────────────────────────

class _AppState:
    """
    Singleton-ish state container for the Flask app.

    Holds loaded races, predictions, odds, and model objects.
    Lazy-loads everything on first request.
    """

    def __init__(self):
        self.races = []
        self.predictions: Dict[int, object] = {}
        self.odds: Dict[int, Dict] = {}
        self.profile_cache: Dict[str, object] = {}
        self.weather = None
        self.trainer = None
        self.predictor = None
        self.backtester = None
        self.elo = None
        self.jt_stats = None
        self.draw_stats = None
        self.load_error = None
        self._loaded = False

    def reset(self):
        """Reset state to force a fresh load."""
        self._loaded = False
        self.load_error = None
        self.races = []
        self.predictions = {}
        self.odds = {}
        self.profile_cache = {}

    def _get_horse_profile_cached(self, horse_code: str):
        """Fetch horse profile once per app refresh cycle to reduce repeated network calls."""
        from scraper.horse_profile import fetch_horse_profile

        if horse_code in self.profile_cache:
            return self.profile_cache[horse_code]

        profile = fetch_horse_profile(horse_code)
        self.profile_cache[horse_code] = profile
        return profile

    def ensure_loaded(self):
        """Load all data if not already loaded."""
        if self._loaded:
            return self.load_error is None
        try:
            self._load_all()
            self.load_error = None
            self._loaded = True
            return True
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load app state: {e}", exc_info=True)
            self._loaded = True  # prevent infinite retry loop
            return False

    def _load_all(self):
        """Load races, model, features, and generate predictions."""
        from config import config
        logger.info("Loading HKJC Predictor application state...")

        # ── Scrapers ───────────────────────────────────────────────────────
        from scraper.racecard import fetch_racecard
        from scraper.odds import fetch_odds, mock_odds
        from scraper.weather import fetch_weather

        today = date.today().strftime("%Y-%m-%d")
        self.races = fetch_racecard(today)
        if config.REAL_DATA_ONLY and not config.DEMO_MODE and not self.races:
            raise RuntimeError("No real races available while REAL_DATA_ONLY=True")
        self.weather = fetch_weather(self.races[0].venue_code if self.races else "ST")

        logger.info(f"Loaded {len(self.races)} races")

        # ── Feature components ─────────────────────────────────────────────
        from features.elo import ELOSystem
        from features.jockey_trainer import JockeyTrainerStats
        from features.draw import DrawStats

        self.elo = ELOSystem(k_factor=config.ELO_K_FACTOR, base_rating=config.ELO_BASE)
        self.elo.load_ratings("data/processed/elo_ratings.json")

        self.jt_stats = JockeyTrainerStats()
        self.jt_stats.load()

        self.draw_stats = DrawStats()
        if not self.draw_stats.load():
            if not config.REAL_DATA_ONLY:
                self.draw_stats.bootstrap_from_mock()

        # ── Model ──────────────────────────────────────────────────────────
        from model.trainer import EnsembleTrainer
        from model.predictor import RacePredictor
        from model.backtest import Backtester

        self.trainer = EnsembleTrainer()
        if not self.trainer.load():
            if config.REAL_DATA_ONLY and not config.ALLOW_SYNTHETIC_TRAINING:
                raise RuntimeError(
                    "Model not found and synthetic training is disabled (REAL_DATA_ONLY=True)."
                )
            logger.info("Training model on synthetic data...")
            self.trainer.train()
            self.trainer.save()

        self.predictor = RacePredictor(trainer=self.trainer, top_n=config.TOP_N)
        self.backtester = Backtester()

        # ── Generate predictions ───────────────────────────────────────────
        for race in self.races:
            try:
                # Fetch horse profiles
                horse_profiles = {}
                for horse in race.horses:
                    horse_profiles[horse.horse_code] = self._get_horse_profile_cached(horse.horse_code)

                # Fetch odds
                race_odds = fetch_odds(
                    race.race_date, race.venue_code, race.race_number,
                    [h.horse_number for h in race.horses]
                )
                self.odds[race.race_number] = race_odds

                # Generate prediction
                pred = self.predictor.predict_race_from_components(
                    race=race,
                    horse_profiles=horse_profiles,
                    odds=race_odds,
                    elo=self.elo,
                    jt_stats=self.jt_stats,
                    draw_stats=self.draw_stats,
                )
                self.predictions[race.race_number] = pred

                logger.info(
                    f"Race {race.race_number}: top pick = "
                    f"#{pred.top5[0].horse_number} {pred.top5[0].horse_name} "
                    f"({pred.top5[0].confidence:.1%} confidence)"
                    if pred.top5 else f"Race {race.race_number}: no prediction"
                )

            except Exception as e:
                logger.error(f"Error predicting race {race.race_number}: {e}", exc_info=True)

        logger.info(f"Generated {len(self.predictions)} race predictions")


# ── Standalone run ─────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    from config import config
    logging.basicConfig(level=logging.INFO)
    app.run(
        host="0.0.0.0",
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
