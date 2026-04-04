"""
HKJC 賽馬預測系統 - 模型訓練模組
XGBoost + LightGBM ensemble training for top-3 prediction.

Generates 2 seasons of synthetic training data if no real data exists.
"""
import json
import logging
import os
import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception as e:
    XGB_AVAILABLE = False
    logging.warning("XGBoost unavailable (%s); ensemble will use LightGBM only", e)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception as e:
    LGB_AVAILABLE = False
    logging.warning("LightGBM unavailable (%s); ensemble will use XGBoost only", e)

from features.builder import MODEL_FEATURE_COLS

logger = logging.getLogger(__name__)

MODEL_DIR = "data/models"


class EnsembleTrainer:
    """
    Trains an XGBoost + LightGBM ensemble for horse racing prediction.

    The target variable is binary: did the horse finish in top 3 (place)?
    Both models output probability scores, which are averaged for the ensemble.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = MODEL_FEATURE_COLS
        self.is_trained = False
        self.training_source: str = "unknown"
        self.training_generated_at: Optional[str] = None

        # XGBoost hyperparameters
        self.xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "scale_pos_weight": 2.5,  # accounts for class imbalance (top3 ~= 25%)
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }

        # LightGBM hyperparameters
        self.lgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }

    def generate_synthetic_data(self, n_races: int = 800) -> pd.DataFrame:
        """
        Generate synthetic training data representing ~2 Hong Kong racing seasons.

        Args:
            n_races: Number of simulated races

        Returns:
            DataFrame with features + target column "is_top3"
        """
        logger.info(f"Generating {n_races} synthetic races for training data...")
        rng = random.Random(42)
        np.random.seed(42)

        rows = []
        base_date = date.today() - timedelta(days=730)  # ~2 years ago

        for race_num in range(n_races):
            n_runners = rng.randint(8, 14)
            race_date = base_date + timedelta(days=race_num // 3)
            distance = rng.choice([1000, 1200, 1400, 1600, 1800, 2000, 2400])
            venue_is_st = rng.choice([0, 1])

            # Generate raw horse attributes
            true_quality = np.random.normal(0, 1, n_runners)  # latent quality
            elo_ratings = 1500 + true_quality * 80 + np.random.normal(0, 30, n_runners)
            handicap_ratings = 70 + true_quality * 15 + np.random.normal(0, 8, n_runners)
            form_scores = np.clip(0.5 + true_quality * 0.15 + np.random.normal(0, 0.1, n_runners), 0, 1)
            draws = rng.sample(range(1, n_runners + 1), n_runners)

            # Actual finishing order (based on true quality + randomness)
            noise = np.random.normal(0, 0.8, n_runners)
            scores = true_quality + noise
            finish_order = np.argsort(-scores)  # descending

            for i in range(n_runners):
                horse_idx = i
                finish_pos = np.where(finish_order == horse_idx)[0][0] + 1
                is_top3 = int(finish_pos <= 3)

                draw = draws[i]
                # Draw bias effect
                draw_adv = (n_runners / 2 - draw) / n_runners * 0.3

                jk_win_30d = np.clip(rng.gauss(0.18, 0.06), 0.05, 0.35)
                tr_win_30d = np.clip(rng.gauss(0.16, 0.05), 0.04, 0.28)
                combo_win = np.clip((jk_win_30d + tr_win_30d) / 2 * rng.gauss(1, 0.2), 0.03, 0.40)

                win_odds = max(1.2, 1.0 / max(0.02, (true_quality[i] - min(true_quality) + 0.5) / n_runners * 1.1))
                place_odds = max(1.05, win_odds / rng.uniform(3.0, 4.5))

                days_since = rng.randint(10, 200)

                rows.append({
                    # Features
                    "elo_rating": round(elo_ratings[i], 2),
                    "elo_vs_field": round(elo_ratings[i] - np.mean(elo_ratings), 2),
                    "elo_delta_last3": round(rng.gauss(0, 5), 2),
                    "jockey_win_rate_30d": round(jk_win_30d, 4),
                    "jockey_place_rate_30d": round(min(jk_win_30d * 2.8, 0.65), 4),
                    "jockey_win_rate_overall": round(jk_win_30d * rng.uniform(0.9, 1.1), 4),
                    "jockey_venue_win_rate": round(jk_win_30d * rng.uniform(0.8, 1.2), 4),
                    "jockey_distance_win_rate": round(jk_win_30d * rng.uniform(0.8, 1.2), 4),
                    "trainer_win_rate_30d": round(tr_win_30d, 4),
                    "trainer_place_rate_30d": round(min(tr_win_30d * 2.8, 0.60), 4),
                    "trainer_win_rate_overall": round(tr_win_30d * rng.uniform(0.9, 1.1), 4),
                    "combo_win_rate": round(combo_win, 4),
                    "combo_place_rate": round(min(combo_win * 2.5, 0.55), 4),
                    "draw_advantage_score": round(draw_adv + rng.gauss(0, 0.1), 4),
                    "draw_position": draw,
                    "draw_normalised": round(draw / n_runners, 4),
                    "recent_form_score": round(float(form_scores[i]), 4),
                    "win_rate": round(max(0, true_quality[i] * 0.1 + 0.15 + rng.gauss(0, 0.05)), 4),
                    "place_rate": round(max(0, true_quality[i] * 0.15 + 0.35 + rng.gauss(0, 0.08)), 4),
                    "distance_aptitude": round(np.clip(0.35 + rng.gauss(0, 0.15), 0, 1), 4),
                    "going_aptitude": round(np.clip(0.35 + rng.gauss(0, 0.12), 0, 1), 4),
                    "handicap_rating": round(float(handicap_ratings[i]), 1),
                    "rating_trend": round(rng.gauss(0, 3), 4),
                    "rating_vs_avg": round(float(handicap_ratings[i] - np.mean(handicap_ratings)), 4),
                    "weight_carried": rng.randint(113, 133),
                    "weight_vs_avg": round(rng.gauss(0, 0.05), 4),
                    "days_since_last_run": days_since,
                    "win_odds": round(win_odds, 2),
                    "place_odds": round(place_odds, 2),
                    "implied_win_probability": round(1 / win_odds, 4),
                    "implied_place_probability": round(1 / place_odds, 4),
                    "race_distance": distance,
                    "n_runners": n_runners,
                    "venue_is_st": venue_is_st,
                    # Target
                    "is_top3": is_top3,
                    "finish_position": finish_pos,
                })

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} horse-race records ({df['is_top3'].sum()} top-3 finishes)")
        return df

    def build_real_incremental_data(self, settled_records: List, min_rows: int = 30) -> Optional[pd.DataFrame]:
        """
        Build a real-data incremental training set from settled backtest records.

        The current history stores compact race-level records (predicted top-3 and outcomes),
        not full per-runner feature snapshots. This method expands each settled race into up to
        three horse-level rows using available real market signals and race metadata.
        """
        rows: List[Dict] = []

        # Preferred path: use full per-runner feature snapshots captured at prediction time
        # and auto-labeled after official results are recorded.
        snapshot_rows: List[Dict] = []
        for rec in settled_records or []:
            rec_rows = list(getattr(rec, "feature_rows", []) or [])
            if not rec_rows:
                continue
            actual_top3 = set((getattr(rec, "actual_result", []) or [])[:3])
            finish_pos_map = {
                int(h): idx + 1
                for idx, h in enumerate((getattr(rec, "actual_result", []) or []))
                if isinstance(h, int)
            }
            for raw in rec_rows:
                if not isinstance(raw, dict):
                    continue
                horse_no = int(raw.get("horse_number", 0) or 0)
                if horse_no <= 0:
                    continue
                row = {"horse_number": horse_no}
                for col in self.feature_cols:
                    value = raw.get(col, 0.0)
                    try:
                        row[col] = float(value)
                    except (TypeError, ValueError):
                        row[col] = 0.0

                if "is_top3" in raw:
                    row["is_top3"] = int(raw.get("is_top3", 0) or 0)
                else:
                    row["is_top3"] = 1 if horse_no in actual_top3 else 0

                if "finish_position" in raw:
                    row["finish_position"] = int(raw.get("finish_position", 0) or 0)
                else:
                    row["finish_position"] = int(finish_pos_map.get(horse_no, 0))

                snapshot_rows.append(row)

        if len(snapshot_rows) >= min_rows:
            df = pd.DataFrame(snapshot_rows)
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            if df["is_top3"].nunique() >= 2:
                logger.info(
                    "Built real incremental data from labeled snapshots: rows=%s races=%s positive_rate=%.3f",
                    len(df),
                    len(set(getattr(r, "race_id", "") for r in settled_records or [])),
                    float(df["is_top3"].mean()),
                )
                return df[self.feature_cols + ["is_top3", "finish_position"]]
            logger.warning("Labeled snapshots available but single-class; fallback to proxy rows")

        for rec in settled_records or []:
            if not getattr(rec, "actual_result", None):
                continue

            predicted_top3 = list(getattr(rec, "predicted_top3", []) or [])[:3]
            predicted_top3_odds = list(getattr(rec, "predicted_top3_odds", []) or [])[:3]
            if not predicted_top3:
                continue

            actual_top3 = set((getattr(rec, "actual_result", []) or [])[:3])
            n_runners = max(6, min(18, len(getattr(rec, "actual_result", []) or []) or 12))
            venue = str(getattr(rec, "venue", "") or "")
            venue_is_st = 1 if "sha" in venue.lower() else 0
            race_distance = int(getattr(rec, "distance", 1400) or 1400)

            for idx, horse_no in enumerate(predicted_top3):
                if not horse_no:
                    continue
                win_odds = float(predicted_top3_odds[idx]) if idx < len(predicted_top3_odds) and predicted_top3_odds[idx] else 12.0
                win_odds = max(1.01, win_odds)
                place_odds = max(1.01, win_odds / 3.6)

                # A lightweight proxy for horse-level strength using only settled real fields.
                rank_hint = idx + 1
                rec_conf = float(getattr(rec, "model_confidence", 0.5) or 0.5)
                confidence_hint = max(0.05, min(0.99, rec_conf - (rank_hint - 1) * 0.08))

                row = {
                    "elo_rating": 1500.0 + (3 - rank_hint) * 8.0,
                    "elo_vs_field": float(3 - rank_hint) * 6.0,
                    "elo_delta_last3": 0.0,
                    "jockey_win_rate_30d": 0.16,
                    "jockey_place_rate_30d": 0.38,
                    "jockey_win_rate_overall": 0.15,
                    "jockey_venue_win_rate": 0.15,
                    "jockey_distance_win_rate": 0.15,
                    "trainer_win_rate_30d": 0.14,
                    "trainer_place_rate_30d": 0.34,
                    "trainer_win_rate_overall": 0.14,
                    "combo_win_rate": 0.12,
                    "combo_place_rate": 0.30,
                    "draw_advantage_score": 0.0,
                    "draw_position": min(n_runners, 2 + idx * 3),
                    "draw_normalised": float(min(n_runners, 2 + idx * 3)) / float(max(n_runners, 1)),
                    "recent_form_score": confidence_hint,
                    "win_rate": max(0.01, confidence_hint * 0.40),
                    "place_rate": max(0.01, confidence_hint * 0.75),
                    "distance_aptitude": 0.50,
                    "going_aptitude": 0.50,
                    "handicap_rating": 70.0 + (3 - rank_hint) * 1.8,
                    "rating_trend": 0.0,
                    "rating_vs_avg": float(3 - rank_hint) * 1.2,
                    "weight_carried": 123,
                    "weight_vs_avg": 0.0,
                    "days_since_last_run": 28,
                    "win_odds": win_odds,
                    "place_odds": place_odds,
                    "implied_win_probability": 1.0 / win_odds,
                    "implied_place_probability": 1.0 / place_odds,
                    "race_distance": race_distance,
                    "n_runners": n_runners,
                    "venue_is_st": venue_is_st,
                    "is_top3": 1 if int(horse_no) in actual_top3 else 0,
                    "finish_position": 0,
                }
                rows.append(row)

        if len(rows) < min_rows:
            logger.info(
                "Real incremental data too small: rows=%s min_rows=%s",
                len(rows),
                min_rows,
            )
            return None

        df = pd.DataFrame(rows)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_cols + ["is_top3", "finish_position"]]

        # Need both classes for supervised training.
        if df["is_top3"].nunique() < 2:
            logger.warning("Real incremental data has single class only; skipping retrain")
            return None

        logger.info(
            "Built real incremental data: rows=%s races=%s positive_rate=%.3f",
            len(df),
            len(set(getattr(r, "race_id", "") for r in settled_records or [])),
            float(df["is_top3"].mean()),
        )
        return df

    def train(self, df: Optional[pd.DataFrame] = None, training_source: str = "synthetic") -> Dict:
        """
        Train XGBoost + LightGBM on the provided (or synthetic) data.

        Args:
            df: Training DataFrame with features and "is_top3" column.
                If None, generates synthetic data.

        Returns:
            Dict with training metrics.
        """
        if df is None:
            df = self.generate_synthetic_data()
            training_source = "synthetic"

        X = df[self.feature_cols].copy()
        y = df["is_top3"].values

        # Handle missing values
        X = X.fillna(X.median())

        logger.info(f"Training on {len(X)} samples, {X.shape[1]} features. Positive rate: {y.mean():.3f}")

        metrics = {}

        # ── XGBoost ──────────────────────────────────────────────────────────
        if XGB_AVAILABLE:
            logger.info("Training XGBoost...")
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
            self.xgb_model.fit(X, y)
            xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
            metrics["xgb_auc_train"] = round(roc_auc_score(y, xgb_pred), 4)
            logger.info(f"XGBoost train AUC: {metrics['xgb_auc_train']}")

        # ── LightGBM ─────────────────────────────────────────────────────────
        if LGB_AVAILABLE:
            logger.info("Training LightGBM...")
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            self.lgb_model.fit(X, y)
            lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
            metrics["lgb_auc_train"] = round(roc_auc_score(y, lgb_pred), 4)
            logger.info(f"LightGBM train AUC: {metrics['lgb_auc_train']}")

        # ── Ensemble AUC ──────────────────────────────────────────────────────
        ensemble_pred = self._ensemble_proba(X)
        metrics["ensemble_auc_train"] = round(roc_auc_score(y, ensemble_pred), 4)
        logger.info(f"Ensemble train AUC: {metrics['ensemble_auc_train']}")

        self.is_trained = True
        self.training_source = training_source
        self.training_generated_at = datetime.now().isoformat()
        return metrics

    def _ensemble_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute ensemble probability as weighted average of XGB and LGB."""
        preds = []
        if XGB_AVAILABLE and self.xgb_model is not None:
            preds.append(self.xgb_model.predict_proba(X)[:, 1])
        if LGB_AVAILABLE and self.lgb_model is not None:
            preds.append(self.lgb_model.predict_proba(X)[:, 1])
        if not preds:
            return np.full(len(X), 1.0 / 12)
        return np.mean(preds, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return ensemble probability for each horse.

        Args:
            X: Feature DataFrame (rows = horses)

        Returns:
            Array of top-3 probabilities
        """
        if not self.is_trained:
            logger.warning("Model not trained; loading from disk or training on synthetic data...")
            if not self.load():
                self.train()

        # Align columns
        missing_cols = [c for c in self.feature_cols if c not in X.columns]
        for col in missing_cols:
            X[col] = 0.0

        X_model = X[self.feature_cols].copy().fillna(0.0)
        return self._ensemble_proba(X_model)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance from both models."""
        rows = []
        if XGB_AVAILABLE and self.xgb_model is not None:
            importance = self.xgb_model.feature_importances_
            for col, imp in zip(self.feature_cols, importance):
                rows.append({"feature": col, "model": "XGBoost", "importance": imp})
        if LGB_AVAILABLE and self.lgb_model is not None:
            importance = self.lgb_model.feature_importances_
            for col, imp in zip(self.feature_cols, importance):
                rows.append({"feature": col, "model": "LightGBM", "importance": imp})
        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    def save(self) -> None:
        """Save models to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        if XGB_AVAILABLE and self.xgb_model is not None:
            joblib.dump(self.xgb_model, os.path.join(self.model_dir, "xgb_model.pkl"))
        if LGB_AVAILABLE and self.lgb_model is not None:
            joblib.dump(self.lgb_model, os.path.join(self.model_dir, "lgb_model.pkl"))
        meta = {
            "feature_cols": self.feature_cols,
            "xgb_available": XGB_AVAILABLE and self.xgb_model is not None,
            "lgb_available": LGB_AVAILABLE and self.lgb_model is not None,
            "is_trained": self.is_trained,
            "training_source": self.training_source,
            "training_generated_at": self.training_generated_at,
        }
        with open(os.path.join(self.model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Models saved to {self.model_dir}")

    def load(self) -> bool:
        """Load models from disk. Returns True if successful."""
        xgb_path = os.path.join(self.model_dir, "xgb_model.pkl")
        lgb_path = os.path.join(self.model_dir, "lgb_model.pkl")
        meta_path = os.path.join(self.model_dir, "meta.json")

        if not os.path.exists(meta_path):
            return False

        try:
            from config import config
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta.get("feature_cols", MODEL_FEATURE_COLS)
            self.training_source = meta.get("training_source", "unknown")
            self.training_generated_at = meta.get("training_generated_at")

            if config.REAL_DATA_ONLY and not str(self.training_source).startswith("real"):
                logger.error(
                    "Model provenance is not verified real data (training_source=%s); refusing load in REAL_DATA_ONLY mode",
                    self.training_source,
                )
                return False

            if os.path.exists(xgb_path) and XGB_AVAILABLE:
                self.xgb_model = joblib.load(xgb_path)
            if os.path.exists(lgb_path) and LGB_AVAILABLE:
                self.lgb_model = joblib.load(lgb_path)

            self.is_trained = bool(self.xgb_model or self.lgb_model)
            logger.info(f"Models loaded from {self.model_dir}")
            return self.is_trained
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
