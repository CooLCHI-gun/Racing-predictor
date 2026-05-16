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
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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

from features.builder import MODEL_FEATURE_COLS, INTERACTION_COLS

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
        self.feature_cols: List[str] = list(dict.fromkeys(MODEL_FEATURE_COLS + INTERACTION_COLS))
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
        }

    def generate_synthetic_data(self, n_races: int = 800) -> pd.DataFrame:
        """
        Generate realistic synthetic training data.

        Key properties (calibrated for realism):
        - Top-3 rate ~25-30% (balanced class)
        - Feature-target correlations similar to real racing (noisy, not obvious)
        - ELO range 1350-1650 (most horses 1400-1600)
        - Win rate 5-15%, place rate 15-35%
        - Draw advantage ±15% impact
        - Jockey/trainer combo effect 5-10%
        """
        logger.info(f"Generating {n_races} synthetic races for training data...")
        rows: List[Dict] = []
        rng = random.Random(42)

        for _ in range(n_races):
            n_runners = rng.choice([10, 11, 12, 13, 14, 12, 12, 12])
            distance = rng.choice([1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400])
            venue_is_st = rng.randint(0, 1)

            # True quality: mostly noise, tiny signal from ELo/handicap
            # Real racing ~10-15% predictable; 85-90% noise
            base_noise = np.random.normal(0, 1.0, n_runners)
            elo_ratings = [np.clip(np.random.normal(1500, 50), 1350, 1650) for _ in range(n_runners)]
            handicap_ratings = [rng.randint(40, 100) for _ in range(n_runners)]
            elo_factor = [(e - 1500) / 300.0 for e in elo_ratings]
            hcp_factor = [(h - 70) / 50.0 for h in handicap_ratings]

            # True quality: 90% noise, 10% signal
            true_quality = (base_noise * 0.90 + np.array(elo_factor) * 0.05 + np.array(hcp_factor) * 0.05)
            # Normalize to [0, 1] range within race
            true_quality = (true_quality - true_quality.min()) / max(true_quality.max() - true_quality.min(), 0.01)

            draws = list(range(1, n_runners + 1))
            rng.shuffle(draws)

            # Form scores: mostly random
            form_scores = [np.clip(rng.random() * 0.4 + tq * 0.1, 0.0, 0.5) for tq in true_quality]

            # Finishing order: almost entirely noise
            noise_std = 3.0
            scores = true_quality * 0.5 + np.random.normal(0, noise_std, n_runners)
            finish_order = np.argsort(-scores)

            for i in range(n_runners):
                horse_idx = i
                finish_pos = np.where(finish_order == horse_idx)[0][0] + 1
                is_top3 = int(finish_pos <= 3)

                draw = draws[i]
                draw_adv = (n_runners / 2 - draw) / n_runners * 0.3 * rng.uniform(0.5, 1.0)

                # Jockey/trainer: weakly correlated with horse quality
                jk_win_30d = np.clip(rng.gauss(0.14, 0.06) + elo_factor[i] * 0.02, 0.04, 0.30)
                tr_win_30d = np.clip(rng.gauss(0.12, 0.05) + hcp_factor[i] * 0.02, 0.03, 0.25)
                combo_win = np.clip((jk_win_30d + tr_win_30d) / 2 * rng.gauss(1, 0.15), 0.03, 0.30)

                win_odds = max(1.5, 1.0 / max(0.01, (true_quality[i] + 0.05) / n_runners * 1.1 + rng.gauss(0, 0.02)))
                win_odds = min(99.0, win_odds)
                place_odds = max(1.1, win_odds / rng.uniform(2.8, 4.5))
                days_since = rng.randint(7, 250)

                rows.append({
                    "elo_rating": round(elo_ratings[i], 2),
                    "elo_vs_field": round(elo_ratings[i] - np.mean(elo_ratings), 2),
                    "elo_delta_last3": round(rng.gauss(0, 8), 2),
                    "jockey_win_rate_30d": round(jk_win_30d, 4),
                    "jockey_place_rate_30d": round(min(jk_win_30d * 2.5, 0.60), 4),
                    "jockey_win_rate_overall": round(jk_win_30d * rng.uniform(0.85, 1.15), 4),
                    "jockey_venue_win_rate": round(jk_win_30d * rng.uniform(0.75, 1.25), 4),
                    "jockey_distance_win_rate": round(jk_win_30d * rng.uniform(0.75, 1.25), 4),
                    "trainer_win_rate_30d": round(tr_win_30d, 4),
                    "trainer_place_rate_30d": round(min(tr_win_30d * 2.5, 0.55), 4),
                    "trainer_win_rate_overall": round(tr_win_30d * rng.uniform(0.85, 1.15), 4),
                    "combo_win_rate": round(combo_win, 4),
                    "combo_place_rate": round(min(combo_win * 2.5, 0.50), 4),
                    "draw_advantage_score": round(draw_adv + rng.gauss(0, 0.15), 4),
                    "draw_position": draw,
                    "draw_normalised": round(draw / n_runners, 4),
                    "recent_form_score": round(float(form_scores[i]), 4),
                    "win_rate": round(max(0.01, true_quality[i] * 0.15 + 0.08 + rng.gauss(0, 0.06)), 4),
                    "place_rate": round(max(0.02, true_quality[i] * 0.20 + 0.22 + rng.gauss(0, 0.08)), 4),
                    "distance_aptitude": round(np.clip(0.30 + rng.gauss(0, 0.18), 0, 1), 4),
                    "going_aptitude": round(np.clip(0.30 + rng.gauss(0, 0.15), 0, 1), 4),
                    "handicap_rating": round(float(handicap_ratings[i]), 1),
                    "rating_trend": round(rng.gauss(0, 4), 4),
                    "rating_vs_avg": round(float(handicap_ratings[i] - np.mean(handicap_ratings)), 4),
                    "weight_carried": rng.randint(113, 133),
                    "weight_vs_avg": round(rng.gauss(0, 0.06), 4),
                    "days_since_last_run": days_since,
                    "win_odds": round(win_odds, 2),
                    "place_odds": round(place_odds, 2),
                    "implied_win_probability": round(1.0 / max(win_odds, 1.01), 4),
                    "implied_place_probability": round(1.0 / max(place_odds, 1.01), 4),
                    "race_distance": distance,
                    "n_runners": n_runners,
                    "venue_is_st": venue_is_st,
                    # Interaction features
                    "elo_x_draw": round((elo_ratings[i] - 1500) / 100 * draw_adv, 4),
                    "jockey_x_trainer": round(jk_win_30d * tr_win_30d, 4),
                    "form_x_odds": round(form_scores[i] / max(win_odds, 1.01), 4),
                    "distance_x_draw": round(distance * draw / n_runners, 4),
                    "win_rate_x_odds": round(max(0.01, true_quality[i] * 0.10 + 0.10) / max(win_odds, 1.01), 4),
                    "class_change": round(rng.choice([-2, -1, 0, 0, 0, 1]), 1),
                    "class_performance": round(np.clip(0.30 + rng.gauss(0, 0.15), 0, 1), 4),
                    "weight_change": round(rng.gauss(0, 6), 1),
                    "avg_winning_margin": round(max(0, rng.gauss(5, 4)), 2),
                    "class_x_dist": round(rng.choice([-2, -1, 0, 0, 0, 1]) * np.clip(0.30 + rng.gauss(0, 0.18), 0, 1), 4),
                    "weight_x_change": round(rng.randint(113, 133) * np.clip(rng.gauss(0, 6), -25, 25), 1),
                    "margin_x_form": round(max(0, rng.gauss(5, 4)) * (1.0 - float(form_scores[i])), 4),
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

        # ── Time-based train/validation split ──────────────────────────────────
        # Only do time split if we have race_date column (indicates real data)
        has_date_col = "race_date" in df.columns
        if has_date_col:
            try:
                sorted_idx = df["race_date"].argsort()
                split_idx = int(len(sorted_idx) * 0.8)
                train_idx = sorted_idx[:split_idx]
                val_idx = sorted_idx[split_idx:]
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                logger.info(f"Time split: {len(X_train)} train, {len(X_val)} validation (cutoff ~row {split_idx})")
            except Exception as e:
                logger.warning(f"Time split failed ({e}), using full data")
                X_train, X_val = X, X.iloc[:0]
                y_train, y_val = y, y[:0]
        else:
            X_train, X_val = X, X.iloc[:0]
            y_train, y_val = y, y[:0]

        has_val = len(X_val) > 0 and y_val.sum() > 0

        metrics = {}

        # ── XGBoost ──────────────────────────────────────────────────────────
        if XGB_AVAILABLE:
            logger.info("Training XGBoost...")
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
            self.xgb_model.fit(X_train, y_train)
            xgb_pred = self.xgb_model.predict_proba(X_train)[:, 1]
            metrics["xgb_auc_train"] = round(roc_auc_score(y_train, xgb_pred), 4)
            logger.info(f"XGBoost train AUC: {metrics['xgb_auc_train']}")
            if has_val:
                xgb_val_pred = self.xgb_model.predict_proba(X_val)[:, 1]
                metrics["xgb_auc_val"] = round(roc_auc_score(y_val, xgb_val_pred), 4)
                logger.info(f"XGBoost val AUC: {metrics['xgb_auc_val']}")

        # ── LightGBM ─────────────────────────────────────────────────────────
        if LGB_AVAILABLE:
            logger.info("Training LightGBM...")
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            self.lgb_model.fit(X_train, y_train)
            lgb_pred = self.lgb_model.predict_proba(X_train)[:, 1]
            metrics["lgb_auc_train"] = round(roc_auc_score(y_train, lgb_pred), 4)
            logger.info(f"LightGBM train AUC: {metrics['lgb_auc_train']}")
            if has_val:
                lgb_val_pred = self.lgb_model.predict_proba(X_val)[:, 1]
                metrics["lgb_auc_val"] = round(roc_auc_score(y_val, lgb_val_pred), 4)
                logger.info(f"LightGBM val AUC: {metrics['lgb_auc_val']}")

        # ── Ensemble AUC ──────────────────────────────────────────────────────
        ensemble_pred = self._ensemble_proba(X_train)
        metrics["ensemble_auc_train"] = round(roc_auc_score(y_train, ensemble_pred), 4)
        logger.info(f"Ensemble train AUC: {metrics['ensemble_auc_train']}")
        if has_val:
            ensemble_val_pred = self._ensemble_proba(X_val)
            metrics["ensemble_auc_val"] = round(roc_auc_score(y_val, ensemble_val_pred), 4)
            logger.info(f"Ensemble val AUC: {metrics['ensemble_auc_val']}")

        # ── Feature importance ────────────────────────────────────────────────
        feature_importance = self._get_feature_importance()
        if feature_importance:
            metrics["top_features"] = [f[0] for f in feature_importance[:10]]
            logger.info("Top 10 features:")
            for name, score in feature_importance[:10]:
                logger.info(f"  {name}: {score:.4f}")

        self.is_trained = True
        self.training_source = training_source
        self.training_generated_at = datetime.now().isoformat()
        return metrics

    def tune_hyperparams(self, df, n_trials: int = 20):
        """Use Optuna to improve hyperparameters. Falls back to defaults if unavailable."""
        if not OPTUNA_AVAILABLE:
            logger.info("Optuna not installed; using default hyperparameters")
            return

        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        y = df["is_top3"].values

        def objective(trial, model_type):
            if model_type == "xgb":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 4.0),
                    "random_state": 42,
                    "verbosity": 0,
                }
                model = xgb.XGBClassifier(**params)
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 64),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                    "class_weight": "balanced",
                    "random_state": 42,
                    "verbose": -1,
                }
                model = lgb.LGBMClassifier(**params)
            # Use TimeSeriesSplit for temporal consistency (3 splits)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
            return scores.mean()

        for model_type in ("xgb", "lgb"):
            try:
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda t: objective(t, model_type), n_trials=n_trials, n_jobs=1)
                best = study.best_params
                logger.info("Optuna %s best AUC=%.4f params=%s", model_type, study.best_value, best)
            except Exception as e:
                logger.warning("Optuna %s failed: %s", model_type, e)

    def _get_feature_importance(self) -> List[Tuple[str, float]]:
        """
        Aggregate feature importance across XGBoost and LightGBM models.
        Returns sorted list of (feature_name, importance_score).
        """
        imp_map: Dict[str, float] = {}
        if XGB_AVAILABLE and self.xgb_model is not None:
            xgb_imp = self.xgb_model.feature_importances_
            for name, score in zip(self.feature_cols, xgb_imp):
                imp_map[name] = imp_map.get(name, 0) + score
        if LGB_AVAILABLE and self.lgb_model is not None:
            lgb_imp = self.lgb_model.feature_importances_
            for name, score in zip(self.feature_cols, lgb_imp):
                imp_map[name] = imp_map.get(name, 0) + score
        sorted_imp = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_imp

    def compute_dynamic_weights(self, backtest_records, segment_key: str = ""):
        """Return (xgb_weight, lgb_weight) based on recent performance."""
        if not backtest_records:
            return (0.5, 0.5)
        settled = [r for r in backtest_records if getattr(r, "actual_winner", 0) > 0][-200:]
        if not settled:
            return (0.5, 0.5)
        xgb_correct = sum(1 for r in settled if getattr(r, "winner_correct", False))
        lgb_correct = sum(1 for r in settled if getattr(r, "lgb_winner_correct", False))
        total = max(xgb_correct + lgb_correct, 1)
        return (max(0.05, xgb_correct / total), max(0.05, lgb_correct / total))

    def _ensemble_proba(self, X: pd.DataFrame, weights: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Compute ensemble probability using dynamic or given weights."""
        if weights is None:
            weights = (0.5, 0.5)
        xgb_w, lgb_w = weights
        preds = []
        w_sum = 0.0
        if XGB_AVAILABLE and self.xgb_model is not None:
            preds.append(self.xgb_model.predict_proba(X)[:, 1] * xgb_w)
            w_sum += xgb_w
        if LGB_AVAILABLE and self.lgb_model is not None:
            preds.append(self.lgb_model.predict_proba(X)[:, 1] * lgb_w)
            w_sum += lgb_w
        if not preds or w_sum == 0:
            return np.full(len(X), 1.0 / 12)
        return sum(preds) / w_sum

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

        X_model = X[self.feature_cols].copy()
        # Defensive coercion: runtime feature frames may include string/datetime fields
        # (e.g. race_date/race_start_time/race_datetime). Models require numeric dtypes.
        for col in X_model.columns:
            if pd.api.types.is_datetime64_any_dtype(X_model[col]):
                # Convert datetime to unix seconds; NaT -> 0.
                X_model[col] = (X_model[col].view("int64") // 10**9)
            else:
                X_model[col] = pd.to_numeric(X_model[col], errors="coerce")

        X_model = X_model.fillna(0.0)
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
