"""
HKJC 賽馬預測系統 - 模型模組
Model package: training, prediction, and backtesting.
"""
from model.trainer import EnsembleTrainer
from model.predictor import RacePredictor, PredictionResult
from model.backtest import Backtester, BacktestResult
from model.self_optimizer import StrategyProfile, StrategySelfOptimizer

__all__ = [
    "EnsembleTrainer",
    "RacePredictor",
    "PredictionResult",
    "Backtester",
    "BacktestResult",
    "StrategyProfile",
    "StrategySelfOptimizer",
]
