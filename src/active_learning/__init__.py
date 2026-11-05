"""
Active Learning package for Brain CT Hemorrhage Segmentation.
Contains uncertainty scoring (MC Dropout), pool filtering, stratified selection,
retraining loop with buffer preservation, and label-efficiency evaluation.
"""

from src.active_learning.uncertainty import MCDropoutScorer
from src.active_learning.pool_scorer import PoolScorer
from src.active_learning.selector import StratifiedSampler, RandomSampler, TopKSampler, SimulatedOracle
from src.active_learning.loop import ActiveLearningManager
from src.active_learning.visualize import (
    plot_label_efficiency_curves,
    plot_entropy_drift,
    plot_uncertainty_overlay,
    plot_stratified_vs_topk,
)

__all__ = [
    "MCDropoutScorer",
    "PoolScorer",
    "StratifiedSampler",
    "RandomSampler",
    "TopKSampler",
    "SimulatedOracle",
    "ActiveLearningManager",
    "plot_label_efficiency_curves",
    "plot_entropy_drift",
    "plot_uncertainty_overlay",
    "plot_stratified_vs_topk",
]
