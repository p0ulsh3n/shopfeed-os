"""
ML Model Architectures — ShopFeed Recommendation System
=========================================================
Pure nn.Module definitions. No training logic here.

Models:
    TwoTowerModel  — User-Item dual encoder (retrieval)
    DeepFM         — Feature interaction model (pre-ranking)
    MTLModel       — Multi-task PLE (ranking)
    DINModel       — Deep Interest Network (attention-based ranking)
    DIENModel      — Deep Interest Evolution Network (sequential ranking)
    BSTModel       — Behavior Sequence Transformer (transformer ranking)
    SIMModel       — Search-based Interest Model (long-term history)
    GeoClassifier  — Geographic tier classification
"""

from ml.models.two_tower import TwoTowerModel
from ml.models.deepfm import DeepFM
from ml.models.mtl_model import MTLModel, TASK_NAMES, NUM_TASKS, TASK_CONFIGS
from ml.models.din import DINModel, DINLoss
from ml.models.dien import DIENModel, DIENLoss
from ml.models.bst import BSTModel, BSTLoss
from ml.models.sim import SIMModel
from ml.models.geo_classifier import GeoClassifier

__all__ = [
    "TwoTowerModel",
    "DeepFM",
    "MTLModel", "TASK_NAMES", "NUM_TASKS", "TASK_CONFIGS",
    "DINModel", "DINLoss",
    "DIENModel", "DIENLoss",
    "BSTModel", "BSTLoss",
    "SIMModel",
    "GeoClassifier",
]
