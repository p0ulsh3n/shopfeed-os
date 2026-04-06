# ShopFeed ML — Notebooks (EDA & Evaluation)
#
# These notebooks are for EXPLORATION & EVALUATION ONLY — not production code.
# All production logic lives in ml/ as .py modules.
#
# ┌──────────────────────────────────────────────────────────────┐
# │  WORKFLOW                                                    │
# │                                                              │
# │  1️⃣ AVANT entraînement → 01_data_exploration.ipynb           │
# │     ├── Distribution des labels (imbalance)                  │
# │     ├── Valeurs manquantes (heatmap)                         │
# │     ├── Corrélations entre features                          │
# │     ├── Outliers (boxplots)                                  │
# │     └── Patterns temporels                                   │
# │                                                              │
# │  2️⃣ ENTRAÎNEMENT → python -m ml.training.train               │
# │     └── Sauvegarde training_history.json automatiquement     │
# │                                                              │
# │  3️⃣ APRÈS entraînement → 02_model_evaluation.ipynb           │
# │     ├── Learning curves (overfitting detection)              │
# │     ├── AUC per task (click, cart, purchase)                 │
# │     ├── Comparaison des modèles (bar charts)                │
# │     ├── Train vs Test performance                            │
# │     └── Learning rate schedule                               │
# │                                                              │
# │  4️⃣ ANALYSE → 03_feature_analysis.ipynb                      │
# │     ├── Feature distributions par label                      │
# │     ├── Corrélation feature ↔ target                        │
# │     └── Scatter plots                                        │
# │                                                              │
# │  5️⃣ PRODUCTION → 04_drift_visualization.ipynb                │
# │     ├── Feature distribution drift                           │
# │     ├── PSI scores over time                                 │
# │     └── Alertes de dégradation                               │
# └──────────────────────────────────────────────────────────────┘
#
# Usage:
#   cd notebooks/
#   jupyter lab
#
# All notebooks import from the parent ml/ package:
#   import sys; sys.path.insert(0, '..')
#   from ml.models import TwoTowerModel
