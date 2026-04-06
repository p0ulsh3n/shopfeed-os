# ShopFeed OS — ML Training
#
# NOTE: Model architectures (nn.Module classes) have moved to ml/models/.
# This package now contains ONLY training orchestration:
#   - train.py       — Main training loop & ModelTrainer
#   - finetune.py    — LoRA fine-tuning pipeline
#   - spark_config.py — Spark batch processing config
#
# For backward compatibility, models can still be imported from here:
#   from ml.training.two_tower import TwoTowerModel  ← deprecated
#   from ml.models.two_tower import TwoTowerModel     ← preferred
