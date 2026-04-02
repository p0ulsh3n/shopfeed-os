"""
Spark Batch Training Configuration (archi-2026 §6.2)
======================================================
Apache Spark for large-scale batch ML training and feature computation.

Scheduled jobs:
    1. TrainRecommendationModel — every 6h on 7 days of data
    2. ComputeUserEmbeddings — every 2h
    3. VideoEmbeddings — on each upload batch
    4. DailyAnalyticsReport — midnight UTC
    5. ContentModerationBatch — hourly re-moderation

Stack:
    - Apache Spark 4 on Kubernetes
    - PySpark for data processing
    - Delta Lake for ACID storage on S3
    - Apache Arrow for data transfer to PyTorch

Requires:
    pip install pyspark>=3.5 delta-spark>=3.1
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession
    from pyspark import SparkConf
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False
    logger.warning("pyspark not installed — Spark batch jobs disabled. pip install pyspark>=3.5")


# ── Configuration ──────────────────────────────────────────────

SPARK_CONFIG = {
    "spark.app.name": "shopfeed-ml-training",
    "spark.master": os.getenv("SPARK_MASTER", "k8s://https://kubernetes.default.svc"),
    "spark.executor.memory": os.getenv("SPARK_EXECUTOR_MEMORY", "8g"),
    "spark.executor.cores": os.getenv("SPARK_EXECUTOR_CORES", "4"),
    "spark.executor.instances": os.getenv("SPARK_EXECUTOR_INSTANCES", "8"),
    "spark.driver.memory": os.getenv("SPARK_DRIVER_MEMORY", "4g"),
    # Delta Lake
    "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    # Arrow optimization for pandas/PyTorch interop
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
    # Dynamic allocation
    "spark.dynamicAllocation.enabled": "true",
    "spark.dynamicAllocation.minExecutors": "2",
    "spark.dynamicAllocation.maxExecutors": "32",
    # Serialization
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
}

# Data paths (S3 / Delta Lake)
DATA_PATHS = {
    "interactions": os.getenv("INTERACTIONS_PATH", "s3://shopfeed-data/delta/interactions"),
    "user_features": os.getenv("USER_FEATURES_PATH", "s3://shopfeed-data/delta/user_features"),
    "item_features": os.getenv("ITEM_FEATURES_PATH", "s3://shopfeed-data/delta/item_features"),
    "training_data": os.getenv("TRAINING_DATA_PATH", "s3://shopfeed-data/delta/training"),
    "embeddings": os.getenv("EMBEDDINGS_PATH", "s3://shopfeed-data/delta/embeddings"),
}


class SparkMLPipeline:
    """Spark batch ML pipeline for large-scale training data preparation.

    This handles the batch processing that Flink cannot do efficiently:
        - Historical feature computation (7-30 day windows)
        - Training data preparation with temporal splits
        - User/item embedding computation at scale
        - Delta Lake management (ACID on S3)
    """

    def __init__(self, config: Optional[dict] = None):
        self._spark = None
        self._config = {**SPARK_CONFIG, **(config or {})}

    @property
    def spark(self) -> Optional[Any]:
        """Get or create SparkSession."""
        if self._spark is None and HAS_PYSPARK:
            try:
                builder = SparkSession.builder
                for key, value in self._config.items():
                    builder = builder.config(key, value)
                self._spark = builder.getOrCreate()
                logger.info("SparkSession created: %s", self._config.get("spark.app.name"))
            except Exception as e:
                logger.error("SparkSession creation failed: %s", e)
        return self._spark

    def prepare_training_data(
        self,
        days: int = 7,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Prepare training data from the last N days of interactions.

        This reads from Delta Lake, joins user/item features,
        and outputs a dataset ready for PyTorch training.

        Temporal split:
            - Train: first 80% of time window
            - Validation: last 20%
        """
        if not self.spark:
            return None

        output = output_path or DATA_PATHS["training_data"]

        try:
            # Read interactions from Delta Lake
            interactions = self.spark.read.format("delta").load(DATA_PATHS["interactions"])

            # Filter to last N days
            from pyspark.sql import functions as F
            cutoff = F.date_sub(F.current_date(), days)
            recent = interactions.filter(F.col("event_date") >= cutoff)

            # Join with user features
            user_features = self.spark.read.format("delta").load(DATA_PATHS["user_features"])
            with_user = recent.join(user_features, on="user_id", how="left")

            # Join with item features
            item_features = self.spark.read.format("delta").load(DATA_PATHS["item_features"])
            with_all = with_user.join(item_features, on="item_id", how="left")

            # Temporal split marker
            with_split = with_all.withColumn(
                "split",
                F.when(
                    F.col("event_timestamp") < F.expr(f"date_sub(current_date(), {int(days * 0.2)})"),
                    "train",
                ).otherwise("val"),
            )

            # Write to Delta Lake
            with_split.write.format("delta") \
                .mode("overwrite") \
                .partitionBy("split", "event_date") \
                .save(output)

            count = with_split.count()
            logger.info("Training data prepared: %d rows → %s", count, output)
            return output

        except Exception as e:
            logger.error("Training data preparation failed: %s", e)
            return None

    def compute_user_embeddings(self, output_path: Optional[str] = None) -> Optional[str]:
        """Compute user embedding features at scale (batch, every 2h).

        Aggregates long-term user behavior (30 days) into feature vectors.
        """
        if not self.spark:
            return None

        output = output_path or DATA_PATHS["user_features"]

        try:
            from pyspark.sql import functions as F

            interactions = self.spark.read.format("delta").load(DATA_PATHS["interactions"])

            # Aggregate user behavior over 30 days
            user_agg = interactions.filter(
                F.col("event_date") >= F.date_sub(F.current_date(), 30)
            ).groupBy("user_id").agg(
                F.count("*").alias("total_interactions_30d"),
                F.countDistinct("item_id").alias("unique_items_30d"),
                F.avg(F.when(F.col("action") == "view", F.col("watch_duration_pct"))).alias("avg_watch_pct"),
                F.sum(F.when(F.col("action") == "like", 1).otherwise(0)).alias("total_likes_30d"),
                F.sum(F.when(F.col("action") == "buy_now", 1).otherwise(0)).alias("total_purchases_30d"),
                F.sum(F.when(F.col("action") == "share", 1).otherwise(0)).alias("total_shares_30d"),
                # Engagement rate
                (F.sum(F.when(F.col("action").isin("like", "share", "save"), 1).otherwise(0))
                 / F.count("*")).alias("engagement_rate"),
                # Category diversity (Gini coefficient proxy)
                F.countDistinct("category_id").alias("category_diversity"),
                # Recency
                F.datediff(F.current_date(), F.max("event_date")).alias("days_since_last_action"),
            )

            # Write to Delta Lake
            user_agg.write.format("delta") \
                .mode("overwrite") \
                .save(output)

            count = user_agg.count()
            logger.info("User embeddings computed: %d users → %s", count, output)
            return output

        except Exception as e:
            logger.error("User embedding computation failed: %s", e)
            return None

    def stop(self) -> None:
        """Stop SparkSession."""
        if self._spark:
            self._spark.stop()
            self._spark = None
