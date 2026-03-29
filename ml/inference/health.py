"""
Health check endpoint for the ML Inference API.
Retourne les versions de modèles, taille de l'index FAISS, et lag monolith.
"""

import time
from ml.inference.schemas import HealthResponse, ModelVersions

_start_time = time.time()


async def get_health(registry) -> HealthResponse:
    """
    Construit la réponse health en interrogeant le registry et le monolith.
    """
    uptime = int(time.time() - _start_time)

    # Versions des modèles depuis le registry
    versions = ModelVersions()
    try:
        meta = registry.get_model_metadata()
        versions = ModelVersions(
            two_tower=meta.get("two_tower", "0.0.0"),
            mtl_ple=meta.get("mtl_ple", "0.0.0"),
            din=meta.get("din", "0.0.0"),
            dien=meta.get("dien", "0.0.0"),
            bst=meta.get("bst", "0.0.0"),
            geo_classifier=meta.get("geo_classifier", "0.0.0"),
        )
    except Exception:
        pass

    # Taille de l'index FAISS
    faiss_size = 0
    try:
        from ml.serving.faiss_index import get_index_size
        faiss_size = get_index_size()
    except Exception:
        pass

    # Lag monolith (secondes depuis dernier update)
    monolith_lag = 9999
    last_trained = None
    try:
        from ml.monolith.redis_store import get_last_training_ts
        last_trained = get_last_training_ts()
        if last_trained:
            import datetime
            dt = datetime.datetime.fromisoformat(last_trained)
            monolith_lag = int((datetime.datetime.utcnow() - dt).total_seconds())
    except Exception:
        pass

    # Status global
    status = "ok"
    if monolith_lag > 3600:  # lag > 1h → dégradé
        status = "degraded"

    return HealthResponse(
        status=status,
        model_versions=versions,
        faiss_index_size=faiss_size,
        last_trained_at=last_trained,
        monolith_lag_s=monolith_lag,
        uptime_s=uptime,
    )
