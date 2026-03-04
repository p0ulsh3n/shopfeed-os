# ShopFeed OS

> Commerce Infini Platform — Inspired by ByteDance / Douyin / TikTok Shop
> Python Microservices · ML Pipeline · WebSocket Real-Time · Live Commerce

---

## 🏗️ Architecture

```
shopfeed-os/
├── services/                    # 11 FastAPI microservices
│   ├── api_gateway/             # JWT auth, rate limiting, routing
│   ├── feed_service/            # Feed pipeline <80ms, Session State WS
│   │   └── pipeline.py          # 4-stage: Retrieval → Pre-Rank → MTL → Re-Rank
│   ├── live_service/            # WebSocket live commerce (vendor + viewer)
│   ├── product_service/         # CRUD, variants, vision pipeline
│   ├── user_service/            # Auth, Intent Graph profiles, follows
│   ├── order_service/           # Cart, checkout, payments (Stripe/CinetPay)
│   ├── geosort_service/         # Geo classification <500ms, NLP geocoder
│   ├── moderation_service/      # SightEngine, CLIP, NumberGuard
│   ├── search_service/          # Full-text search, autocomplete
│   ├── notification_service/    # Push FCM, in-app WS, live reminders
│   └── analytics_service/       # Vendor dashboard, audience analytics
│
├── ml/                          # ML Pipeline (trains on rented GPU servers)
│   ├── training/
│   │   ├── two_tower.py         # Two-Tower retrieval (contrastive, FAISS)
│   │   ├── deepfm.py            # DeepFM ranking (FM + DNN)
│   │   ├── mtl_model.py         # PLE 7-task scoring (commerce formula)
│   │   └── train.py             # GPU training orchestrator (AMP, ONNX)
│   ├── feature_store/           # Feature engineering pipeline
│   ├── serving/                 # Model registry, FAISS retrieval
│   └── monolith/                # Online streaming training (Kafka → SGD)
│
├── shared/                      # Shared between all services
│   ├── models/                  # Pydantic data contracts
│   ├── events/                  # Kafka event definitions (8 topics)
│   └── db/                      # Connection factories (PG, Redis, Kafka)
│
└── infra/                       # Infrastructure configs
    ├── postgres/                # SQL migrations (12 tables, pgvector)
    ├── kafka/                   # Topic definitions
    ├── redis/                   # 4 instances config
    └── k8s/                     # Kubernetes manifests + HPA
```

## 🧠 ML Pipeline — 3-Speed Architecture

| Speed | Latency | Description | Technology |
|-------|---------|-------------|------------|
| **V1** | <50ms | Session State | Redis + WebSocket push |
| **V2** | 5-15min | Online Training (Monolith) | Kafka → Flink → Incremental SGD |
| **V3** | Daily | Batch Training | Two-Tower + DeepFM + MTL on GPU |

**Score Final** = `Model_Batch(V3)` + `Delta_Online(V2)` + `Session_Boost(V1)`

### Models

| Model | Stage | Purpose | Architecture |
|-------|-------|---------|-------------|
| Two-Tower | Retrieval | 10M→2K candidates in <10ms | Dual encoder + FAISS ANN |
| DeepFM | Pre-Ranking | CTR prediction for ~400 candidates | FM + DNN |
| MTL/PLE | Ranking | 7-task commerce scoring | Progressive Layered Extraction |
| Monolith | Online | Real-time delta correction | Streaming SGD |

### Commerce Score — Section 02

```
Score = P(buy_now)×12 + P(purchase)×10 + P(add_to_cart)×8
      + P(save)×6 + P(share)×5 + E(watch_time)×3 - P(negative)×8
```

## 🚀 Training on GPU Servers

```bash
# Train Two-Tower retrieval model
python -m ml.training.train --model two_tower --epochs 20 --batch-size 2048

# Train DeepFM ranking model
python -m ml.training.train --model deepfm --epochs 15 --lr 1e-3

# Train MTL scoring model
python -m ml.training.train --model mtl --epochs 10 --lr 1e-4

# Build FAISS index after Two-Tower training
python -m ml.training.train --model two_tower --build-index
```

## 🛠️ Stack Technique

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Database | PostgreSQL + pgvector |
| Cache | Redis (4 instances) |
| Queue | Apache Kafka (8 topics) |
| Search | Elasticsearch + Typesense |
| ML Training | PyTorch 2.5+ (AMP, ONNX) |
| ML Retrieval | FAISS (IVF + PQ) |
| Vision | CLIP (SigLIP) + BLIP-2 |
| Embeddings | sentence-transformers |
| Container | Docker + Kubernetes |

## 📡 Real-Time (WebSocket)

| Endpoint | Description |
|----------|-------------|
| `ws/session/{user_id}` | Feed Session State (re-rank on action) |
| `ws/live/{id}/vendor` | Vendor live dashboard (metrics push 5s) |
| `ws/live/{id}/viewer` | Viewer interactions (comments, purchases) |
| `ws/notifications/{user_id}` | In-app notification delivery |
| `ws/analytics/{vendor_id}` | Real-time analytics updates |

## 🔒 License

Proprietary — All rights reserved.
# shopfeed-os
