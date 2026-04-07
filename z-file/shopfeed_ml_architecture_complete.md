# ShopFeed ML — Architecture Complète (Lu ligne par ligne, Avril 2026)

> **Source de vérité** : ce document est basé sur la lecture intégrale de chaque fichier Python du projet.
> Aucun résumé approximatif — uniquement ce qui existe réellement dans le code.

---

## 1. Vue d'ensemble du répertoire ML

```
ml/
├── models/          Two-Tower, DeepFM, DIN, DIEN, BST, SIM, MTL, GeoClassifier
├── training/        train.py, finetune.py, dpo.py, spark_config.py
├── datasets/        configs.py (16 datasets), loaders.py (HF + Kaggle)
├── feature_store/   encoders.py, multi_domain_encoder.py, transforms.py, pipeline.py
├── monolith/        streaming_trainer.py, cuckoo_table.py, redis_store.py, delta_model.py
├── serving/         registry.py, export_onnx.py, faiss_index.py, milvus_client.py,
│                    triton_client.py, sglang_config.py
├── inference/       app.py (FastAPI), pipeline.py (RankingPipeline), schemas.py, health.py
├── streaming/       redpanda.py, flink_features.py
├── pipelines/       kubeflow_pipeline.py, pretrain_finetune.py
├── search/          hybrid_search.py, visual_search.py, cross_modal.py, reranker.py,
│                    elasticsearch_backend.py, clip_onnx_inference.py, category_router.py,
│                    click_training.py
├── ads/             epsilon.py, ad_ranker.py, ad_retrieval.py, auction.py,
│                    budget_pacing.py, fatigue.py, targeting.py, uplift.py
├── moderation/      content_moderator.py
├── fraud/           detector.py
├── llm/             __init__.py (Llama 4 Scout vLLM client)
├── analytics/       ...
├── audio/           ...
├── cv/              clip_encoder.py
└── configs/         infrastructure.yaml, training.yaml, encoders.yaml
```

---

## 2. Les Modèles ML — Ce qui existe réellement

### 2.1 Two-Tower (Retrieval — Étape 1)
- **Fichier** : `ml/models/two_tower.py`
- **Rôle** : Réduit le catalogue (10M+ items) à ~2 000 candidats en <10ms via FAISS ANN
- **Architecture** : 2 tours encodeurs indépendants (user + item) → espace embedding partagé 256d
- **2026 upgrades** : MoE (Mixture-of-Experts, 4 experts top-2) dans les FFN + Late Interaction ColBERT-style (+8-15% recall@100)
- **Entraînement** : InfoNCE contrastive loss, température 0.07, in-batch negatives
- **Export** : ONNX pour Triton, FAISS IVF+PQ index pour >100K items

### 2.2 DeepFM (Pre-ranking — Étape 2)
- **Fichier** : `ml/models/deepfm.py`
- **Rôle** : Filtre 2 000 → 400 candidats (<20ms)
- **Architecture** : FM component (factorization machines) + Deep component (MLP)
- **Fix appliqué** : Bug #2 — les sparse_idx étaient `torch.arange(20)` identiques pour tous les batches → remplacé par hash features réels

### 2.3 MTL / PLE — Multi-Task Learning (Scoring final — Étape 3)
- **Fichier** : `ml/models/mtl_model.py`
- **Rôle** : Score les 7 tâches simultanément sur les 400 candidats (<30ms)
- **Architecture** : Progressive Layered Extraction (PLE), 2 couches, 4 experts partagés + 2 par tâche
- **7 tâches** :
  ```
  P(buy_now)      x 12  → achat immédiat
  P(purchase)     x 10  → achat panier
  P(add_to_cart)  x  8  → ajout panier
  P(save)         x  6  → wishlist
  P(share)        x  5  → partage social
  E(watch_time)   x  3  → régression (temps de vue)
  P(negative)     x -8  → skip / "pas intéressé"
  ```
- **Score final** : `Score = Σ(weight × prediction)` — context-aware (feed/marketplace/live)

### 2.4 DIN — Deep Interest Network (Marketplace ranking)
- **Fichier** : `ml/models/din.py`
- **Rôle** : CTR/CVR pour fiches produits marketplace (pages statiques)
- **Architecture** : Local Activation Unit — attention dynamique sur historique comportemental par rapport au candidat
- **Output** : P(click), P(add_to_cart), P(purchase)
- **Dataset entraînement** : Alibaba UserBehavior (100M+ comportements)

### 2.5 DIEN — Deep Interest Evolution Network
- **Fichier** : `ml/models/dien.py`
- **Rôle** : Capture l'évolution temporelle des intérêts (séquences longues)
- **Architecture** : GRU + Attention-GRU (AUGRU) pour modéliser la drift d'intérêt

### 2.6 BST — Behavior Sequence Transformer
- **Fichier** : `ml/models/bst.py`
- **Rôle** : Séquences comportementales via Transformer (recommandé pour utilisateurs nouveaux)
- **Architecture** : Transformer encoder + position embedding temporel

### 2.7 SIM — Search-based Interest Modeling
- **Fichier** : `ml/models/sim.py`
- **Rôle** : Utilisateurs avec historique long (1000+ interactions)
- **Architecture** :
  - Hard Search : top-K attention sur historique long-terme
  - Soft Search : SDIM (hash embeddings) pour représentation compressée
  - BST intégré pour court-terme (50 dernières interactions)
- **Fusion** : `[user_features, hard_repr, soft_repr, bst_repr]` → MLP → logit

### 2.8 GeoClassifier
- **Fichier** : `ml/models/geo_classifier.py`
- **Rôle** : Classification géographique pour targeting ads et recommandations régionales

---

## 3. Les Datasets — Ce qui est configuré

### Fichier `ml/datasets/configs.py` — 16 datasets enregistrés

| Dataset | Source | Taille | Usage réel dans le code |
|---------|--------|--------|------------------------|
| **Alibaba UserBehavior** | HuggingFace `kk15507/alibaba_user_behavior` | 100M+ behaviors, 987K users, 4.1M items | Pré-entraînement Two-Tower, DIN, DIEN, BST |
| **Amazon Reviews 2023** | HuggingFace `McAuley-Lab/Amazon-Reviews-2023` | 233M reviews, 43M users | NLP text embeddings, CF matrix |
| **RetailRocket** | Kaggle | 2.7M events, 1.4M visitors | Séquences comportementales DIEN |
| **DeepFashion** | HuggingFace `detection-datasets/deepfashion` | 800K+ images, 50 catégories | Fine-tuning CLIP fashion |
| **DeepFashion2** | HuggingFace `Yufei-Gao/DeepFashion2` | 491K images, segmentation | Détection vêtements dans vidéos |
| **Marqo-FashionSigLIP** | HuggingFace `Marqo/marqo-fashionSigLIP` | ViT-B-16-SigLIP | Adapter LoRA fashion (TÉLÉCHARGÉ, pas entraîné) |
| **FashionCLIP 2.0** | HuggingFace `patrickjohncyh/fashion-clip` | ViT-B/32 | Fallback mode |
| **iMaterialist Fashion** | Kaggle | 1M+ images, 228 attributs | Classificateur attributs fins |
| **Fashionpedia** | Kaggle | 48K images annotées | Segmentation pixel-level (try-on) |
| **Polyvore Outfits** | URL GitHub | 21K outfits | Cross-sell "Complete the Look" |
| **Fashion IQ** | URL GitHub | 77K triplets NL | Search conversationnel |
| **Fashion200K** | URL GitHub | 200K images Lyst.com | Retrieval attributs |
| **Food-101** | HuggingFace `ethz/food101` | 101K images, 101 catégories | Auto-classification food |
| **Recipe1M+** | URL MIT | 1M+ recettes, 13M images | Cross-sell ingrédients |
| **VIREO Food-172** | URL CityU | 110K images, 172 catégories | Spécialisation cuisine asiatique |
| **PlantNet** | HuggingFace `bertiqwerty/plantnet` | 8M+ images, 300K espèces | Identification plantes médicinales |

### Loaders implémentés dans `ml/datasets/loaders.py`

```python
# Ces classes streament RÉELLEMENT les données (pas de simulation)

AlibabaUserBehaviorLoader    # → Two-Tower pre-training
AlibabaSequenceLoader        # → DIN/DIEN/BST pre-training (5M interactions max)
ProprietaryInteractionsLoader  # → Fine-tuning sur tes données ShopFeed
BehaviorSequenceDataset      # → Traitement séquences universelles
FashionEmbeddingDataset      # → Wrapper HF pour CLIP fashion
```

---

## 4. Les 3 Types d'Entraînement — Comment ils coexistent

### TYPE 1 — Online Learning Continu (Monolith Streaming Trainer)

**Fichier** : `ml/monolith/streaming_trainer.py`

```
Redpanda topics (shopfeed.user.events, shopfeed.commerce.events, shopfeed.vendor.events)
        |
MonolithStreamingTrainer.process_event()
        | (buffer de 512 events)
_train_micro_batch()
  → CuckooEmbeddingTable.update() par item (online SGD immédiat)
  → DeltaModel gradient accumulation (4 micro-batches)
        |
Sync toutes les 60s → Redis Feature Store
Sync trending items (>50 events depuis dernier sync) → toutes les 15s
Checkpoint toutes les 1h
        |
Triton charge les nouveaux poids toutes les 5-15 min
```

- **Démarre** : dès le premier utilisateur en production, sans dataset
- **Ce qui apprend** : embeddings items/users (CuckooEmbeddingTable, 2M capacity, eviction 30j)
- **Action weights** : buy_now=12, purchase=10, cart=8, save=6, share=5, skip=-4, not_interested=-8
- **Aucune dépendance** de données pré-existantes

---

### TYPE 2 — Offline Batch Training (GPU loué, puis Kubeflow toutes les 6h)

**Fichier** : `ml/training/train.py`

**Phase A — Pré-entraînement sur données publiques** (avant prod, une fois)
```bash
# Two-Tower + MTL + DeepFM → AlibabaUserBehaviorLoader
python -m ml.training.train --model two_tower --data alibaba
python -m ml.training.train --model deepfm    --data alibaba
python -m ml.training.train --model mtl       --data alibaba

# DIN + DIEN + BST → AlibabaSequenceLoader (max 5M interactions)
python -m ml.training.train --model din  --data alibaba
python -m ml.training.train --model dien --data alibaba
python -m ml.training.train --model bst  --data alibaba
```

**Phase B — Fine-tuning sur tes propres données** (dès que tu as des interactions)
```bash
# ProprietaryInteractionsLoader — attend data/finetune/din/train.parquet
# FileNotFoundError explicite si le fichier n'existe pas (pas de simulation silencieuse)
python -m ml.training.finetune --model din --checkpoint checkpoints/din/model_best.pt
```

**Validation gate Kubeflow** (avant promotion en prod) :
- AUC > 0.82 (click-task)
- NDCG@10 > 0.71
- Latence p99 < 10ms
- Lift engagement A/B > +2%

**Schedule Kubeflow** (`ml/pipelines/kubeflow_pipeline.py`) :
```
recommendation_model  → toutes les 6h
user_embeddings       → toutes les 2h
fraud_model           → hebdomadaire (lundi)
moderation_model      → mensuel (1er du mois)
drift_check           → quotidien 6h
visual_lora_adapters  → mensuel (1er du mois 2h)
projection_head       → mensuel (1er du mois 3h)
faiss_reindex         → hebdomadaire (dimanche 4h)
```

---

### TYPE 3 — Pre-Production Fine-tuning (script `pretrain_finetune.py`)

**Fichier** : `ml/pipelines/pretrain_finetune.py`

```
Étape 1 → LoRA adapters visuels    (tes photos + titres)
Étape 2 → Projection Head 1024→512 (tes photos + titres)
Étape 3 → Fine-tune DIN/BST        (tes interactions si disponibles)
Étape 4 → DPO alignment            (si ≥500 achats + 500 skips)
Étape 5 → FAISS re-indexation      (tous tes produits)
Étape 6 → Validation recall@10 + latence
```

**Ce script est la seule chose que tu lances manuellement avant la prod.**

---

## 5. Visual Encoder — Architecture Complète

### Fichiers impliqués
- `ml/feature_store/multi_domain_encoder.py` — coeur : `EcommerceEncoder` singleton
- `ml/feature_store/encoders.py` — API publique : `encode_product_image()`, `encode_query_image()`
- `ml/cv/clip_encoder.py` — facade legacy avec mapping `category_id → domain`
- `configs/encoders.yaml` — configuration centralisée

### Architecture à 3 niveaux

```
MODÈLE DE BASE (GELÉ)
  marqo-ecommerce-embeddings-L
  652M params, 1024d output
  Pré-entraîné par Marqo sur 4M produits (Amazon, Google Shopping)
  Téléchargé depuis HuggingFace, jamais réentraîné

  +

ADAPTERS LoRA (Entraînés sur TES produits)
  fashion_lora.pt     (~2MB) → optimisé textures/couleurs mode
  electronics_lora.pt (~2MB) → optimisé formes/marques tech
  food_lora.pt        (~2MB) → optimisé packaging/couleurs
  auto_lora.pt        (~2MB) → silhouettes véhicules
  ...
  Chargés via PEFT, MERGÉS dans les poids au démarrage → zéro latence inférence

  +

PROJECTION HEAD (Apprise sur TES produits)
  Linear(1024, 512) → LayerNorm → dropout
  Entraîné par InfoNCE contrastif (image ↔ titre)
  Réduit 1024d → 512d pour compatibilité FAISS + Two-Tower
```

### Deux flux d'utilisation

```
OFFLINE (indexation produits)          ONLINE (requête utilisateur)
────────────────────────────           ────────────────────────────
encode_product_image(img, "fashion")   encode_query_image(img)
  → ecommerce-L + fashion_lora          → ecommerce-L seul (pas d'adapter)
  → 1024d → projection → 512d           → 1024d → projection → 512d
  → ~200ms (GPU, acceptable)             → <10ms (GPU, critique)
  → stocké en DB + FAISS                 → comparé au FAISS index
```

### Mapping category_id → domain (backward compat)

```python
CATEGORY_ID_TO_DOMAIN = {
    1: "fashion", 2: "electronics", 3: "food",
    4: "beauty",  5: "home",        6: "sports",
    7: "auto",    8: "baby",        9: "health",
    # 0 ou autre → "default"
}
```

---

## 6. Pipeline d'Inférence — De la Requête au Résultat (<80ms)

**Fichier** : `ml/inference/pipeline.py` — classe `RankingPipeline`

```
Requête utilisateur (user_id, context: feed|marketplace|live)
        |
ÉTAPE 1 — Two-Tower FAISS ANN (<10ms)
  Query = user embedding (Redis)
  FAISS.search(query_emb, k=2000) → 2000 candidats
  (marketplace: k=3000 pour plus de variété)

        |
ÉTAPE 2 — DeepFM Pre-ranking (<20ms)
  registry.predict_deepfm(user_id, 2000 candidats)
  → 400 candidats (500 pour marketplace)
  Fallback : premiers 400 si DeepFM indisponible

        |
ÉTAPE 3 — MTL/PLE Scoring (<30ms)
  7 tâches simultanées sur les 400 candidats
  Score context-aware selon le contexte :

  FEED          MARKETPLACE        LIVE
  watch×10      purchase×14        buy_now×15
  share×8       buy_now×13         purchase×12
  save×7        cart×11            cart×10
  cart×5        save×8             watch×9
  purchase×4    share×6            share×7
  buy_now×3     watch×5            save×4
  negative×-5   negative×-12       negative×-10

        |
ÉTAPE 4 — Context-Aware Diversity (<10ms)
  Feed:        max 1 vendor, 2 catégories, +15% boost nouveaux vendeurs
  Marketplace: max 2 vendor, 3 catégories, +25% discovery, price_variety
  Live:        max 5 vendor, pas de filtre catégorie

        |
ÉTAPE 5 — Pool Filtering (<5ms)
  L1: 200-800 impressions (nouveaux)
  L2: 1K-5K
  L3: 5K-30K
  L4: 30K-200K
  L5: 200K-2M
  L6: 2M+ (bestsellers)

        |
ÉTAPE 6 — Cross-sell Injection (<5ms)
  Si trigger buy_now dans Redis session:
  Feed: 2 items complémentaires (position 3)
  Marketplace: 3 items complémentaires (position 3)
  Live: désactivé

        |
  RÉSULTAT: RankResponse avec scores MTL détaillés + diversity flags
  SLA total: <80ms
```

---

## 7. Hybrid Search — Recherche Texte et Visuelle

**Fichier** : `ml/search/hybrid_search.py`

```
Requête texte "robe d'été en soie"
        | (parallèle, max latency = max de A,B,C — pas la somme)
A: BM25 (Elasticsearch prod / in-memory dev)
   Champs: title, description, tags, brand | k1=1.5, b=0.75
B: Text semantic (paraphrase-multilingual 768d → Milvus item_embeddings 256d)
C: Cross-modal CLIP (ViT-B-32 text → 512d → Milvus visual_embeddings 512d)
   ONNX CLIP si disponible (2-5x plus rapide)
        |
RRF Fusion (k=60) — Score(doc) = Σ 1/(60 + rank_in_list_i)
        |
CategoryRouter pre-filter (catégorie prédite depuis le texte)
        |
LambdaMART re-ranking (ml/search/reranker.py)
        |
Cross-modal enrichment (vidéos associées via cross_modal.find_videos_for_products)
        |
{products, associated_videos, query_intent, retrieval_counts, pipeline_ms}
```

---

## 8. Serving — Comment les Modèles sont Déployés

**Fichier** : `ml/serving/registry.py`

```
ModelRegistry — SUPPORTED_MODELS = [two_tower, deepfm, mtl, din, dien, bst, delta]

  ├── Two-Tower    → Triton (ONNX) ou local PyTorch fallback
  ├── DeepFM       → Triton (ONNX)
  ├── MTL          → Triton (ONNX)
  ├── DIN/DIEN/BST → Triton (ONNX)
  └── Delta        → Monolith delta embeddings (online updates)

Méthodes :
  registry.predict_deepfm(user_id, candidates)
  registry.predict_mtl(user_id, candidates, session_actions, intent_level)
  registry.apply_delta(scores, item_ids)
```

**Double index vectoriel** :
- `serving/faiss_index.py` → Two-Tower recommendation space (256d)
- `serving/milvus_client.py` → Search space : item_embeddings (256d) + visual_embeddings (512d)

---

## 9. LLM — Llama 4 Scout (UN seul modèle pour tout)

**Fichier** : `ml/llm/__init__.py`

```
Llama 4 Scout 17B-16E Instruct (Meta 2025)
├── 17B params actifs / 109B total (MoE 16 experts)
├── Multimodal natif (texte + image)
├── Contexte 10M tokens
├── Tourne sur 1 A100 80GB
└── Déployé via vLLM (port 8200, var SCOUT_URL)

ModelRouter.vision()       → quality scoring, modération, enrichissement produit
ModelRouter.text()         → ad copy, descriptions SEO, copy e-commerce
ModelRouter.reason()       → vendor insights, analyse audience, stratégie

Remplace : Qwen2.5-VL-7B + Phi-4 Mini + (anciennement Maverick)
```

---

## 10. Modération — 4 Niveaux

**Fichier** : `ml/moderation/content_moderator.py`

```
NIVEAU 1 → pHash (<1ms)          → blocage instantané (connu interdit)
NIVEAU 2 → ViT classifier (<30s) → {violence, nudity, self_harm, hate_symbols,
                                     dangerous_activity, spam, copyright,
                                     misinformation, counterfeit,
                                     regulated_product, price_manipulation}
           score > 0.85 → rejeté
NIVEAU 3 → Scout explanation     → WHY flaggé (transparence vendeur)
NIVEAU 4 → Human review queue    → si score 0.5-0.85
```

---

## 11. Fraud Detection

**Fichier** : `ml/fraud/detector.py` — LightGBM + rule-based fallback

```
34 features par utilisateur (Flink les calcule en <50ms) :
  - Comportement court-terme : likes/min, comments/min, action_interval_std
  - Signaux compte           : age, profile_completeness, followers
  - Signaux device           : is_emulator, is_vpn, fingerprint_reuse
  - Biométrie comportementale: touch_pressure_std, scroll_speed, tap_interval
  - Contenu                  : comment_duplicate_rate, like_without_view_rate
  - Transactions             : orders_24h, failed_payments_1h, addresses_7d

Actions :
  score > 0.9 → shadowban
  score > 0.7 → captcha
  score > 0.5 → review
  sinon       → allow

Training : FraudDetector.train(data.parquet, output.lgb)
  Temporal split 80/20, scale_pos_weight auto, early stopping 50 rounds
```

---

## 12. Chronologie Complète — Du Jour 0 à la Production

```
════════════════════════════════════════════════════════════════════
PHASE 0 — TÉLÉCHARGEMENTS (aucun entraînement)
════════════════════════════════════════════════════════════════════
marqo-ecommerce-embeddings-L    (1.3GB, GELÉ à vie)
paraphrase-multilingual-mpnet   (1.1GB, GELÉ)
ViT-Base (modération)           (350MB, GELÉ)
Llama 4 Scout 17B               (A100 80GB, GELÉ)
Alibaba UserBehavior            (streaming HuggingFace, pas de DL complet)

════════════════════════════════════════════════════════════════════
PHASE 1 — PRÉ-ENTRAÎNEMENT (GPU loué — RunPod/Lambda)
════════════════════════════════════════════════════════════════════
Source : Alibaba UserBehavior (streaming HuggingFace — DONNÉES RÉELLES)

train --model two_tower  → AlibabaUserBehaviorLoader    (~3h GPU)
train --model deepfm     → AlibabaUserBehaviorLoader    (~2h GPU)
train --model mtl        → AlibabaUserBehaviorLoader    (~4h GPU)
train --model din        → AlibabaSequenceLoader 5M     (~5h GPU)
train --model dien       → AlibabaSequenceLoader 5M     (~6h GPU)
train --model bst        → AlibabaSequenceLoader 5M     (~5h GPU)

Résultat → checkpoints/{model}/model_best.pt + metrics.json

════════════════════════════════════════════════════════════════════
PHASE 2 — PRE-PROD FINE-TUNING (pretrain_finetune.py)
════════════════════════════════════════════════════════════════════
Source : TES produits (photos + titres) — disponibles dès J0

Étape 1 → LoRA adapters visuels   (tes photos)          ~30min/catégorie
Étape 2 → Projection head 1024→512 (InfoNCE tes produits) ~20min
Étape 3 → Fine-tune DIN/BST        (si interactions.parquet existe)  ~2h
           SKIP AUTO si data/finetune/din/train.parquet manquant
Étape 4 → DPO alignment            (si ≥500 achats + ≥500 skips)    ~1h
           SKIP AUTO si pas assez de paires
Étape 5 → FAISS re-indexation      (tous produits → 512d)     ~1h/1M produits
Étape 6 → Validation recall@10 + latence p50

════════════════════════════════════════════════════════════════════
PHASE 3 — DÉPLOIEMENT
════════════════════════════════════════════════════════════════════
Triton Inference Server → Two-Tower, DeepFM, DIN, BST, MTL (ONNX)
vLLM (port 8200)        → Llama 4 Scout 17B
Redis                   → Feature store (<5ms reads)
Redpanda (3 brokers)    → 3 topics d'events
FAISS                   → chargé en mémoire
Elasticsearch           → index BM25 produits
Milvus                  → item_embeddings + visual_embeddings
Apache Flink (5 jobs)   → feature engineering continu
FastAPI ML              → inference/app.py

════════════════════════════════════════════════════════════════════
PRODUCTION (automatique dès le 1er jour)
════════════════════════════════════════════════════════════════════
TYPE 1 — Monolith Streaming Trainer
  Démarre immédiatement, apprend depuis le 1er clic
  Sync Redis 60s / trending 15s / checkpoint 1h / Triton reload 5-15min

TYPE 2 — Kubeflow Batch (toutes les 6h)
  Source : ClickHouse (accumule depuis J0)
  recommendation → 6h | user_embeddings → 2h | fraud → 1/semaine | ...

TYPE 3 — Visual adapters mensuels (si >50K nouveaux produits)
  visual_lora_adapters → mensuel
  faiss_reindex        → hebdomadaire (dimanche 4h)
```

---

## 13. Flux de Données Global

```
  APP (iOS/Android/Web)
      |
      | Events: like, view, cart, purchase, skip, share...
      v
  REDPANDA (3 brokers)
  shopfeed.user.events | shopfeed.commerce.events | shopfeed.vendor.events
      |                              |
      v                              v
  FLINK (<50ms)            MONOLITH STREAMING TRAINER
  - like_rate 1h           - buffer 512 events
  - trending 1h            - CuckooEmbeddingTable (online SGD)
  - fraud 1min/10s         - DeltaModel (gradient accum x4)
  - user agg 1h            - sync Redis 60s / trending 15s
      |                              |
      v                              v
  REDIS Feature Store ←─────────────┘
  user:{id}:features (<5ms)
  item:{id}:impressions
  session:{id}:cross_sell_trigger
  category:{id}:top_items
      |
      v
  CLICKHOUSE (accumule tous les events)
      |
      v (toutes les 6h)
  KUBEFLOW → Spark features → train.py → validation → ONNX → Triton


  VENDOR upload produit
      |
      v
  Modération 4 niveaux
      |
      v (si approuvé)
  encode_product_image(img, category)
  → marqo-ecommerce-L (GELÉ) + LoRA adapter + ProjectionHead
  → 512d
      |
      v
  PostgreSQL (metadata) + FAISS (512d) + Milvus visual_embeddings


  USER demande son feed (feed|marketplace|live)
      |
      v
  RankingPipeline (<80ms SLA)
  1. Two-Tower FAISS → 2000 candidats    (<10ms)
  2. DeepFM pre-rank → 400 candidats    (<20ms)
  3. MTL/PLE scoring → 7 scores x 400   (<30ms)
  4. Diversity       → vendor/cat caps  (<10ms)
  5. Pool filter     → L1-L6            (<5ms)
  6. Cross-sell      → items complement  (<5ms)
      |
      v
  RankResponse: items + MTL scores + diversity flags


  USER recherche "robe d'été en soie"
      |
      v (parallèle)
  BM25 (ES) | Text 768d→Milvus | CLIP 512d→Milvus
      |
      v
  RRF fusion k=60 → CategoryRouter → LambdaMART
      |
      v
  {products, videos, intent, pipeline_ms}
```

---

## 14. Gotchas et Points Critiques

> **ProprietaryInteractionsLoader** lève `FileNotFoundError` si `data/finetune/{model}/train.parquet` manque. Pas de données simulées — c'est intentionnel.

> **Cold Start J0** : DIN/BST/DeepFM sont pré-entraînés sur Alibaba (comportements e-commerce généraux). Ils fonctionnent immédiatement, mais ne connaissent pas TES utilisateurs. Le Monolith personnalise dès le 1er clic réel.

> **LoRA Merge** : adapters LoRA mergés dans les poids au démarrage (`merge_and_unload()`). À l'inférence = modèle standard, zéro surcoût.

> **DOUBLE FAISS** : `serving/faiss_index.py` (256d, recommendation) DISTINCT de `serving/milvus_client.py` (512d visual + 256d text, search).

> **Fallbacks** partout : DeepFM → premiers 400. MTL → `_popularity_fallback()` Redis impressions. LightGBM fraud → règles déterministes. Cross-modal CLIP → PyTorch fallback si ONNX non dispo.

> **DPO** : skip automatique si < 500 paires (achats+skips). Pas d'erreur — juste ignoré dans `pretrain_finetune.py`.

> **Kubeflow validation gate** : AUC > 0.82, NDCG@10 > 0.71, latence p99 < 10ms, lift A/B > +2%. Bloc la promotion si non atteint.
