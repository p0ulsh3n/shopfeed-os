# Guide RunPod & Coûts Entraînement Pré-Production — ShopFeed OS (Avril 2026)

> [!CAUTION]
> Ce document a été refait après un audit complet du code source.
> Le projet contient **16 datasets** dans `ml/datasets/configs.py` totalisant
> **350M+ d'interactions + 11M+ d'images + 1M+ de recettes**.
> Les estimations précédentes sous-estimaient massivement l'ampleur des données.

---

## PARTIE 1 — Comprendre RunPod : Que Choisir ?

### Menu RunPod (Sidebar)

```
The Hub
  ├─ Serverless repos     ← Déploiement code sans serveur
  ├─ Pod templates        ← Sauvegarder ta config Docker
  └─ Public endpoints     ← APIs publiques (Whisper, LLMs...)

Resources
  ├─ Serverless           ← Exécution à la demande (cold start)
  ├─ Pods                 ← Serveurs GPU loués 24/7 ✅ TON CHOIX
  ├─ Clusters             ← Multi-GPU NVLink interconnectés
  ├─ Storage              ← Volumes réseau persistants ✅ ESSENTIEL
  ├─ Fine tuning          ← UI de fine-tuning guidé
  └─ My templates         ← Tes Docker templates sauvegardés
```

### ❌ Serverless — PAS pour ShopFeed
- Démarrage à froid **10-60 secondes** → Feed doit répondre en **< 10ms**
- Triton, vLLM, FAISS doivent tourner **en permanence** en mémoire
- Aucun maintien d'état entre les requêtes

### ✅ Pods — C'EST CE QU'IL TE FAUT
- Serveur Linux dédié avec GPU attaché, accès SSH complet
- Tourne **24h/7j** tant que tu décides
- Plans : On-Demand / 3 mois / **6 mois** ← ton choix / 1 an / Spot

| Plan | A100 SXM | H200 SXM | Interruptible ? |
| :--- | :--- | :--- | :--- |
| On-Demand | $1.49/hr | $3.59/hr | Non |
| 3 months savings | $1.30/hr | $3.37/hr | Non |
| **6 months savings** | **$1.27/hr** | **$3.12/hr** | **Non ✅** |
| 1 year savings | $1.22/hr | Contact sales | Non |
| **Spot** | **$0.95/hr** | **$2.29/hr** | **Oui ⚠️** |

### ❌ Clusters — PAS pour ShopFeed maintenant
- Conçu pour entraîner des LLMs from scratch sur 100+ GPUs
- Tes modèles de ranking sont < 100M params → 1 GPU suffit largement

### Écran "Configure Deployment" — Explication

```
┌────────────────────────────────────────────────────────────┐
│  Pod name : rich_blue_mongoose                              │
│  → Nom du serveur. Renomme en "shopfeed-inference-node"     │
│    ou "shopfeed-ai-node"                                    │
├────────────────────────────────────────────────────────────┤
│  Pod template : Runpod Pytorch 2.4.0 ✅                     │
│  runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04  │
│  → Image Docker avec Ubuntu + CUDA 12.4 + Python 3.11 +    │
│    PyTorch 2.4.0 pré-installés. C'EST LE BON pour les      │
│    2 nœuds. Tu installeras vLLM/Triton/Ray par-dessus.     │
├────────────────────────────────────────────────────────────┤
│  GPU count : [1] 2 3 4 5 6 7                               │
│  → Choisis 1 pour chaque Pod.                              │
│    1× A100 pour le Nœud #1 (Triton)                        │
│    1× H200 pour le Nœud #2 (Scout + Média)                 │
└────────────────────────────────────────────────────────────┘
```

### Network Volume — Stockage Partagé Entre Tes 2 Pods

```
Propriétés :
  ✅ Monté dans /workspace
  ✅ Persistant (ne s'efface PAS quand le Pod s'arrête)
  ✅ Partageable entre plusieurs Pods du même datacenter
  ✅ $0.07/GB/mois (< 1 TB)  |  $0.05/GB/mois (> 1 TB)

Stocker sur le Network Volume :
  ├─ Llama 4 Scout FP8 quantifié         (~110 GB)
  ├─ Tous les checkpoints modèles        (~5-10 GB)
  ├─ Index FAISS                          (~5-10 GB)
  ├─ Datasets d'entraînement (Parquet)    (~50-100 GB)
  └─ Datasets images (DeepFashion etc.)   (~200-500 GB)
```

> [!IMPORTANT]
> **Créer le Network Volume AVANT les Pods**, dans le même datacenter.
> Taille recommandée : **500 GB** → ~$35/mois.
> Puis attacher ce volume aux 2 Pods lors de leur création.

---

## PARTIE 2 — Inventaire COMPLET des Datasets du Projet

> Source : `ml/datasets/configs.py` — 16 datasets référencés

### Section 32 — Datasets E-Commerce Généraux

| # | Dataset | Source | Taille | Utilisation dans ShopFeed |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Alibaba UserBehavior** | HuggingFace | **100M+ behaviors**, 987K users, 4.1M items | Pré-entraînement DIN/DIEN/BST (séquences comportementales) |
| 2 | **Amazon Reviews 2023** | HuggingFace | **233M+ reviews**, 43M users, 9.9M items | Embeddings texte NLP, matrice user-item pour CF |
| 3 | **RetailRocket** | Kaggle | **2.7M events**, 1.4M visitors, 417K items | Entraînement comportemental DIEN (view/cart/transaction) |

### Section 35 — Datasets Mode (Fashion)

| # | Dataset | Source | Taille | Utilisation dans ShopFeed |
| :--- | :--- | :--- | :--- | :--- |
| 4 | **DeepFashion** | HuggingFace | **800K+ images**, 50 catégories, 1000 attributs | Pré-entraînement CLIP fashion + classifieur catégories |
| 5 | **DeepFashion2** | HuggingFace | **491K images**, 13 catégories, 801K items annotés | Segmentation vêtements, auto-tag produits |
| 6 | **Marqo-FashionSigLIP** | HuggingFace | Modèle ViT-B-16-SigLIP | BASE pour embeddings visuels mode (+22% recall) |
| 7 | **FashionCLIP 2.0** | HuggingFace | ViT-B/32, 800K+ produits Farfetch | Backup embedding si SigLIP indisponible |
| 8 | **iMaterialist** | Kaggle | **1M+ images**, 228 attributs, 8 groupes | Classifieur attributs fins (matière, couleur, motif, style) |
| 9 | **Fashionpedia** | Kaggle | **48K images**, 27 catégories, 294 attributs | Segmentation pixel-level pour virtual try-on |
| 10 | **Polyvore Outfits** | GitHub | **21K outfits**, 365K items | Cross-sell "Complete the Look" |
| 11 | **Fashion IQ** | GitHub | **77K triplets** NL feedback | Recherche conversationnelle ("pareil mais plus long et bleu") |
| 12 | **Fashion200K** | GitHub | **200K+ images**, 9 classes | Retrieval visuel par attributs |

### Section 35 — Datasets Food & Wellness

| # | Dataset | Source | Taille | Utilisation dans ShopFeed |
| :--- | :--- | :--- | :--- | :--- |
| 13 | **Food-101** | HuggingFace | **101K images**, 101 catégories | Auto-classification food dans vidéos/photos vendeurs |
| 14 | **Recipe1M+** | MIT CSAIL | **1M+ recettes, 13M images** | Enrichissement descriptions produits alimentaires |
| 15 | **VIREO Food-172** | CityU HK | **110K images**, 172 catégories, 353 ingrédients | Spécialisation cuisine asiatique/africaine |
| 16 | **PlantNet** | HuggingFace | **8M+ images**, 300K espèces | Auto-identification plantes médicinales |

### Volumétrie Totale

```
INTERACTIONS COMPORTEMENTALES :
  Alibaba UserBehavior    : 100,000,000+ interactions
  Amazon Reviews 2023     : 233,000,000+ reviews
  RetailRocket            :   2,700,000  events
  ─────────────────────────────────────────────────
  TOTAL INTERACTIONS      : ~335,700,000

IMAGES :
  DeepFashion             :    800,000+
  DeepFashion2            :    491,000
  iMaterialist            :  1,000,000+
  Fashionpedia            :     48,000
  Fashion200K             :    200,000+
  Food-101                :    101,000
  Recipe1M+ images        : 13,000,000
  VIREO Food-172          :    110,000
  PlantNet                :  8,000,000+
  ─────────────────────────────────────────────────
  TOTAL IMAGES            : ~23,750,000

TEXTES / TRIPLETS :
  Polyvore Outfits        :    365,000 items
  Fashion IQ              :     77,000 triplets
  Recipe1M+ recettes      :  1,000,000+ recettes
```

> [!WARNING]
> **La volumétrie est MASSIVE.** Si tu télécharges tout, tu as besoin de :
> - ~50-80 GB pour les interactions (Parquet compressé)
> - ~500-1000 GB pour toutes les images (JPEG/PNG non compressé)
> - ~500 GB total en stockage réseau est un MINIMUM
>
> **Stratégie : Ne pas tout télécharger.** Utiliser le **streaming mode**
> de HuggingFace `datasets` pour traiter les données en flux sans tout stocker.

---

## PARTIE 3 — Pipeline d'Entraînement (Tiré du Code Source)

Le pipeline est en 4 phases selon le code dans `scripts/` :

```
Phase 1a : pretrain.py → Two-Tower sur Alibaba (100M interactions)
Phase 1b : (à implémenter) → CLIP fashion sur DeepFashion2 + iMaterialist
Phase 1c : pretrain.py → DIN/DIEN/BST sur Alibaba (séquences temporelles)
Phase 2  : train.py    → DeepFM, MTL/PLE, SIM, Ad Ranker, Fraude, Geo
Phase 3  : finetune.py → LoRA fine-tuning sur données propriétaires ShopFeed
Phase 4  : export_triton.py → Export ONNX + TensorRT + FAISS
```

---

### Phase 1a — Pré-entraînement Two-Tower (100M+ interactions)

> Source : `scripts/pretrain.py` → `pretrain_two_tower()`

| | Détails |
| :--- | :--- |
| **Dataset** | Alibaba UserBehavior (100M+ behaviors HuggingFace streaming) |
| **Modèle** | Two-Tower (user_dim=764, item_dim=1348, embedding=256) |
| **Tâche** | Contrastive learning, in-batch negatives |
| **GPU** | 1× A100 Spot ($0.95/hr) |
| **Batch size** | 2048 (configurable dans `TrainConfig`) |
| **Epochs** | 5-10 (config par défaut : 20, mais 5 suffit pour le pré-entraînement) |
| **Temps estimé** | ~8-15h (100M interactions × 5 epochs, batch 2048, AMP FP16) |
| **Coût** | **~$8 - $14** |

---

### Phase 1c — Pré-entraînement DIN/DIEN/BST (séquences, 100M+)

> Source : `scripts/pretrain.py` → `pretrain_behavior_models()`

Les 3 modèles sont entraînés **séquentiellement** sur le même Pod.

| Modèle | Params | Particularité | Temps estimé |
| :--- | :--- | :--- | :--- |
| **DIN** | ~25M | Attention sur historique court | ~6-10h |
| **DIEN** | ~30M | GRU évolution d'intérêt + aux loss | ~8-12h |
| **BST** | ~25M | Transformer séquentiel multi-head | ~6-10h |
| **TOTAL Phase 1c** | | 3 modèles séquentiels | **~20-32h** |

| | Détails |
| :--- | :--- |
| **Dataset** | Alibaba UserBehavior → BehaviorSequenceDataset (max_seq_len=200) |
| **Taille processée** | 100M interactions → ~60-80M séquences d'entraînement |
| **GPU** | 1× A100 Spot ($0.95/hr) |
| **Coût** | **~$19 - $30** |

---

### Phase 1b — Embeddings Vision : CLIP/SigLIP + Classifieurs (Images)

> **Pas codé dans `pretrain.py` mais référencé dans les configs.**
> Il s'agit de fine-tuner les modèles visuels sur les datasets mode/food.

#### a) Fashion SigLIP / CLIP fine-tuning

| | Détails |
| :--- | :--- |
| **Modèle base** | Marqo-FashionSigLIP (ViT-B-16-SigLIP, pré-entraîné) |
| **Datasets** | DeepFashion (800K) + DeepFashion2 (491K) + iMaterialist (1M+) |
| **Tâche** | Contrastive image-text learning adapté aux produits africains |
| **Total images** | ~2.3M images fashion |
| **GPU** | 1× A100 Spot ($0.95/hr) |
| **Temps estimé** | ~15-25h (2.3M images × 3 epochs, batch 256, ViT-B FP16) |
| **Coût** | **~$14 - $24** |

#### b) Classifieur Food (Food-101 + VIREO)

| | Détails |
| :--- | :--- |
| **Modèle** | ViT-Base ou ResNet fine-tuné sur CLIP |
| **Datasets** | Food-101 (101K) + VIREO Food-172 (110K) |
| **Total images** | ~211K images |
| **GPU** | 1× A100 Spot ($0.95/hr) |
| **Temps estimé** | ~2-4h |
| **Coût** | **~$2 - $4** |

#### c) Classifieur PlantNet (optionnel — si catégorie wellness)

| | Détails |
| :--- | :--- |
| **Dataset** | PlantNet (8M+ images, streaming) |
| **Tâche** | Classification top-500 espèces médicinales |
| **GPU** | 1× A100 Spot ($0.95/hr) |
| **Temps estimé** | ~10-20h (sous-échantillon 500K-1M images, 3 epochs) |
| **Coût** | **~$10 - $19** |

#### d) Fashion IQ / Polyvore (recherche conversationnelle + cross-sell)

| | Détails |
| :--- | :--- |
| **Datasets** | Fashion IQ (77K triplets) + Polyvore (365K items, 21K outfits) |
| **Tâche** | Fine-tuner CLIP pour retrieval compositionnel NL |
| **GPU** | 1× A100 Spot ($0.95/hr) |
| **Temps estimé** | ~3-5h |
| **Coût** | **~$3 - $5** |

---

### Phase 2 — Entraînement des Autres Modèles (From Scratch)

> Source : `ml/training/train.py` → classe `Trainer`

| Modèle | Params | Dataset | Epochs | Temps | Coût |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepFM** | ~20M | Alibaba/Synthetic (features hashées) | 20 | ~4-6h | ~$4-6 |
| **MTL/PLE** | ~15M | Alibaba (7 tâches multi-labels) | 10 | ~5-8h | ~$5-8 |
| **SIM** | ~35M | Alibaba (historique long SDIM) | 10 | ~6-10h | ~$6-10 |
| **Ad Ranker (EPSILON)** | ~20M | Synthetic (pCTR/pCVR/pROAS/pStoreVisit) | 10 | ~3-5h | ~$3-5 |
| **Uplift T-Learner** | ~3M | Synthetic (traitement/contrôle) | 5 | ~1h | ~$1 |
| **LightGBM Fraude** | ~2M | Synthetic (35 features fraude) | N/A (GBM) | ~30min | ~$0.50 |
| **GeoClassifier** | ~3M | Données géo (L1→L4 classification) | 10 | ~30min | ~$0.50 |
| **FAISS Index Build** | — | Embeddings Two-Tower | — | ~1h | ~$1 |
| **TOTAL Phase 2** | | | | **~21-32h** | **~$21-32** |

---

### Phase 3 — Fine-Tuning LoRA sur Données Propriétaires ShopFeed

> Source : `ml/training/finetune.py` + `scripts/finetune_run.py`

#### a) LoRA Fine-tuning des modèles de ranking (DIN/DIEN/BST)

> Utilise la classe `FineTuneTrainer` avec `LoRALinear` (rank=8, alpha=16)
> Gèle les couches basses → entraîne < 1% des params

| | Détails |
| :--- | :--- |
| **Modèles** | DIN, DIEN, BST (séquentiels), Two-Tower, MTL/PLE |
| **Config** | LoRA rank=8, alpha=16, dropout=0.05, only upper MLP layers |
| **Données** | Données propriétaires ShopFeed (premières ~10K-50K interactions bêta) |
| **GPU** | 1× A100 Spot ($0.95/hr) |
| **Temps estimé** | ~5-10h (5 modèles × 3 epochs × ~10K-50K données) |
| **Coût** | **~$5 - $10** |

#### b) Fine-tuning Llama 4 Scout (QLoRA)

| | Détails |
| :--- | :--- |
| **Modèle** | Llama 4 Scout 17B-16E MoE (109B total) en QLoRA 4-bit |
| **Tâches** | Quality scoring, enrichissement produit, copywriting pub, modération, recherche conv., insights vendeur, catégorisation |
| **Données** | ~50K-100K exemples préparés (textes/images produits africains) |
| **GPU** | 1× A100 Spot ($0.95/hr) — QLoRA charge le modèle en 4-bit (~60GB VRAM, tient sur A100 80GB) |
| **Libraire** | Unsloth (2× plus rapide) ou HuggingFace PEFT |
| **Temps estimé** | ~15-40h (100K exemples × 3 epochs, LoRA r=16) |
| **Coût** | **~$14 - $38** |

---

### Phase 4 — Export, Compilation & Validation

> Source : `scripts/export_triton.py` + `ml/serving/export_onnx.py`

| Tâche | Temps | Coût |
| :--- | :--- | :--- |
| Export ONNX des 12 modèles ranking | ~30-60 min | ~$0.50-1 |
| Compilation TensorRT (1ère fois) | ~1-2h | ~$1-2 |
| Tests de validation inférence | ~1h | ~$1 |
| Build FAISS index final | ~30 min | ~$0.50 |
| Test de charge (load testing) | ~2h | ~$2 |
| **TOTAL Phase 4** | **~5-7h** | **~$5-7** |

---

## PARTIE 4 — Récapitulatif Coût Total Pré-Production

| Phase | Description | Datasets/Données | GPU | Heures | Coût |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1a** | Pré-train Two-Tower | Alibaba (100M) | A100 Spot | ~8-15h | **$8-14** |
| **1b-Fashion** | Fine-tune SigLIP/CLIP mode | DeepFashion+iMat (2.3M imgs) | A100 Spot | ~15-25h | **$14-24** |
| **1b-Food** | Classifieur food | Food-101+VIREO (211K imgs) | A100 Spot | ~2-4h | **$2-4** |
| **1b-Plant** | Classifieur plantes (optionnel) | PlantNet (sous-ensemble 1M) | A100 Spot | ~10-20h | **$10-19** |
| **1b-FashionIQ** | Retrieval conversationnel | FashionIQ+Polyvore (442K) | A100 Spot | ~3-5h | **$3-5** |
| **1c** | Pré-train DIN/DIEN/BST | Alibaba séquences (100M) | A100 Spot | ~20-32h | **$19-30** |
| **2** | Train ranking + ads + fraude | Alibaba + Synthetic | A100 Spot | ~21-32h | **$21-32** |
| **3a** | LoRA ranking (propre données) | Bêta ShopFeed (~50K) | A100 Spot | ~5-10h | **$5-10** |
| **3b** | QLoRA Scout LLM | Données ShopFeed (~100K) | A100 Spot | ~15-40h | **$14-38** |
| **4** | Export + Build + Tests | — | A100 Spot | ~5-7h | **$5-7** |
| | | | | | |
| **TOTAL GPU** | | | | **~104-190h** | **~$101-$183** |
| **Stockage** | Network Volume ~500 GB pendant 2 mois | | | | **$70** |
| **Marge (bugs, re-runs, debug)** | ~30% sécurité | | | | **~$50** |
| | | | | | |
| **BUDGET TOTAL PRÉ-PRODUCTION** | | | | | **~$220 - $300** |

> [!TIP]
> **Budget recommandé : ~$300** (avec marge confortable pour les itérations).
>
> Ce budget couvre :
> - ✅ 16 datasets dont 100M+ interactions Alibaba + 2.3M images fashion
> - ✅ 12 modèles de ranking entraînés
> - ✅ Embeddings visuels CLIP/SigLIP fine-tunés pour la mode africaine
> - ✅ Classifieurs food + plantes entraînés
> - ✅ Scout QLoRA fine-tuné sur tes données
> - ✅ FAISS index construit
> - ✅ Tout exporté en ONNX + TensorRT
> - ✅ Validé et testé en charge

---

## PARTIE 5 — Calendrier Pré-Production Recommandé

```
SEMAINE 1 — Préparation ($0)
├─ □ Créer Network Volume RunPod (500 GB, même datacenter)
├─ □ Préparer les datasets de fine-tuning Scout (exemples annotés)
├─ □ Collecter ~50K interactions bêta (ou synthetic fallback)
└─ □ Valider les scripts d'entraînement localement (CPU, batch réduit)

SEMAINE 2 — Pré-entraînement Phase 1 (~$40-68)
├─ □ Lancer Pod A100 Spot → Phase 1a (Two-Tower, 100M interactions)
├─ □ → Phase 1c (DIN/DIEN/BST séquentiellement)
├─ □ Sauvegarder checkpoints sur Network Volume
└─ □ Script auto-stop Pod via RunPod SDK à la fin

SEMAINE 3 — Vision + Ranking (~$40-65)
├─ □ Phase 1b : Fine-tune SigLIP sur DeepFashion+iMaterialist
├─ □ Phase 1b : Classifieurs Food-101 + VIREO
├─ □ Phase 2 : DeepFM, MTL/PLE, SIM, Ad Ranker, Fraude, Geo
└─ □ Valider les métriques (AUC > 0.70, LogLoss < 0.50)

SEMAINE 4 — Fine-tuning + Export (~$24-55)
├─ □ Phase 3a : LoRA fine-tuning ranking sur données ShopFeed
├─ □ Phase 3b : QLoRA Scout sur données annotées
├─ □ Phase 4 : Export ONNX, TensorRT, FAISS
├─ □ Test de charge (latence < 10ms feed, < 100ms Scout)
└─ □ Deploy sur les 2 Pods production (A100 + H200)

LANCEMENT 🚀
```

---

## PARTIE 6 — Optimisations de Coût

1. **Spot Instances ($0.95/hr vs $1.49/hr = -36%)** :
   Tous les entraînements utilisent des Spot. Le checkpointing intégré
   dans `train.py` (`_save_checkpoint()`) permet de reprendre si interrompu.

2. **HuggingFace Streaming Mode** :
   Les gros datasets (Alibaba 100M, Amazon 233M, PlantNet 8M) sont chargés
   en **streaming** (`streaming=True` dans `loaders.py`) → pas besoin de tout
   télécharger sur le disque. Économie de ~500 GB de stockage.

3. **Mixed Precision (AMP FP16)** :
   Activé par défaut dans `Trainer.__init__()` → `mixed_precision=True`.
   Réduit le temps d'entraînement de ~40-50% et la VRAM de ~30%.

4. **Auto-stop Pod à la fin du script** :
   ```python
   # Ajouter à la fin de chaque script d'entraînement :
   import runpod
   runpod.api_key = "TON_API_KEY"
   runpod.stop_pod(runpod.get_current_pod_id())
   ```

5. **Enchaîner les phases sur 1 seul Pod** :
   Les phases 1a, 1c, 2 peuvent tourner séquentiellement sur le même Pod Spot
   sans interruption. Un script maître lance tout :
   ```bash
   python -m scripts.pretrain --phase all --epochs 5
   python -m ml.training.train --model deepfm --epochs 20
   python -m ml.training.train --model mtl --epochs 10
   python -m ml.training.train --model sim --epochs 10
   python -m scripts.finetune_run --model all --phase 3 --epochs 3
   python -m scripts.export_triton
   ```

6. **Ne PAS entraîner sur Recipe1M+ (13M images) sauf nécessité** :
   Ce dataset prend ~100-200 GB et n'est utile que pour le cross-sell alimentaire.
   Reporter au Palier 2 quand tu auras un vrai catalogue food.

7. **PlantNet est optionnel** :
   Les 8M+ images ne sont utiles que si tu as une catégorie "plantes médicinales".
   Sous-échantillonner à 500K images max pour la v1.
