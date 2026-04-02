# Guide RunPod & Coûts Entraînement Pré-Production — ShopFeed OS (Avril 2026)

> [!NOTE]
> Ce document couvre deux sujets essentiels avant la mise en production :
> 1. **Guide RunPod** : Comprendre l'interface, les options et quoi choisir
> 2. **Coûts d'entraînement initial** : Entraîner tous les modèles from scratch + fine-tuner Llama 4 Scout

---

## PARTIE 1 — Comprendre RunPod : Serverless vs Pods vs Clusters

### Vue d'ensemble des produits RunPod

Quand tu te connectes à RunPod, tu vois ce menu dans la sidebar :

```
The Hub
  ├─ Serverless repos     ← Déploiement sans serveur
  ├─ Pod templates        ← Sauvegarder ta config Docker
  └─ Public endpoints     ← API publique (Whisper, etc.)

Resources
  ├─ Serverless           ← Exécution à la demande
  ├─ Pods                 ← Serveurs GPU loués ✅ (ce qu'il te faut)
  ├─ Clusters             ← Multi-GPU haute performance
  ├─ Storage              ← Stockage persistant ✅
  ├─ Fine tuning          ← Interface fine-tuning guidée
  └─ My templates         ← Tes templates Docker sauvegardés
```

---

### ❌ Serverless — PAS pour ShopFeed

> **C'est quoi** : Tu envoies une requête → RunPod démarre un GPU → exécute → s'arrête automatiquement.
> Tu paies **à la requête**, par seconde d'exécution.

**Problèmes pour ShopFeed :**
- Le démarrage à froid prend **10 à 60 secondes** → ton feed doit répondre en **< 10ms** → totalement incompatible
- Triton Inference Server, vLLM, Ray, Redis, FAISS doivent tourner **en permanence** en mémoire
- Impossible de maintenir un index FAISS chargé entre deux requêtes
- Imprévisible pour des SLA de production

---

### ✅ Pods — CE QU'IL TE FAUT (tu es au bon endroit)

> **C'est quoi** : Tu loues un serveur Linux avec un ou plusieurs GPU attachés.
> Il tourne **24h/7j** tant que tu le décides. Tu as un accès SSH complet.
> C'est exactement comme louer un serveur dédié mais avec GPU.

**Pour ShopFeed :**
- **Nœud #1** : Pod A100 SXM → tu y installes Triton Inference Server + FAISS
- **Nœud #2** : Pod H200 SXM → tu y installes vLLM (Scout FP8) + Ray + faster-whisper + VideoMAE

**Types de plans disponibles dans un Pod :**

| Plan | Prix | Interruptible ? | Pour qui ? |
| :--- | :--- | :--- | :--- |
| **On-Demand** | $1.49/hr (A100) | Non — garanti 24/7 | Production |
| **3 months savings** | $1.30/hr (A100) | Non — garanti | Production moyen terme |
| **6 months savings** | **$1.27/hr (A100)** | Non — garanti | ✅ **Notre choix lancement** |
| **1 year savings** | $1.22/hr (A100) | Non — garanti | Long terme |
| **Spot** | $0.95/hr (A100) | ⚠️ Oui — peut être interrompu | Training / batch |

---

### ❌ Clusters — PAS pour ShopFeed maintenant

> **C'est quoi** : Plusieurs GPU dans le même datacenter, connectés en NVLink/InfiniBand.
> Conçu pour entraîner des LLMs from scratch sur des milliers de GPUs.

**Pourquoi pas pour toi :**
- Ton Monolith Training est léger (modèles de ranking < 100M params) → 1 seul GPU suffit
- Le fine-tuning de Scout se fait avec LoRA sur 1 à 4 GPUs → pas besoin d'un cluster
- Coût beaucoup plus élevé, complexité inutile

---

### Explication de l'écran "Configure Deployment" (ton image)

Quand tu crées un Pod, tu vois 3 champs importants :

```
┌──────────────────────────────────────────────────────┐
│  Pod name : rich_blue_mongoose                        │
│  → C'est juste le NOM de ton serveur.                 │
│    RunPod en génère un aléatoire (tu peux le changer) │
│    Exemples : "shopfeed-inference-node" ,             │
│               "shopfeed-ai-node"                      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Pod template : Runpod Pytorch 2.4.0                  │
│  runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-...    │
│                                                       │
│  → C'est l'IMAGE DOCKER pré-installée sur ton serveur │
│    Ce template = Ubuntu 22.04 + CUDA 12.4.1 +        │
│                  Python 3.11 + PyTorch 2.4.0          │
│    Tout ça est déjà installé quand le Pod démarre.    │
│                                                       │
│  ✅ C'est LE BON template pour les 2 nœuds.          │
│    Tu n'auras qu'à installer vLLM, Triton, Ray,       │
│    faster-whisper etc. par-dessus via pip/conda.      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  GPU count : 1  2  3  4  5  6  7                     │
│                                                       │
│  → Combien de GPUs tu veux sur ce Pod.               │
│                                                       │
│  ✅ Nœud #1 (A100) : choisir 1                       │
│  ✅ Nœud #2 (H200) : choisir 1                       │
│    (Scout FP8 tient sur 1 seul H200 141GB)           │
└──────────────────────────────────────────────────────┘
```

> [!TIP]
> **Quel template choisir pour chaque nœud ?**
>
> - **Nœud #1 (A100 — Triton)** : Garde `Runpod Pytorch 2.4.0`. Tu installeras Triton Inference Server par-dessus : `pip install tritonclient[all]` et tu lanceras le serveur Triton via Docker in Docker.
> - **Nœud #2 (H200 — Scout)** : Garde `Runpod Pytorch 2.4.0`. Tu installeras vLLM par-dessus : `pip install vllm`. C'est la configuration recommandée par l'équipe vLLM.
>
> Ne choisis PAS le template "Stable Diffusion" ou "Text Generation WebUI" — ceux-là sont pour des usages différents.

---

### Network Volume — Le stockage partagé entre tes 2 Pods

Dans l'image, tu vois le filtre **"Network volume"** activé. Voici pourquoi c'est crucial :

```
┌─────────────────────────────────────────────────────────────┐
│  Network Volume (Volume Réseau)                              │
│                                                             │
│  ✅ Stockage persistant monté dans /workspace               │
│  ✅ N'est PAS supprimé quand le Pod s'arrête                │
│  ✅ Peut être monté sur plusieurs Pods simultanément        │
│  ✅ $0.07/GB/mois (sous 1 TB)                               │
│                                                             │
│  Pour ShopFeed, stocker sur le Network Volume :             │
│  ├─ Llama 4 Scout FP8 (~110 GB) ← ne pas re-télécharger   │
│  ├─ Index FAISS (~5-10 GB)                                  │
│  ├─ Checkpoints Monolith (~2-5 GB)                         │
│  └─ Datasets d'entraînement (~10-50 GB)                    │
│                                                             │
│  Taille recommandée : 150-200 GB → ~$10.50-14/mois         │
└─────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Toujours créer le Network Volume AVANT les Pods**, dans le même datacenter (ex: EU-RO-1).
> Ensuite, lors de la création de chaque Pod, sélectionner ce volume.
> Les 2 Pods pourront accéder aux mêmes fichiers sans re-télécharger Scout à chaque redémarrage.

---

### Résumé : Quoi choisir sur RunPod

| Élément | Choix | Raison |
| :--- | :--- | :--- |
| **Type de ressource** | **Pods** | Serveurs 24/7 avec contrôle total |
| **Nœud #1** | A100 SXM 80GB — 6-month plan | $1.27/hr → Triton Inference |
| **Nœud #2** | H200 SXM 141GB — 6-month plan | $3.12/hr → Scout FP8 + Média |
| **Storage** | Network Volume ~150-200 GB | $10.50-14/mois → partagé entre Pods |
| **Template** | Runpod Pytorch 2.4.0 | ✅ Pour les 2 Pods |
| **GPU count** | 1 | Un seul GPU par Pod |
| **Plan pricing** | 6 months savings | On-demand interruptible pour le training |
| **Serverless** | ❌ Non | Démarrage trop lent, pas adapté |
| **Clusters** | ❌ Non | Inutile à ce stade |

---

## PARTIE 2 — Entraînement Pré-Production sur RunPod

> [!NOTE]
> Avant de lancer l'application, tu dois :
> 1. **Entraîner tous les modèles de ranking** from scratch sur tes données
> 2. **Fine-tuner Llama 4 Scout** sur les données spécifiques à ShopFeed (produits africains, descriptions vendeurs, etc.)
>
> **Stratégie coût** : Utiliser des **instances Spot** (interruptibles) pour tout l'entraînement.
> Le prix Spot A100 SXM = **$0.95/hr** (vs $1.49/hr on-demand → -36%).
> Pour le fine-tuning Scout qui prend plus longtemps, utiliser des Spots avec checkpointing.

---

### Phase 1 — Entraînement des Modèles de Ranking (From Scratch)

Ces modèles sont **légers** et s'entraînent rapidement sur un seul A100.

#### Dataset hypothétique de référence
> Pour les estimations, on suppose ~**1-5M d'interactions** (lignes de logs comportementaux)
> à l'ouverture bêta, ce qui est réaliste pour un lancement Afrique francophone.

| Modèle | Description | GPU requis | Temps estimé | |
| :--- | :--- | :--- | :--- | :--- |
| **Two-Tower** | Retrieval utilisateur-item (256D) | 1× A100 Spot | ~2-4h | |
| **DeepFM** | Pre-ranking FM + DNN | 1× A100 Spot | ~1-2h | |
| **MTL / PLE** | Ranking final multi-tâches (7 objectifs) | 1× A100 Spot | ~3-6h | |
| **DIN** | Attention séquentielle historique | 1× A100 Spot | ~2-4h | |
| **DIEN** | Évolution d'intérêt GRU | 1× A100 Spot | ~3-5h | |
| **BST** | Transformer séquentiel | 1× A100 Spot | ~2-4h | |
| **SIM** | Historique long (SDIM) | 1× A100 Spot | ~3-6h | |
| **Ad Ranker EPSILON** | MTL/PLE publicitaire (pCTR/pCVR/pROAS) | 1× A100 Spot | ~2-4h | |
| **Uplift T-Learner** | Modèle causal T-Learner LightGBM | 1× A100 Spot | ~0.5-1h | |
| **LightGBM Fraude** | Détection fraude (35 features) | 1× A100 Spot | ~0.5-1h | |
| **GeoClassifier** | MLP classification géographique | 1× A100 Spot | ~0.5h | |
| **FAISS Index Build** | Construction index ANN (Two-Tower) | 1× A100 Spot | ~0.5-1h | |

**Coût Phase 1 :**

```
Temps total estimé (séquentiel) : ~21-38h
GPU utilisé : A100 SXM Spot → $0.95/hr

Optimisation : entraîner en PARALLÈLE (2 modèles simultanés
               sur le même Pod si VRAM le permet, ou 2 Pods Spot)

Scénario conservateur  : 38h × $0.95 = $36.10
Scénario optimiste     : 21h × $0.95 = $19.95
Estimation réaliste    : ~25-30h × $0.95 = ~$24-$29

+ Stockage disque Pod  : $0.006/hr × 30h = ~$0.18 (négligeable)
──────────────────────────────────────────────────────────────
TOTAL Phase 1          : ~$25 - $40
```

> [!TIP]
> **Optimisation clé** : Les modèles DIN, DIEN, BST peuvent s'entraîner sur le **même Pod A100**
> l'un après l'autre sans interruption. Crée un script Python qui les enchaîne automatiquement
> et qui s'auto-arrête à la fin → pas de GPU qui tourne à vide.
>
> ```python
> # Script d'entraînement séquentiel (exemple)
> import subprocess, sys
> models = ["two_tower", "deepfm", "mtl", "din", "dien", "bst", "sim"]
> for m in models:
>     subprocess.run([sys.executable, f"train_{m}.py"])
> print("✅ Tous les modèles entraînés. Arrêt du Pod.")
> # Le Pod s'auto-stop via l'API RunPod après ce script
> ```

---

### Phase 2 — Fine-Tuning de Llama 4 Scout (LoRA / QLoRA)

C'est la phase la plus importante et la plus coûteuse. L'objectif est d'adapter Scout
aux spécificités de ShopFeed : produits africains, descriptions vendeurs, langue française +
langues locales, scoring de qualité photo adapté au contexte.

#### Pourquoi LoRA et pas un fine-tuning complet ?

> Fine-tuning complet de 109B params nécessiterait ~420 GB de VRAM (optimizer states + gradients).
> Impossible même avec 8× H100.
>
> **LoRA (Low-Rank Adaptation)** : On n'entraîne que ~0.1-1% des paramètres (matrices d'adaptation).
> Nécessite ~20-40 GB de VRAM pour Scout en 4-bit (QLoRA). **Faisable sur 1× A100 ou 1× H100.**

#### Données à préparer pour le fine-tuning

| Dataset | Contenu | Taille estimée |
| :--- | :--- | :--- |
| **Quality Scoring** | Paires (image, score_qualité, justification) | ~10K-50K exemples |
| **Product Enrichment** | (titre_brut → titre_optimisé, description_complète) | ~20K-100K produits |
| **Ad Copywriting** | (infos_produit → texte_pub_performant) | ~5K-20K exemples |
| **Content Moderation** | (image+texte → verdict+justification) | ~5K-20K exemples |
| **Conversational Search** | (requête_utilisateur → query_structurée) | ~10K-50K exemples |
| **Seller Insights** | (données_vente → conseil_vendeur) | ~5K-10K exemples |

#### Configuration technique recommandée

```python
# Configuration QLoRA optimale pour Scout 17B MoE sur A100 80GB
# Via Unsloth (2x plus rapide que HuggingFace seul)

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    max_seq_length = 2048,
    dtype = None,       # Auto-détecte BF16
    load_in_4bit = True # QLoRA → ~60GB VRAM
)

# LoRA config : r=16 adapte le ratio qualité/temps
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)
```

#### Estimation du temps de fine-tuning

| Scénario | GPU | Dataset | Epochs | Temps | Coût |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Léger** (quality + enrichissement) | 1× A100 Spot | ~10K exemples | 3 | ~4-6h | ~$4-6 |
| **Complet** (toutes les tâches) | 1× A100 Spot | ~100K exemples | 3 | ~20-40h | ~$19-38 |
| **Complet rapide** | 2× A100 Spot | ~100K exemples | 3 | ~10-20h | ~$19-38 |
| **Complet haute qualité** | 1× H100 Spot | ~100K exemples | 5 | ~15-25h | ~$34-57 |

> Spot A100 SXM = **$0.95/hr** / Spot H100 SXM = **$2.29/hr**

**Notre recommandation : 1× A100 SXM Spot, ~30h, ~$29**

```
Raisons :
- Le fine-tuning LoRA n'a pas besoin de plus de 1 GPU (pas de parallelisme tensor nécessaire)
- $0.95/hr × 30h = $28.50 → budget très raisonnable
- Avec Unsloth, c'est 2× plus rapide → 15h → $14.25
- Checkpointing toutes les 500 steps → si le Spot est interrompu, on reprend
```

---

### Phase 3 — Export ONNX & Déploiement Triton (une seule fois)

Après entraînement, il faut exporter les modèles PyTorch en ONNX et les déployer sur Triton.
Votre projet a déjà `ml/serving/export_onnx.py` qui automatise tout cela.

| Tâche | Temps | Coût |
| :--- | :--- | :--- |
| Export ONNX des 12 modèles | ~30-60 min | ~$0.80 (A100 Spot) |
| Compilation TensorRT (Triton) | ~1-2h (première fois) | ~$1.90 (A100 Spot) |
| Tests de validation inférence | ~1h | ~$0.95 (A100 Spot) |
| **TOTAL Phase 3** | ~3-4h | **~$2.85 - $3.80** |

---

### Récapitulatif Coût Total Pré-Production

| Phase | Description | GPU (Spot) | Durée estimée | Coût estimé |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Entraînement 12 modèles de ranking | A100 SXM Spot ($0.95/hr) | ~25-30h | **~$24 - $29** |
| **Phase 2** | Fine-tuning Scout QLoRA (Unsloth) | A100 SXM Spot ($0.95/hr) | ~15-30h | **~$14 - $29** |
| **Phase 3** | Export ONNX + Compilation TensorRT | A100 SXM Spot ($0.95/hr) | ~3-4h | **~$3 - $4** |
| **Stockage** | Network Volume ~200 GB pendant la phase | $0.07/GB/mois | ~1 mois | **~$14** |
| **Marge erreurs/itérations** | Ré-entraînements, bugs, debug | — | — | **~$30** |
| | | | | |
| **TOTAL PRÉ-PRODUCTION** | | | | **~$85 - $106** |

> [!TIP]
> **Budget total pré-production recommandé : ~$100-150**
>
> Avec ce budget :
> - ✅ Tous les modèles entraînés et validés
> - ✅ Scout fine-tuné sur tes données (qualité optimale pour le marché africain)
> - ✅ Triton configuré et testé
> - ✅ Marge pour les itérations et debugging
>
> C'est **négligeable** comparé aux $19 280 du budget 6 mois de production.

---

### Checklist Pré-Production (Ordre Recommandé)

```
SEMAINE 1 — Préparation
□ Créer le Network Volume (200 GB) sur RunPod (même datacenter que les futurs Pods)
□ Préparer et nettoyer les datasets d'entraînement
□ Préparer les datasets de fine-tuning Scout (quality, enrichissement, copywriting)

SEMAINE 2-3 — Entraînement Ranking
□ Lancer Pod A100 Spot → exécuter le script d'entraînement séquentiel Phase 1
□ Valider chaque modèle (AUC, NDCG, métriques métier)
□ Sauvegarder les checkpoints sur le Network Volume
□ Construire l'index FAISS

SEMAINE 3-4 — Fine-tuning Scout
□ Lancer Pod A100 Spot → fine-tuner Scout avec QLoRA / Unsloth
□ Évaluer sur un dataset de test hold-out
□ Merger les adaptateurs LoRA dans le modèle de base
□ Sauvegarder le modèle FP8 fusionné sur le Network Volume

SEMAINE 4 — Préparation Production
□ Exporter tous les modèles ranking en ONNX
□ Compiler TensorRT sur le Nœud #1 (A100)
□ Déployer Scout FP8 sur Nœud #2 (H200) et tester vLLM
□ Test de charge (load testing) sur les 2 Pods
□ Monitorer les latences Feed + LLM

LANCEMENT 🚀
```

---

### Outils Recommandés

| Outil | Pour quoi | Lien |
| :--- | :--- | :--- |
| **Unsloth** | Fine-tuning LoRA 2× plus rapide, moins de VRAM | `pip install unsloth` |
| **wandb / MLflow** | Suivi des métriques d'entraînement en temps réel | Déjà dans ton projet (`ml/tracking/`) |
| **vLLM** | Serveur d'inférence Scout FP8 | `pip install vllm` |
| **faster-whisper** | Transcription audio INT8 (4× Whisper standard) | `pip install faster-whisper` |
| **RunPod SDK** | Auto-stop le Pod quand l'entraînement finit | `pip install runpod` |

```python
# Auto-stop Pod à la fin du script (évite de payer pour un GPU idle)
import runpod
runpod.api_key = "TON_API_KEY"

# À la fin du script d'entraînement :
runpod.stop_pod(runpod.get_current_pod_id())
print("✅ Entraînement terminé. Pod arrêté automatiquement.")
```
