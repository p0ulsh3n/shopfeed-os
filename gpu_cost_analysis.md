# Analyse des Coûts GPU & Infrastructure — ShopFeed OS (Avril 2026)

> [!NOTE]
> **Architecture de référence : 2 nœuds loués pour 6 mois avec le plan d'économies RunPod.**
> Prix vérifiés directement sur [runpod.io/pricing](https://www.runpod.io/pricing) le 02/04/2026
> depuis le compte RunPod connecté.
> Llama 4 Scout est **entièrement self-hosté** sur le Nœud #2. **Aucune API LLM externe.**

---

## 1. Configuration Finale — Phase de Lancement (6 mois)

### Nœud #1 — "Inference Node" → 1× A100 SXM 80GB

| | Valeur |
| :--- | :--- |
| **GPU** | NVIDIA A100 SXM |
| **VRAM** | 80 GB |
| **RAM** | 117 GB |
| **vCPUs** | 16 |
| **Plan** | 6 months savings plan |
| **Prix/heure** | **$1.27/hr** (vs $1.49/hr on-demand) |
| **Coût total 6 mois (upfront)** | **$5 577.84** |
| **Économie vs on-demand** | $966.24 économisés |
| **Disque (running)** | $0.006/hr |
| **Disque (stopped)** | $0.006/hr |

### Nœud #2 — "AI Node" → 1× H200 SXM 141GB

| | Valeur |
| :--- | :--- |
| **GPU** | NVIDIA H200 SXM ⭐ Featured |
| **VRAM** | **141 GB** |
| **RAM** | 188 GB |
| **vCPUs** | 12 |
| **Plan** | 6 months savings plan |
| **Prix/heure** | **$3.12/hr** (vs $3.59/hr on-demand) |
| **Coût total 6 mois (upfront)** | **$13 703.04** |
| **Économie vs on-demand** | $2 064.24 économisés |
| **Disque (running)** | $0.006/hr |
| **Disque (stopped)** | $0.006/hr |

### Récapitulatif Financier — Phase Lancement (6 mois)

| | Nœud #1 (A100 SXM) | Nœud #2 (H200 SXM) | **TOTAL** |
| :--- | :--- | :--- | :--- |
| **Coût upfront 6 mois** | $5 577.84 | $13 703.04 | **$19 280.88** |
| **Équivalent mensuel** | ~$929.64/mois | ~$2 283.84/mois | **~$3 213.48/mois** |
| **Prix horaire effectif** | $1.27/hr | $3.12/hr | $4.39/hr total |
| **Économie totale vs on-demand** | $966.24 | $2 064.24 | **$3 030.48 économisés** |

> [!TIP]
> Le plan 6 mois te fait économiser **$3 030.48** par rapport au tarif on-demand sur la même période.
> Soit l'équivalent d'environ **2 mois de GPU #1 offerts** !

---

## 2. Avantage Clé : H200 141GB Change Tout pour Scout

> [!IMPORTANT]
> Avec 141 GB de VRAM sur le Nœud #2, Llama 4 Scout n'a **plus besoin d'être quantifié en INT4**.
>
> | Précision | VRAM Scout (109B params) | Compatible H200 141GB ? | Qualité |
> | :--- | :--- | :--- | :--- |
> | FP16 / BF16 | ~210 GB | ❌ Dépasse les 141 GB | Maximale |
> | **FP8** (natif H200/H100) | **~110 – 120 GB** | **✅ Rentre dans 141 GB** | **Excellente** |
> | INT4 / AWQ | ~55 – 60 GB | ✅ Rentre mais inutile | Dégradée |
>
> **Scout FP8 sur H200 = qualité quasi-FP16 avec la moitié de la VRAM.**
> Le H200 supporte le FP8 nativement (comme le H100). La perte de qualité vs FP16 est < 1%.
> Plus de contexte limité non plus : on peut aller jusqu'à ~16K-32K tokens avec le KV cache restant.

---

## 3. Correction Critique : Llama 4 Scout 17B-16E MoE

> [!CAUTION]
> Llama 4 Scout est un modèle **MoE (Mixture-of-Experts)** avec **109 milliards de paramètres
> TOTAUX** (seulement 17B actifs par token, mais les 109B doivent tous être chargés en VRAM).
>
> La version précédente du document était erronée (35-38 GB indiqué). La réalité :
>
> | Précision | VRAM poids seuls | VRAM réaliste (+ KV cache) |
> | :--- | :--- | :--- |
> | FP16 / BF16 | ~210 GB | ~220-250 GB ❌ |
> | **FP8** (natif H200) | **~110 GB** | **~120-135 GB ✅ sur H200** |
> | INT4 / AWQ / GPTQ | ~55 GB | ~60-70 GB ✅ sur A100 (mais qualité réduite) |

---

## 4. Inventaire Complet VRAM par Modèle

### Nœud #1 — A100 80GB — "Inference Node" (24h/7j)

Rôle : servir le feed et le moteur publicitaire EPSILON avec latence < 10ms.

| Modèle | Rôle | VRAM (FP16 + TensorRT) |
| :--- | :--- | :--- |
| Two-Tower | Retrieval FAISS (2 000 candidats) | ~10 MB |
| DeepFM | Pré-ranking (2 000 → 400) | ~40 MB |
| MTL / PLE | Ranking final 7 tâches (click, cart, purchase…) | ~30 MB |
| DIN | Attention séquentielle historique court | ~50 MB |
| DIEN | Évolution d'intérêt (GRU + attention) | ~60 MB |
| BST | Transformer séquentiel (multi-head) | ~50 MB |
| SIM | Historique long (SDIM module) | ~70 MB |
| Ad Ranker EPSILON | pCTR × pCVR × pROAS × pStoreVisit | ~40 MB |
| Uplift T-Learner | Impact causal incrémental des pubs | ~10 MB |
| LightGBM Fraude | Détection bots / spam (35 features) | ~10 MB |
| GeoClassifier | Classification géographique L1-L4 (MLP) | ~6 MB |
| Delta Model (Monolith) | Corrections temps réel (scores delta) | ~10 MB |
| **Buffers TensorRT + CUDA Graphs + Response Cache** | | ~3-4 GB |
| **TOTAL Nœud #1** | | **~4 – 6 GB / 80 GB ✅** |
| **Marge disponible** | | **~74 GB 🟢** |

> [!NOTE]
> La marge de ~74 GB garantit zéro contention même lors de pics de trafic extrêmes.
> Le feed, les pubs EPSILON et la fraude tournent toujours à < 10ms.

---

### Nœud #2 — H200 141GB — "AI Node" (LLM + Média + Training)

#### Charge Permanente (24h/7j)

| Composant | Rôle | VRAM |
| :--- | :--- | :--- |
| **Llama 4 Scout FP8** (vLLM) | Quality scoring, modération, copywriting pub, enrichissement produit, recherche conversationnelle, insights vendeur | **~110 – 120 GB** |
| **KV Cache + overhead vLLM** | Buffers d'inférence (contexte 16K-32K tokens possible) | ~8 – 12 GB |
| **TOTAL Permanent Nœud #2** | | **~120 – 132 GB / 141 GB** |
| **Marge disponible** | | **~9 – 21 GB 🟡** |

#### Charge "À la Demande" (uploads vendeurs — dans la marge)

Ces modèles s'activent uniquement lors d'un **upload vendeur**. Gérés via **Ray Workers**.

| Modèle | Rôle | VRAM | Déclencheur |
| :--- | :--- | :--- | :--- |
| **faster-whisper INT8** | Transcription audio + extraction entités (4× plus rapide que Whisper standard) | ~3 – 4 GB | Upload vidéo |
| **VideoMAE ViT-Base** | Embeddings spatio-temporels (16 frames) | ~2 – 3 GB | Upload vidéo |
| **CLIP ViT-B/32** | Embeddings image 512D (produits généraux) | ~0.5 – 1 GB | Upload image |
| **SigLIP (Fashion)** | Embeddings image mode spécialisé | ~0.5 – 1 GB | Upload image mode |
| **VGGish** | Embeddings audio 128D | ~0.3 – 0.5 GB | Upload vidéo |
| **Wav2Vec2 (base)** | Features vocales (pitch, débit, énergie) | ~0.4 – 0.8 GB | Upload vidéo |
| **ViT-Base (Modération)** | Classification NSFW / violence | ~0.5 – 1 GB | Upload image |
| **TOTAL Pipeline Média (pic complet)** | | **~7 – 11 GB** | |

> [!WARNING]
> **Limite d'uploads simultanés sur Nœud #2** :
> Scout FP8 (~125 GB) + Pipeline Média complet (~10 GB) = **~135 GB / 141 GB**.
> Confortable mais surveiller. Limiter à **max 3 uploads vidéo simultanés** avec
> `ray.remote(max_concurrency=3)`. Dépasser = OOM et crash vLLM.

#### Charge Hors-Heures (Monolith Training — 22h → 6h)

| Composant | Rôle | VRAM |
| :--- | :--- | :--- |
| Ray Workers PyTorch | Entraînement distribué sur flux Redpanda/Kafka | ~4 – 6 GB |
| Gradients + Optimizer Adam | Buffers de backpropagation (modèles légers, pas Scout) | ~2 – 4 GB |
| **TOTAL Training Nœud #2** | | **~6 – 10 GB** (dans la marge) |

---

## 5. Vue Consolidée VRAM des 2 Nœuds

```
╔══════════════════════════════════════════════════════════════════╗
║  NŒUD #1 — A100 SXM 80 GB — Inference Node                      ║
╠══════════════════════════════════════════════════════════════════╣
║  12 modèles Triton + buffers CUDA                      ~5 GB     ║
║  ▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    5/80 GB        ║
║  Marge libre : ~75 GB 🟢                                         ║
║  Latence feed : < 10 ms                                          ║
║  Plan 6 mois : $1.27/hr → $5 577.84 upfront                      ║
╠══════════════════════════════════════════════════════════════════╣
║  NŒUD #2 — H200 SXM 141 GB — AI Node                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Scout FP8 (permanent)                              ~120-132 GB  ║
║  ██████████████████████████████████████████░░  132/141 GB        ║
║  Marge libre : ~9-21 GB 🟡                                        ║
║                                                                  ║
║  + Média à la demande (max 3 vidéos simultanées)    +7-11 GB     ║
║  → Total pic (Scout + Média)                 ~135-139/141 GB ⚠️   ║
║  ████████████████████████████████████████████████ 139/141 GB    ║
║                                                                  ║
║  + Training Monolith (22h-6h, dans la marge)        +6-10 GB    ║
║  Plan 6 mois : $3.12/hr → $13 703.04 upfront                    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 6. Paliers Évolutifs & Coûts

### 🟢 Palier 1 : Phase de Lancement — 6 Mois (0 – 5 000 DAU)

| Composant | GPU | Plan | Coût upfront | Équivalent/mois |
| :--- | :--- | :--- | :--- | :--- |
| Nœud #1 (Triton + EPSILON) | A100 SXM 80GB | 6 months | **$5 577.84** | ~$929/mois |
| Nœud #2 (Scout FP8 + Média + Training) | H200 SXM 141GB | 6 months | **$13 703.04** | ~$2 284/mois |
| Stockage réseau partagé (100 GB) | Network Volume | — | ~$42/6mois | ~$7/mois |
| **TOTAL** | | | **~$19 323** | **~$3 220/mois** |

**Économies réalisées vs on-demand même période :**
> ($3.59 + $1.49) × 730h × 6 mois = **$22 250** on-demand
> vs **$19 323** plan 6 mois → **Économie : $2 927**

**Ce que ce budget inclut :**
- ✅ Feed temps réel < 10ms pour tous les utilisateurs
- ✅ Moteur publicitaire EPSILON (6 étapes, 12 modèles)
- ✅ Llama 4 Scout 17B **FP8** self-hosté (pas INT4 ! qualité quasi-FP16)
- ✅ Contexte Scout jusqu'à 16K-32K tokens (vs 4K en INT4 sur A100)
- ✅ 7 tâches LLM : quality scoring, copywriting pub, modération, enrichissement, recherche conv., insights vendeur, catégorisation
- ✅ Pipeline vidéo complet (faster-whisper INT8, VideoMAE, CLIP, SigLIP, VGGish, Wav2Vec2)
- ✅ Entraînement Monolith continu (mises à jour toutes les 5-15 min, nuit)
- ✅ Détection fraude (LightGBM 35 features)
- ✅ Zéro API LLM externe — confidentialité totale des données vendeurs

---

### 🟡 Palier 2 : Croissance (5 000 – 50 000 DAU) — Après 6 mois

À la fin des 6 mois, le volume justifie d'ajouter un 3ème nœud pour séparer le training.

| Composant | GPU | Plan | Coût mensuel |
| :--- | :--- | :--- | :--- |
| Nœud #1 (Inference) | A100 SXM 80GB | 6 months renouvelé | ~$929/mois |
| Nœud #2 (AI — Scout + Média) | H200 SXM 141GB | 6 months renouvelé | ~$2 284/mois |
| Nœud #3 (Training Monolith dédié) | A100 SXM 80GB | Spot (~$0.95/hr) | ~$694/mois |
| **TOTAL** | | | **~$3 907/mois** |

---

### 🔴 Palier 3 : Échelle (50 000+ DAU)

| Composant | GPU | Coût mensuel estimé |
| :--- | :--- | :--- |
| Cluster Inférence (2× A100 SXM) | A100 SXM | ~$1 858/mois |
| Cluster LLM (1× H200 ou 2× H100 SXM) | H200 / H100 | ~$2 284 – 3 928/mois |
| Cluster Training (2× A100 Spot) | A100 SXM Spot | ~$1 388/mois |
| Cluster Média (1× A100 dédié) | A100 SXM | ~$929/mois |
| **TOTAL** | | **~$6 500 – 8 100/mois** |

---

## 7. Tableau Récapitulatif des 3 Paliers

| | 🟢 Palier 1 — 6 mois | 🟡 Palier 2 — Croissance | 🔴 Palier 3 — Échelle |
| :--- | :--- | :--- | :--- |
| **DAU** | 0 – 5 000 | 5 000 – 50 000 | 50 000+ |
| **GPUs** | A100 SXM + H200 SXM | A100 + H200 + A100 Spot | Cluster 4-8 GPUs |
| **Coût upfront** | **$19 323 (6 mois)** | — | — |
| **Coût /mois** | **~$3 220** | **~$3 907** | **~$7 300** |
| **Latence Feed** | < 10 ms | < 10 ms | < 5 ms |
| **Latence Scout** | ~60-80 ms (FP8 local) | ~60 ms | ~30 ms |
| **Précision Scout** | **FP8** (quasi-FP16) | **FP8** | **FP8** |
| **API LLM externe** | ❌ Aucune | ❌ Aucune | ❌ Aucune |
| **Max uploads vidéo simultanés** | 3 | 5 | Illimité |

---

## 8. Recommandations Techniques

1. **`--dtype fp8` dans vLLM pour Scout** :
   ```bash
   vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
     --dtype fp8 \
     --max-model-len 16384 \
     --gpu-memory-utilization 0.90
   ```
   Le flag `fp8` est supporté nativement sur H200. Utiliser `0.90` pour garder ~14 GB de marge.

2. **`faster-whisper` INT8 obligatoire** :
   Utiliser `faster-whisper` (CTranslate2) au lieu du Whisper standard réduit la VRAM
   de ~10 GB à ~3-4 GB et transcrit 4× plus vite.

3. **`ray.remote(max_concurrency=3)` sur les workers média** :
   Limite stricte à 3 uploads vidéo simultanés (Scout FP8 ~125 GB + 3× média ~15 GB = ~140 GB).

4. **Monolith Training uniquement hors-heures (22h-6h)** :
   Les heures creuses libèrent du KV cache (moins de requêtes LLM actives) → ~15-20 GB disponibles
   pour Ray workers, ce qui est largement suffisant.

5. **Volume réseau partagé (Network Volume RunPod)** :
   Stocker les checkpoints Monolith + index FAISS + modèle Scout FP8 sur un volume partagé
   entre les deux nœuds (~100 GB recommandé → ~$7/mois).
   - Avant 1 TB : **$0.07/GB/mois**
   - Après 1 TB : **$0.05/GB/mois**

6. **Renouvellement après 6 mois** :
   Surveiller la disponibilité du **H200 SXM** en Secure Cloud pour le plan suivant.
   Si indisponible : 2× H100 NVL (94 GB chacun → 188 GB totaux en Tensor Parallelism)
   à $3.07/hr/GPU = $6.14/hr pour les deux (Scout en FP16 alors possible).
