# ShopFeed OS - Architecture Audit (April 2026)

> [!IMPORTANT]
> This document overrides previous architecture drafts following the integration of the EPSILON Ads Engine, Llama 4 Scout (17B MoE), Monolith streaming trainer, and advanced multimodal encoders.

## 1. High-Level System Architecture

The architecture has shifted from a simplified inference pipeline to a heavily optimized, multimodal, online-learning-capable system, mirroring architectures seen at ByteDance and Meta.

### 1.1 Core Components
*   **API Gateway & BFF:** Handles incoming requests from React Native / Web clients.
*   **Feed Engine:** Orchestrates candidate retrieval, pre-ranking, and final ranking.
*   **Media Ingestion Pipeline:** Asynchronous processing of uploaded content (Images, Videos, Audio) using Ray distributed workers.
*   **Feature Store:** Redis cluster maintaining user profiles, item embeddings, and real-time temporal/desire graphs.
*   **EPSILON Ads Engine:** Real-time auction and placement system.
*   **Monolith Streaming Trainer:** Continuous online learning system using Redpanda and Ray.

## 2. Machine Learning Architecture

### 2.1 Inference Orchestration (Triton)
We use **Triton Inference Server** to serve the fast ranker and retrieval models in FP16/TensorRT.
*   **Two-Tower FAISS:** Fast candidate retrieval (2,000 items).
*   **DeepFM:** Pre-ranking (400 items).
*   **DIN / DIEN / BST:** Sequential user interest modeling.
*   **MTL / PLE:** multi-task final ranking (click, cart, purchase, eCPM).
*   **LightGBM Fraud:** Lightning-fast tree inference on GPU via ONNX.

### 2.2 Unification of LLM Tasks: Llama 4 Scout (17B-16E MoE)
Instead of fragmented micro-models, all complex reasoning and multimodal tasks have been consolidated into **Llama 4 Scout (17B MoE)** served via `vLLM`.
*   **Vision Tasks:** Quality scoring, duplicate detection (pHash), category verification.
*   **Text Tasks:** Copywriting, ad generation, conversational search querying, content moderation insights.
*   **Memory Footprint:** ~35GB VRAM in FP16 (or roughly 18-20GB if quantized).

### 2.3 Specialized Media Encoders
For raw media representation before feeding into ranking or search:
*   **Video:** VideoMAE (Spatio-temporal embeddings, 16 frames).
*   **Audio:** Whisper large-v3 (Transcription, Entity extraction) + Wav2Vec2 / VGGish (vocal features, pitch, energy).
*   **Images:** CLIP (SigLIP specifically for fashion).

### 2.4 Monolith Streaming Trainer
An online training architecture that updates model weights continuously:
*   **Ingestion:** Real-time user interactions via Redpanda (Kafka).
*   **Distribution:** PyTorch distributed training orchestrated by Ray.
*   **Serving:** Delta updates applied to base Triton scores every 5-15 minutes.

## 3. The EPSILON Ads Engine

The EPSILON engine is a 6-stage pipeline built for maximal eCPM and user retention.
1.  **Audience Matching & Targeting:** Two-Tower behavioral retrieval + Desire Graph (Psychographics) + Visual similarities.
2.  **Uplift Modeling (T-Learner):** Predicts the true causal impact of an ad to avoid wasting impressions on users who would buy anyway.
3.  **Ad Ranking (MTL/PLE):** Estimates pCTR, pCVR, pROAS, and importantly, **pStoreVisit** (heavily weighted).
4.  **Ad Fatigue Re-scoring:** Penalizes ads seen repeatedly by the same user.
5.  **GSP Auction (Generalized Second-Price):** Determines the final winner and standardizes the CPC.
6.  **Placement Validation:** Ensures ad spacing (e.g., minimum 5 organic posts between ads).

## 4. Operational Bottlenecks & Future Scalability
*   **GPU VRAM Contention:** Sharing Triton models, Llama 4 Scout, and multimedia encoders on a single device requires strict memory pinning and batch size management.
*   **Event Latency:** Monolith streaming relies on sub-second Redis and Redpanda event streaming; network jitter can cause stale features.
*   **Audio/Video Spikes:** Whisper large-v3 and VideoMAE are highly computationally expensive. Spikes in user uploads require careful queue management (Ray workers) to not starve real-time feed Triton queries.
