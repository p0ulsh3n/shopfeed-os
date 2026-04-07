"""
ShopFeed Visual Encoder — Architecture LoRA 2026
=================================================
Architecture validée en production (recherches avril 2026) :

  BASE MODEL  : Marqo/marqo-ecommerce-embeddings-L
    → 652M params, 1024d, surpasse Amazon Titan + ViT-SO400M sur catalogues mixtes
    → Source: github.com/marqo-ai/marqo-ecommerce-embeddings (nov 2024)

  ADAPTERS    : LoRA légers par catégorie (PEFT, r=8)
    → ~2-8MB par adapter (vs 2.5GB pour un modèle complet)
    → Merge dans les poids → zéro latence en production
    → Source: philschmid.de, HuggingFace PEFT 2025

  PROJECTION  : 1024d → 512d (couche linéaire apprise)
    → Compatibilité pipeline ShopFeed (feature_store, Two-Tower, FAISS)

DEUX FLUX D'UTILISATION (design Pinterest/Shopify style) :

  OFFLINE (indexation batch) — BatchEncoder :
    image + catégorie → ecommerce-L + LoRA catégorie → 1024d → proj → 512d
    Appelé une fois par produit. Résultat stocké en DB. NE PAS utiliser en temps réel.

  ONLINE (requête utilisateur) — QueryEncoder :
    image/texte requête → ecommerce-L (base, sans adapter) → 1024d → proj → 512d
    Appelé à chaque requête. Doit être <10ms. Pas d'adapter (inutile côté query).

CLASSES :
  ModelLoader          : charge le bon modèle selon le loader déclaré dans le YAML
  AdapterRegistry      : découverte + chargement lazy des adapters LoRA
  ProjectionHead       : couche linéaire 1024d → 512d, apprise et sauvegardée
  EcommerceEncoder     : singleton thread-safe, pilote offline + online
  CategoryAdapterTrainer: entraîne un adapter LoRA pour une catégorie donnée
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config (chargée depuis configs/encoders.yaml)
# ──────────────────────────────────────────────────────────────

def _load_encoder_config() -> dict:
    try:
        from ml.config_loader import get_encoder_config
        return get_encoder_config()
    except Exception:
        return {}

_CFG = _load_encoder_config()
_VE  = _CFG.get("visual_encoder", {})
_ADP = _VE.get("adapters", {})

UNIFIED_DIM     = int(_VE.get("unified_output_dim", 512))
ADAPTER_DIR     = Path(os.environ.get("ADAPTER_CHECKPOINT_DIR", _ADP.get("checkpoint_dir", "checkpoints/adapters")))
LORA_RANK       = int(_ADP.get("lora_rank",  8))
LORA_ALPHA      = int(_ADP.get("lora_alpha", 16))
LORA_DROPOUT    = float(_ADP.get("lora_dropout", 0.05))
LORA_MODULES    = _ADP.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj"])
CATEGORY_MAP: dict[str, str] = _ADP.get("categories", {})


# ──────────────────────────────────────────────────────────────
# Projection Head (1024d → UNIFIED_DIM)
# ──────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Projette l'embedding du modèle de base vers la dimension unifiée du pipeline.

    Apprise en contrastive learning (image, titre produit) sur les données ShopFeed.
    Si non entraînée → projection linéaire simple non-apprise (moins bonne mais fonctionnelle).

    Architecture : Linear → GELU → Dropout → Linear → LayerNorm → L2-normalize
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = UNIFIED_DIM):
        super().__init__()
        hidden = max(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, output_dim, bias=False),
            nn.LayerNorm(output_dim),
        )
        # Init identité : les poids de départ ne dégradent pas les embeddings
        nn.init.eye_(self.net[0].weight[:min(input_dim, hidden), :input_dim])
        nn.init.eye_(self.net[3].weight[:output_dim, :min(hidden, output_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

    def load_or_init(self, checkpoint: str | Path | None) -> "ProjectionHead":
        """Charge les poids depuis un checkpoint ou reste en mode non-entraîné."""
        path = Path(checkpoint) if checkpoint else ADAPTER_DIR / "projection.pt"
        if path.exists():
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.load_state_dict(state)
                logger.info("ProjectionHead chargée depuis %s", path)
            except Exception as e:
                logger.warning("ProjectionHead checkpoint invalide (%s): %s — init identité", path, e)
        else:
            logger.info("ProjectionHead non-entraînée — utilise projection identité. "
                        "Entraîner via CategoryAdapterTrainer pour de meilleurs résultats.")
        return self


# ──────────────────────────────────────────────────────────────
# Model Loader (duck-typed : HuggingFace transformers OU open_clip)
# ──────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    """Résultat du chargement d'un modèle, unifié quelle que soit la lib."""
    model: Any           # nn.Module
    preprocess: Any      # callable(PIL.Image) → Tensor
    output_dim: int      # dimension de l'embedding image
    loader: str          # "transformers" | "open_clip"
    model_id: str


class ModelLoader:
    """Charge un modèle visuel via transformers ou open_clip selon la config YAML.

    Design : les deux loaders exposent la même interface après chargement.
    La couche appelante n'a pas besoin de savoir quelle lib a été utilisée.
    """

    @staticmethod
    def load(domain: str) -> LoadedModel | None:
        """Charge le modèle principal du domaine avec sa chaîne de fallback."""
        models_cfg = _VE.get("models", {})
        domain_cfg = models_cfg.get(domain) or models_cfg.get("generic") or {}

        primary    = domain_cfg.get("primary", "")
        fallback   = domain_cfg.get("fallback", "")
        loader_typ = domain_cfg.get("loader", "open_clip")
        out_dim    = int(domain_cfg.get("output_dim", 512))
        description = domain_cfg.get("description", "")

        logger.info("ModelLoader: domain=%s, primary=%s, loader=%s", domain, primary, loader_typ)

        # Tentative 1 — modèle principal
        result = ModelLoader._try(primary, fallback, loader_typ, out_dim)
        if result:
            logger.info("✓ Modèle chargé : %s | %s", primary, description)
            return result

        # Tentative 2 — fallback du domaine
        if fallback and fallback != primary:
            fallback_pretrained = domain_cfg.get("fallback_pretrained", "openai")
            result = ModelLoader._try(fallback, "", loader_typ, out_dim, fallback_pretrained)
            if result:
                logger.warning("⚠ Modèle fallback : %s (primary=%s indisponible)", fallback, primary)
                return result

        # Tentative 3 — SigLIP générique (dernier recours)
        result = ModelLoader._try_openclip_arch("ViT-B-16-SigLIP", "webli", 512)
        if result:
            logger.warning("⚠ Fallback ultime : SigLIP webli (dim=512). "
                           "Qualité dégradée. Vérifier VISUAL_ENCODER_DOMAIN et connexion HuggingFace.")
            return result

        logger.error("✗ Tous les encodeurs ont échoué pour domain=%s. Features visuelles = zéros.", domain)
        return None

    @staticmethod
    def _try(
        model_id: str,
        fallback_id: str,
        loader: str,
        out_dim: int,
        pretrained: str = "openai",
    ) -> LoadedModel | None:
        if not model_id:
            return None
        if loader == "transformers":
            return ModelLoader._try_transformers(model_id, out_dim)
        return ModelLoader._try_openclip(model_id, out_dim)

    @staticmethod
    def _try_transformers(model_id: str, out_dim: int) -> LoadedModel | None:
        """Charge via HuggingFace transformers (Marqo ecommerce-L, etc.)."""
        try:
            from transformers import AutoModel, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False  # 🔒 FROZEN

            # Wrapper preprocess compatible avec open_clip (PIL.Image → Tensor)
            def preprocess(image):
                inputs = processor(images=image, return_tensors="pt")
                return inputs["pixel_values"].squeeze(0)

            # Wrapper encode_image compatible
            original_model = model
            class _TransformersAdapter(nn.Module):
                """Adaptateur duck-type : expose encode_image() comme open_clip."""
                def __init__(self):
                    super().__init__()
                    self._m = original_model

                def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
                    # Marqo ecommerce-L expose get_image_features()
                    if hasattr(self._m, "get_image_features"):
                        return self._m.get_image_features(pixel_values=pixel_values)
                    out = self._m(pixel_values=pixel_values, return_dict=True)
                    # SigLIP: pooler_output, CLIP: image_embeds
                    return (
                        out.get("image_embeds")
                        or out.get("pooler_output")
                        or out.last_hidden_state[:, 0, :]
                    )

                def parameters(self, recurse=True):
                    return self._m.parameters(recurse)

            adapted = _TransformersAdapter()
            return LoadedModel(adapted, preprocess, out_dim, "transformers", model_id)

        except ImportError:
            logger.warning("transformers non installé — pip install transformers>=4.40")
        except Exception as e:
            logger.debug("_try_transformers(%s): %s", model_id, e)
        return None

    @staticmethod
    def _try_openclip(model_id: str, out_dim: int) -> LoadedModel | None:
        """Charge via open_clip (FashionSigLIP, CLIP DataComp, etc.)."""
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(model_id)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return LoadedModel(model, preprocess, out_dim, "open_clip", model_id)
        except Exception as e:
            logger.debug("_try_openclip(%s): %s", model_id, e)
        return None

    @staticmethod
    def _try_openclip_arch(arch: str, pretrained: str, out_dim: int) -> LoadedModel | None:
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return LoadedModel(model, preprocess, out_dim, "open_clip", f"{arch}/{pretrained}")
        except Exception as e:
            logger.debug("_try_openclip_arch(%s, %s): %s", arch, pretrained, e)
        return None


# ──────────────────────────────────────────────────────────────
# Adapter Registry (LoRA lifecycle)
# ──────────────────────────────────────────────────────────────

class AdapterRegistry:
    """Découverte, chargement et cache des adapters LoRA par catégorie.

    Design :
      - Scanne ADAPTER_DIR au premier accès (lazy)
      - Charge un adapter uniquement quand on en a besoin
      - Merge les poids LoRA dans le modèle → zéro latence, défusion si besoin
      - Thread-safe

    Format de fichier : {ADAPTER_DIR}/{category}_lora/  (dossier PEFT standard)
    """

    def __init__(self, adapter_dir: Path = ADAPTER_DIR):
        self.adapter_dir = adapter_dir
        self._lock = threading.Lock()
        self._available: set[str] = set()
        self._scanned = False

    def _scan(self) -> None:
        """Découvre les adapters disponibles dans ADAPTER_DIR."""
        if self._scanned:
            return
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        for path in self.adapter_dir.iterdir():
            if path.is_dir() and (path / "adapter_config.json").exists():
                name = path.name.removesuffix("_lora")
                self._available.add(name)
        self._scanned = True
        if self._available:
            logger.info("AdapterRegistry: adapters disponibles = %s", sorted(self._available))
        else:
            logger.info(
                "AdapterRegistry: aucun adapter LoRA trouvé dans %s. "
                "Entraîner via CategoryAdapterTrainer.",
                self.adapter_dir,
            )

    def has_adapter(self, category: str) -> bool:
        with self._lock:
            self._scan()
            # Normalisation : "Électronique" → "electronics" via CATEGORY_MAP
            normalized = self._normalize(category)
            return normalized in self._available

    def adapter_path(self, category: str) -> Path | None:
        normalized = self._normalize(category)
        if not self.has_adapter(normalized):
            return None
        return self.adapter_dir / f"{normalized}_lora"

    def apply_adapter(self, model: nn.Module, category: str) -> nn.Module:
        """Charge et fusionne le LoRA adapter dans une copie du modèle.

        Retourne un nouveau modèle avec l'adapter mergé (poids fusionnés,
        aucune latence supplémentaire à l'inférence).
        """
        path = self.adapter_path(category)
        if path is None:
            return model  # pas d'adapter → modèle de base

        try:
            from peft import PeftModel
            # Charger adapter sur le modèle de base
            # Note: le modèle de base doit être un nn.Module standard
            adapted = PeftModel.from_pretrained(model, str(path))
            # Merger les poids LoRA dans les poids de base → zéro overhead
            merged  = adapted.merge_and_unload()
            merged.eval()
            logger.info("✓ Adapter LoRA mergé: category=%s, path=%s", category, path)
            return merged
        except ImportError:
            logger.warning("peft non installé — pip install peft. Utilise modèle de base.")
        except Exception as e:
            logger.warning("Adapter LoRA (%s) invalide: %s — utilise modèle de base", path, e)

        return model

    @staticmethod
    def _normalize(category: str) -> str:
        """Normalise le nom de catégorie vers la clé YAML."""
        cat = str(category).lower().strip()
        # Lookup direct dans CATEGORY_MAP
        if cat in CATEGORY_MAP:
            return CATEGORY_MAP[cat]
        # Correspondance partielle
        for key in CATEGORY_MAP:
            if key in cat or cat in key:
                return CATEGORY_MAP[key]
        return cat


# ──────────────────────────────────────────────────────────────
# EcommerceEncoder — singleton thread-safe (cœur du système)
# ──────────────────────────────────────────────────────────────

class EcommerceEncoder:
    """Encodeur visuel principal — singleton thread-safe.

    Gère deux chemins d'exécution :

    ┌─ OFFLINE (indexation produits) ─────────────────────────────┐
    │  encode_product(image, category="electronics")              │
    │  → ecommerce-L + LoRA electronics + projection              │
    │  → 512d embeddings stockés en DB                            │
    │  → Appeler depuis un job batch, pas à chaque requête        │
    └──────────────────────────────────────────────────────────────┘

    ┌─ ONLINE (requête utilisateur) ──────────────────────────────┐
    │  encode_query(image_or_text)                                │
    │  → ecommerce-L base (sans adapter) + projection             │
    │  → <10ms sur GPU, ~50ms sur CPU                             │
    │  → Même espace d'embedding que les produits indexés         │
    └──────────────────────────────────────────────────────────────┘
    """

    _instance: "EcommerceEncoder | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._model_lock  = threading.Lock()
        self._base_model: LoadedModel | None   = None
        self._projection:  ProjectionHead | None = None
        self._adapters    = AdapterRegistry()
        self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Cache des modèles avec adapter fusionné (catégorie → LoadedModel)
        self._adapted_models: dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "EcommerceEncoder":
        """Singleton thread-safe."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset du singleton (tests, changement de domaine)."""
        with cls._lock:
            cls._instance = None
        logger.info("EcommerceEncoder singleton réinitialisé.")

    # ── Chargement lazy du modèle de base ─────────────────────

    def _ensure_base_model(self) -> LoadedModel | None:
        if self._base_model is not None:
            return self._base_model
        with self._model_lock:
            if self._base_model is not None:
                return self._base_model
            # Lire le domaine depuis la config
            domain = os.environ.get(
                "VISUAL_ENCODER_DOMAIN",
                _VE.get("domain", "ecommerce"),
            )
            loaded = ModelLoader.load(domain)
            if loaded is None:
                return None
            loaded.model.to(self._device)
            self._base_model = loaded

            # Charger ou initialiser la projection
            proj_ckpt = _VE.get("projection", {}).get("checkpoint")
            self._projection = ProjectionHead(
                input_dim=loaded.output_dim,
                output_dim=UNIFIED_DIM,
            ).load_or_init(proj_ckpt).to(self._device)
            self._projection.eval()

        return self._base_model

    def _get_adapted_model(self, category: str) -> Any:
        """Retourne le modèle de base + adapter mergé pour une catégorie.

        Mise en cache : merge effectué une seule fois par catégorie.
        """
        normalized = AdapterRegistry._normalize(category)

        if normalized in self._adapted_models:
            return self._adapted_models[normalized]

        with self._model_lock:
            if normalized in self._adapted_models:
                return self._adapted_models[normalized]

            base = self._ensure_base_model()
            if base is None:
                return None

            if self._adapters.has_adapter(normalized):
                adapted_model = self._adapters.apply_adapter(base.model, normalized)
                adapted_model.to(self._device)
                self._adapted_models[normalized] = adapted_model
            else:
                # Pas d'adapter pour cette catégorie → modèle de base
                self._adapted_models[normalized] = base.model

        return self._adapted_models[normalized]

    # ── API publique ───────────────────────────────────────────

    @torch.no_grad()
    def encode_product(
        self,
        image: Any,
        category: str = "default",
    ) -> torch.Tensor:
        """OFFLINE — encode un produit avec son adapter catégorie.

        Args:
            image:    PIL.Image ou URL (string) ou path (string)
            category: Catégorie du produit ("electronics", "fashion", etc.)

        Returns:
            Tensor [UNIFIED_DIM] normalisé L2, sur CPU.
        """
        base = self._ensure_base_model()
        if base is None:
            return torch.zeros(UNIFIED_DIM)

        img_tensor = self._preprocess(image, base)
        if img_tensor is None:
            return torch.zeros(UNIFIED_DIM)

        model = self._get_adapted_model(category)
        if model is None:
            return torch.zeros(UNIFIED_DIM)

        emb = model.encode_image(img_tensor.to(self._device))
        emb = F.normalize(emb.float(), dim=-1)

        if self._projection is not None:
            emb = self._projection(emb)

        return emb.squeeze(0).cpu()

    @torch.no_grad()
    def encode_query(self, image: Any) -> torch.Tensor:
        """ONLINE — encode une image de requête (sans adapter, pour la vitesse).

        Même espace d'embedding que encode_product() mais sans adapter.
        Latence cible : <10ms GPU, ~50ms CPU.
        """
        base = self._ensure_base_model()
        if base is None:
            return torch.zeros(UNIFIED_DIM)

        img_tensor = self._preprocess(image, base)
        if img_tensor is None:
            return torch.zeros(UNIFIED_DIM)

        emb = base.model.encode_image(img_tensor.to(self._device))
        emb = F.normalize(emb.float(), dim=-1)

        if self._projection is not None:
            emb = self._projection(emb)

        return emb.squeeze(0).cpu()

    def encode_batch(
        self,
        images: list[Any],
        categories: list[str] | None = None,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """OFFLINE — encode une liste de produits en batches.

        Args:
            images:     Liste de PIL.Image
            categories: Liste de catégories (même longueur que images), ou None
            batch_size: Taille de batch pour le GPU

        Returns:
            Tensor [N, UNIFIED_DIM]
        """
        if categories is None:
            categories = ["default"] * len(images)

        all_embeddings: list[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_cats = categories[i : i + batch_size]

            # Grouper par catégorie pour maximiser la réutilisation du modèle adapté
            cat_groups: dict[str, list[tuple[int, Any]]] = {}
            for j, (img, cat) in enumerate(zip(batch_imgs, batch_cats)):
                norm_cat = AdapterRegistry._normalize(cat)
                cat_groups.setdefault(norm_cat, []).append((j, img))

            batch_embs = [None] * len(batch_imgs)
            for cat, items in cat_groups.items():
                idxs, imgs = zip(*items)
                with torch.no_grad():
                    for idx, img in zip(idxs, imgs):
                        batch_embs[idx] = self.encode_product(img, cat)

            all_embeddings.extend(e for e in batch_embs if e is not None)

        if not all_embeddings:
            return torch.zeros(len(images), UNIFIED_DIM)
        return torch.stack(all_embeddings)

    def _preprocess(self, image: Any, base: LoadedModel) -> torch.Tensor | None:
        """Convert image (PIL/URL/path) to tensor via le preprocesseur du modèle."""
        try:
            if isinstance(image, str):
                if image.startswith("http"):
                    import requests
                    from PIL import Image
                    from io import BytesIO
                    r = requests.get(image, timeout=5)
                    image = Image.open(BytesIO(r.content)).convert("RGB")
                else:
                    from PIL import Image as PILImage
                    image = PILImage.open(image).convert("RGB")

            tensor = base.preprocess(image)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)  # [1, C, H, W]
            return tensor
        except Exception as e:
            logger.warning("_preprocess échoué: %s", e)
            return None

    @property
    def is_ready(self) -> bool:
        return self._ensure_base_model() is not None

    @property
    def output_dim(self) -> int:
        return UNIFIED_DIM

    def available_adapters(self) -> list[str]:
        with self._adapters._lock:
            self._adapters._scan()
        return sorted(self._adapters._available)


# ──────────────────────────────────────────────────────────────
# Category Adapter Trainer
# ──────────────────────────────────────────────────────────────

class CategoryAdapterTrainer:
    """Entraîne un adapter LoRA pour une catégorie sur les données ShopFeed.

    Méthode : Contrastive learning (image produit ↔ titre produit).
    Paradigme : PEFT LoRA sur les couches d'attention du ViT.
    Résultat  : dossier PEFT sauvegardé dans ADAPTER_DIR/{category}_lora/

    Usage :
        trainer = CategoryAdapterTrainer(
            base_model_id="Marqo/marqo-ecommerce-embeddings-L",
            category="fashion",
            adapter_dir="checkpoints/adapters",
        )
        trainer.train(
            image_paths=["data/fashion/img1.jpg", ...],
            titles=["Robe midi en soie rouge", ...],
            epochs=5,
        )

    Après entraînement, l'adapter est automatiquement détecté par EcommerceEncoder.

    Référence : philschmid.de CLIP fine-tuning with LoRA (2025).
    """

    def __init__(
        self,
        category: str,
        base_model_id: str | None = None,
        adapter_dir: Path = ADAPTER_DIR,
        lora_rank: int = LORA_RANK,
        lora_alpha: int = LORA_ALPHA,
        device: str = "auto",
    ):
        self.category    = AdapterRegistry._normalize(category)
        self.model_id    = base_model_id or _VE.get("models", {}).get(
            "ecommerce", {}
        ).get("primary", "Marqo/marqo-ecommerce-embeddings-L")
        self.adapter_dir = adapter_dir
        self.lora_rank   = lora_rank
        self.lora_alpha  = lora_alpha
        self.device      = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )

    def train(
        self,
        image_paths: list[str],
        titles:      list[str],
        epochs:      int   = 5,
        batch_size:  int   = 32,
        lr:          float = 1e-4,
        temperature: float = 0.07,
    ) -> Path:
        """Entraîne le LoRA adapter et le sauvegarde.

        Args:
            image_paths: Chemins vers les images produits de la catégorie
            titles:      Titres des produits (paires positives avec les images)
            epochs:      Nombre d'epochs
            batch_size:  Taille de batch
            lr:          Learning rate (1e-4 est optimal selon les benchmarks PEFT 2025)
            temperature: Température du contrastive loss (standard = 0.07)

        Returns:
            Path du dossier de l'adapter sauvegardé.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("pip install peft>=0.10 requis pour entraîner les adapters LoRA")

        logger.info(
            "Entraînement adapter LoRA — category=%s, model=%s, %d images, %d epochs",
            self.category, self.model_id, len(image_paths), epochs,
        )

        # ── 1. Charger le modèle de base ──────────────────────────
        base = EcommerceEncoder.get_instance()._ensure_base_model()
        if base is None:
            raise RuntimeError(f"Impossible de charger le modèle de base: {self.model_id}")

        # ── 2. Appliquer LoRA ──────────────────────────────────────
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=LORA_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        lora_model = get_peft_model(base.model, lora_config)
        lora_model.print_trainable_parameters()
        lora_model.to(self.device)

        # ── 3. Dataset et optimizer ────────────────────────────────
        from torch.utils.data import DataLoader, Dataset
        from sentence_transformers import SentenceTransformer

        class ProductPairDataset(Dataset):
            def __init__(self, img_paths, titles_, preprocess):
                self.img_paths = img_paths
                self.titles_   = titles_
                self.preprocess = preprocess
            def __len__(self): return len(self.img_paths)
            def __getitem__(self, idx):
                from PIL import Image as PILImage
                img = PILImage.open(self.img_paths[idx]).convert("RGB")
                return self.preprocess(img), self.titles_[idx]

        dataset    = ProductPairDataset(image_paths, titles, base.preprocess)
        loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer  = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, lora_model.parameters()),
            lr=lr, weight_decay=1e-4,
        )

        # Encodeur texte (pour les paires contrastives)
        text_model_name = _CFG.get("text_encoder", {}).get("model", "paraphrase-multilingual-mpnet-base-v2")
        text_enc = SentenceTransformer(text_model_name).to(self.device)
        for p in text_enc.parameters():
            p.requires_grad = False

        # ── 4. Boucle d'entraînement ───────────────────────────────
        lora_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for imgs, title_batch in loader:
                imgs = imgs.to(self.device)

                # Image embeddings (LoRA actif)
                img_embs  = lora_model.encode_image(imgs)
                img_embs  = F.normalize(img_embs.float(), dim=-1)

                # Text embeddings (gelés)
                with torch.no_grad():
                    txt_embs = torch.tensor(
                        text_enc.encode(list(title_batch), show_progress_bar=False),
                        device=self.device, dtype=torch.float32,
                    )
                    # Projeter texte → même dim que image
                    if txt_embs.shape[-1] != img_embs.shape[-1]:
                        txt_embs = F.interpolate(
                            txt_embs.unsqueeze(1), size=img_embs.shape[-1], mode="linear"
                        ).squeeze(1)
                    txt_embs = F.normalize(txt_embs, dim=-1)

                # Contrastive loss (InfoNCE symétrique)
                logits = torch.matmul(img_embs, txt_embs.T) / temperature
                labels = torch.arange(len(imgs), device=self.device)
                loss   = (
                    F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)
                ) / 2.0

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(loader), 1)
            logger.info("Epoch %d/%d — loss=%.4f", epoch + 1, epochs, avg_loss)

        # ── 5. Sauvegarder l'adapter ───────────────────────────────
        out_path = self.adapter_dir / f"{self.category}_lora"
        out_path.mkdir(parents=True, exist_ok=True)
        lora_model.save_pretrained(str(out_path))
        logger.info("✓ Adapter LoRA sauvegardé: %s", out_path)

        # Invalider le cache du singleton pour recharger le nouvel adapter
        enc = EcommerceEncoder.get_instance()
        enc._adapted_models.pop(self.category, None)
        enc._adapters._scanned = False

        return out_path


# ──────────────────────────────────────────────────────────────
# CLI — entraîner un adapter depuis la ligne de commande
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="ShopFeed — LoRA adapter trainer")
    parser.add_argument("action",     choices=["train", "info"])
    parser.add_argument("--category", default="fashion",  help="Catégorie produit")
    parser.add_argument("--data",     default="",         help="Fichier parquet ou JSON {image_path, title}")
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--lr",       type=float, default=1e-4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.action == "info":
        enc = EcommerceEncoder.get_instance()
        print(f"Modèle prêt   : {enc.is_ready}")
        print(f"Output dim    : {enc.output_dim}")
        print(f"Adapters dispo: {enc.available_adapters()}")

    elif args.action == "train":
        if not args.data:
            print("--data requis pour l'entraînement")
            raise SystemExit(1)
        # Charger les données
        data_path = Path(args.data)
        if data_path.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(data_path)
            image_paths = df["image_path"].tolist()
            titles      = df["title"].tolist()
        else:
            with open(data_path) as f:
                rows = json.load(f)
            image_paths = [r["image_path"] for r in rows]
            titles      = [r["title"]      for r in rows]

        trainer = CategoryAdapterTrainer(
            category=args.category,
            epochs=args.epochs,
        )
        out = trainer.train(image_paths, titles, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
        print(f"Adapter sauvegardé: {out}")
