"""
Duplicate Detector — détection de photos dupliquées via hash perceptuel (pHash).

Utilise imagehash.phash() pour calculer un fingerprint robuste aux:
  - Redimensionnements
  - Compressions JPEG légères
  - Légères modifications de couleur / luminosité

Distance de Hamming ≤ 8 → images considérées dupliquées.
Utilisé dans le pipeline de modération et upload produit.
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_phash(image_url: str, hash_size: int = 16) -> Optional[str]:
    """
    Calcule le hash perceptuel (pHash) d'une image depuis son URL.

    Args:
        image_url: URL accessible publiquement
        hash_size: taille du hash (16 = 256 bits de hash)

    Returns:
        str — hash hex, ou None si erreur
    """
    try:
        import imagehash
        from PIL import Image
        import httpx
        import io

        resp = httpx.get(image_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        phash = imagehash.phash(img, hash_size=hash_size)
        return str(phash)

    except ImportError:
        logger.warning("imagehash not installed. Duplicate detection disabled.")
        return None
    except Exception as e:
        logger.error(f"pHash computation failed for {image_url}: {e}")
        return None


def is_duplicate(
    new_phash: str,
    existing_hashes: list[str],
    max_hamming_distance: int = 8,
) -> bool:
    """
    Vérifie si une image est dupliquée d'une image existante.

    Args:
        new_phash:           hash de la nouvelle image
        existing_hashes:     liste de hashes existants à comparer
        max_hamming_distance: seuil de distance (défaut: 8 sur 256 bits)

    Returns:
        True si un doublon est trouvé
    """
    try:
        import imagehash
        new_h = imagehash.hex_to_hash(new_phash)
        for existing in existing_hashes:
            try:
                existing_h = imagehash.hex_to_hash(existing)
                if new_h - existing_h <= max_hamming_distance:
                    return True
            except Exception:
                continue
        return False
    except ImportError:
        return False
    except Exception as e:
        logger.error(f"Duplicate check failed: {e}")
        return False


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Retourne la distance de Hamming entre deux pHash.
    0 = identique, 256 = complètement différent.
    """
    try:
        import imagehash
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return int(h1 - h2)
    except Exception:
        return -1


def find_duplicates_in_batch(
    image_urls: list[str],
    max_hamming_distance: int = 8,
) -> list[tuple[int, int, int]]:
    """
    Détecte les paires de doublons dans un batch d'images.

    Returns:
        Liste de (idx_a, idx_b, hamming_distance) pour les paires dupliquées
    """
    hashes = [compute_phash(url) for url in image_urls]
    duplicates = []

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i] is None or hashes[j] is None:
                continue
            dist = hamming_distance(hashes[i], hashes[j])
            if 0 <= dist <= max_hamming_distance:
                duplicates.append((i, j, dist))

    return duplicates
