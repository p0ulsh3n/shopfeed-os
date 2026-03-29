"""
Geo Classifier — Classification géographique niveau commande (L1-L4)
MLP backup si PostGIS non disponible ou adresse ambiguë.

Architecture:
  Input: [buyer_lat, buyer_lon, vendor_lat, vendor_lon,
          buyer_country_emb, vendor_country_emb, distance_km_normalized]
  MLP: [7 → 64 → 32 → 4]
  Output: softmax [P(L1), P(L2), P(L3), P(L4)]
  Training: order_geo_classifications confirmées par PostGIS
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

# Labels des niveaux géographiques
GEO_LEVELS = ["L1", "L2", "L3", "L4"]

# Nombre de pays supportés (embedding lookup)
NUM_COUNTRIES = 256

# Rayon de la Terre (km) pour calcul Haversine
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcule la distance à vol d'oiseau entre deux coordonnées GPS (km)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


class GeoClassifier(nn.Module):
    """
    Classifieur géographique MLP pour déterminer le niveau de livraison L1-L4.

    L1 : Même commune / quartier (hyper-local)
    L2 : Même ville (intra-city)
    L3 : Même pays, villes différentes (national)
    L4 : International

    Utilisé comme backup si PostGIS ST_Contains n'est pas disponible.
    Training: order_geo_classifications avec décision PostGIS comme ground truth.
    """

    def __init__(
        self,
        num_countries: int = NUM_COUNTRIES,
        country_emb_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.country_emb_dim = country_emb_dim

        # Embeddings pays (buyer + vendor)
        self.country_embedding = nn.Embedding(num_countries, country_emb_dim)

        # Feature input: lat(1) + lon(1) + lat(1) + lon(1) + country_emb*2 + distance(1)
        input_dim = 4 + country_emb_dim * 2 + 1  # = 21 avec emb_dim=8

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 classes: L1, L2, L3, L4
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init pour les couches linéaires."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _normalize_coords(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise lat/lon en [-1, 1]"""
        return lat / 90.0, lon / 180.0

    def forward(
        self,
        buyer_lat: torch.Tensor,        # [batch]
        buyer_lon: torch.Tensor,        # [batch]
        vendor_lat: torch.Tensor,       # [batch]
        vendor_lon: torch.Tensor,       # [batch]
        buyer_country: torch.Tensor,    # [batch] — int index
        vendor_country: torch.Tensor,   # [batch] — int index
        distance_km: torch.Tensor,      # [batch]
    ) -> torch.Tensor:
        """Returns logits [batch, 4]"""
        blat, blon = self._normalize_coords(buyer_lat, buyer_lon)
        vlat, vlon = self._normalize_coords(vendor_lat, vendor_lon)
        dist_norm = torch.log1p(distance_km) / 10.0  # log(1+d) / 10

        buyer_emb = self.country_embedding(buyer_country)    # [B, emb_dim]
        vendor_emb = self.country_embedding(vendor_country)  # [B, emb_dim]

        x = torch.cat([
            blat.unsqueeze(1), blon.unsqueeze(1),
            vlat.unsqueeze(1), vlon.unsqueeze(1),
            buyer_emb, vendor_emb,
            dist_norm.unsqueeze(1),
        ], dim=1)  # [B, input_dim]

        return self.mlp(x)  # [B, 4]

    def predict_level(
        self,
        buyer_lat: float, buyer_lon: float,
        vendor_lat: float, vendor_lon: float,
        buyer_country_idx: int, vendor_country_idx: int,
    ) -> str:
        """
        Prédit le niveau géographique pour une commande.
        Retourne 'L1', 'L2', 'L3' ou 'L4'.
        """
        with torch.no_grad():
            dist = haversine_km(buyer_lat, buyer_lon, vendor_lat, vendor_lon)
            t = lambda v: torch.tensor([v], dtype=torch.float32)
            ti = lambda v: torch.tensor([v], dtype=torch.long)

            logits = self.forward(
                t(buyer_lat), t(buyer_lon),
                t(vendor_lat), t(vendor_lon),
                ti(buyer_country_idx), ti(vendor_country_idx),
                t(dist),
            )
            pred_idx = logits.argmax(dim=-1).item()
            return GEO_LEVELS[pred_idx]

    def predict_proba(
        self,
        buyer_lat: float, buyer_lon: float,
        vendor_lat: float, vendor_lon: float,
        buyer_country_idx: int, vendor_country_idx: int,
    ) -> dict[str, float]:
        """Retourne un dict de probabilités par niveau."""
        with torch.no_grad():
            dist = haversine_km(buyer_lat, buyer_lon, vendor_lat, vendor_lon)
            t = lambda v: torch.tensor([v], dtype=torch.float32)
            ti = lambda v: torch.tensor([v], dtype=torch.long)

            logits = self.forward(
                t(buyer_lat), t(buyer_lon),
                t(vendor_lat), t(vendor_lon),
                ti(buyer_country_idx), ti(vendor_country_idx),
                t(dist),
            )
            probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
            return {level: round(p, 4) for level, p in zip(GEO_LEVELS, probs)}


# ── Heuristiques de règles simples (backup sans modèle chargé) ───────────────

def rule_based_geo_level(
    buyer_country: str,
    vendor_country: str,
    buyer_city: str,
    vendor_city: str,
    buyer_commune: str = "",
    vendor_commune: str = "",
    distance_km: float | None = None,
) -> str:
    """
    Classification par règles simples.
    Utilisé si le modèle ML n'est pas disponible.
    """
    if buyer_country != vendor_country:
        return "L4"

    buyer_city_l = buyer_city.strip().lower()
    vendor_city_l = vendor_city.strip().lower()

    if buyer_city_l != vendor_city_l:
        return "L3"

    buyer_com_l = buyer_commune.strip().lower()
    vendor_com_l = vendor_commune.strip().lower()

    if buyer_com_l and vendor_com_l and buyer_com_l == vendor_com_l:
        return "L1"

    # Distance numérique si disponible
    if distance_km is not None:
        if distance_km <= 5:
            return "L1"
        elif distance_km <= 30:
            return "L2"

    return "L2"
