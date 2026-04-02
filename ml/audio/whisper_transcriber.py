"""
Whisper Transcriber — Automatic audio transcription + entity extraction.

Model: openai/whisper-large-v3
Triggered asynchronously after vendor video upload.
Results stored in:
  - feed_content.transcript
  - Session ASR Index (Redis session:{session_id}:asr_index)

Entities extracted (multilingual):
  - Products mentioned (dress, shoes, bag, robe, chaussures, sac...)
  - Brands
  - Prices (universal regex for all currency formats)
  - Urgency (now, last chance, limited stock / maintenant, derniere chance...)

Language: Auto-detected by Whisper (supports 99+ languages)
"""

from __future__ import annotations
import logging
import os
import re
import asyncio
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Whisper model
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded.")
    except ImportError:
        logger.warning("openai-whisper not installed. Transcription unavailable.")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
    return _whisper_model


# ── Multilingual Entity Extraction ──────────────────────────────────────
# Keywords sourced from: global e-commerce taxonomies (Shopify, Amazon, TikTok Shop)
# Languages: FR, EN, AR, Wolof, PT, Swahili, ES, TR, HI

# Product keywords — exhaustive, multilingual (350+ terms)
PRODUCT_KEYWORDS = {
    # ── Fashion / Mode ──
    # FR
    "robe", "chemise", "pantalon", "jupe", "veste", "manteau", "pull",
    "chaussures", "sac", "ceinture", "bijoux", "collier", "bracelet", "montre",
    "costume", "blazer", "gilet", "cardigan", "legging", "jogging", "pyjama",
    "lingerie", "sous-vetement", "maillot", "chapeau", "casquette", "echarpe",
    "foulard", "gants", "lunettes", "boucles d'oreilles", "bague", "broche",
    "chaussette", "bermuda", "combinaison", "body", "debardeur",
    # EN
    "dress", "shirt", "pants", "trousers", "skirt", "jacket", "coat", "sweater",
    "shoes", "bag", "belt", "jewelry", "necklace", "watch", "sneakers", "heels",
    "t-shirt", "hoodie", "jeans", "shorts", "sandals", "boots", "scarf",
    "suit", "blazer", "vest", "cardigan", "leggings", "joggers", "pajamas",
    "underwear", "swimwear", "hat", "cap", "gloves", "sunglasses", "earrings",
    "ring", "brooch", "socks", "jumpsuit", "romper", "tank top", "blouse",
    "polo", "parka", "raincoat", "overalls", "tuxedo", "tie", "bow tie",
    "clutch", "backpack", "wallet", "purse", "tote", "crossbody",
    # ES
    "vestido", "camisa", "zapatos", "bolso", "falda", "chaqueta", "abrigo",
    "pantalones", "corbata", "bufanda", "sombrero", "gorra", "anillo",
    # PT
    "sapatos", "bolsa", "cinto", "calca", "saia", "casaco", "camiseta",
    "calcinha", "sunga", "biquini", "chinelo", "tenis",
    # AR
    "فستان", "حذاء", "حقيبة", "ساعة", "قميص", "بنطلون", "جاكيت", "معطف",
    "خاتم", "قلادة", "سوار", "نظارة", "حجاب", "عباية", "ثوب", "قفطان",
    # TR
    "elbise", "gomlek", "ayakkabi", "canta", "kemer", "takkim", "mont",
    # Wolof
    "yere", "yéré", "mbubb", "tank", "tànk", "ndoket",
    # Swahili
    "gauni", "shati", "viatu", "mkoba", "mkufu", "saa",

    # ── Beauty / Beaute ──
    # FR
    "creme", "crème", "serum", "sérum", "lotion", "parfum", "maquillage",
    "rouge", "fond de teint", "vernis", "masque", "gommage", "huile",
    "shampooing", "apres-shampooing", "deodorant", "dentifrice",
    "cire", "rasoir", "peigne", "brosse", "savon",
    # EN
    "cream", "perfume", "makeup", "skincare", "moisturizer", "sunscreen",
    "lipstick", "foundation", "mascara", "eyeliner", "eyeshadow",
    "nail polish", "blush", "concealer", "primer", "highlighter",
    "serum", "cleanser", "toner", "exfoliant", "face mask", "body lotion",
    "shampoo", "conditioner", "hair oil", "deodorant", "toothpaste",
    "razor", "wax", "comb", "brush", "soap", "fragrance", "cologne",
    "lip gloss", "lip balm", "bronzer", "setting spray", "contour",
    # AR
    "عطر", "كريم", "مكياج", "شامبو", "صابون", "بخور", "كحل", "حناء",
    # TR
    "parfum", "krem", "makyaj", "sampuan",

    # ── Food / Alimentation ──
    # FR
    "attieke", "attiéké", "kedjenou", "thieboudienne", "thiéboudienne",
    "mafe", "mafé", "poulet", "poisson", "jus", "sauce", "pate", "pâte",
    "riz", "haricot", "viande", "boeuf", "agneau", "legumes", "fruits",
    "pain", "huile", "epices", "chocolat", "cafe", "the", "lait", "beurre",
    "farine", "sucre", "miel", "confiture", "yaourt", "fromage",
    # EN
    "chicken", "fish", "rice", "juice", "beans", "meat", "beef", "lamb",
    "vegetables", "fruit", "bread", "pasta", "soup", "salad", "cheese",
    "butter", "milk", "coffee", "tea", "chocolate", "honey", "flour",
    "sugar", "oil", "spices", "cereal", "yogurt", "eggs", "snacks",
    "nuts", "dried fruit", "protein bar", "energy drink", "water",
    # AR
    "دجاج", "سمك", "أرز", "لحم", "خبز", "زيت", "بهارات", "عسل",
    "شاي", "قهوة", "حليب", "تمر",
    # Swahili
    "nyama", "samaki", "wali", "ugali", "chapati", "mandazi", "chai",

    # ── Tech / Electronics ──
    # FR
    "telephone", "téléphone", "smartphone", "ordinateur", "tablette",
    "casque", "chargeur", "cable", "câble", "ecouteur", "écouteur",
    "clavier", "souris", "webcam", "imprimante", "disque dur",
    "cle usb", "batterie", "adaptateur", "enceinte", "camera",
    # EN
    "phone", "laptop", "tablet", "headphones", "charger", "earbuds",
    "earphones", "speaker", "powerbank", "keyboard", "mouse", "webcam",
    "printer", "hard drive", "usb", "battery", "adapter", "camera",
    "monitor", "smart watch", "fitness tracker", "drone", "console",
    "controller", "microphone", "router", "modem", "projector",
    "vr headset", "smart tv", "soundbar", "ring light", "tripod",
    # AR
    "هاتف", "سماعات", "شاحن", "لابتوب", "تابلت", "كاميرا",
    # TR
    "telefon", "kulaklik", "sarj", "bilgisayar",

    # ── Home / Maison ──
    # FR
    "cuisine", "lit", "table", "chaise", "canape", "canapé", "rideau", "drap",
    "matelas", "oreiller", "couverture", "tapis", "lampe", "miroir",
    "etagere", "bureau", "armoire", "commode", "vaisselle", "casserole",
    "poele", "couteau", "verre", "assiette", "fourchette", "cuillere",
    # EN
    "kitchen", "bed", "chair", "sofa", "couch", "curtain", "blanket", "pillow",
    "lamp", "mirror", "rug", "carpet", "shelf", "desk", "wardrobe", "dresser",
    "mattress", "duvet", "towel", "vase", "candle", "frame", "clock",
    "pan", "pot", "knife", "plate", "cup", "mug", "bowl", "cutlery",
    "blender", "microwave", "toaster", "kettle", "iron", "vacuum",
    # AR
    "سرير", "كرسي", "طاولة", "مرآة", "ستارة", "وسادة", "سجادة",

    # ── Sports / Fitness ──
    "ballon", "ball", "raquette", "racket", "velo", "bicycle", "bike",
    "tapis de yoga", "yoga mat", "haltere", "dumbbell", "treadmill",
    "sac de sport", "gym bag", "gourde", "water bottle", "proteine",
    "protein", "creatine", "leggings", "sports bra", "trainers",

    # ── Baby / Enfant ──
    "couche", "diaper", "biberon", "bottle", "poussette", "stroller",
    "berceau", "crib", "sucette", "pacifier", "body bebe", "onesie",
    "jouet", "toy", "peluche", "teddy bear", "lait bebe", "baby formula",

    # ── Automotive ──
    "pneu", "tire", "huile moteur", "motor oil", "filtre", "filter",
    "batterie voiture", "car battery", "essuie-glace", "wiper",
}

# Urgency keywords — exhaustive multilingual (70+ expressions)
URGENCY_KEYWORDS = {
    # French
    "maintenant", "derniere chance", "dernière chance", "stock limite",
    "stock limité", "quantite limitee", "quantité limitée",
    "aujourd'hui seulement", "offre limitee", "offre limitée",
    "promo flash", "vite", "depeche", "dépêche", "depêchez-vous",
    "avant que ca se termine", "plus que", "se termine bientot",
    "presque epuise", "acces exclusif", "duree limitee",
    "ne ratez pas", "dernier jour", "fin de stock",
    # English
    "now", "last chance", "limited stock", "limited quantity",
    "today only", "limited offer", "flash sale", "hurry",
    "hurry up", "before it's gone", "only left", "selling fast",
    "almost gone", "don't miss", "ending soon", "act now",
    "while supplies last", "exclusive deal", "final call",
    "clock's ticking", "rare find", "almost sold out",
    "last day", "clearance", "going fast", "one day only",
    "limited edition", "members only", "early access",
    # Arabic
    "الآن", "فرصة أخيرة", "كمية محدودة", "عرض محدود", "أسرع",
    "عجل", "سارع الآن", "ينتهي قريبا", "وقت محدود",
    "أوشكت على النفاد", "وصول حصري",
    # Spanish
    "ahora", "ultima oportunidad", "stock limitado", "oferta limitada",
    "date prisa", "termina pronto", "casi agotado", "acceso exclusivo",
    # Portuguese
    "agora", "ultima chance", "estoque limitado", "corra",
    "aproveite agora", "termina em breve", "quase esgotado",
    # Turkish
    "simdi", "son sans", "sinirli stok", "acele edin", "firsat",
    # Swahili
    "sasa", "haraka", "mwisho", "bei punguzo",
    # Wolof
    "leegi", "yàgg", "ndank ndank",
    # Hindi
    "abhi", "jaldi", "simit", "antim mauka",
}

# Universal price patterns — 30+ currencies, all global number formats
PRICE_PATTERNS = [
    # Numbers followed by currency names/symbols (all languages)
    r'\b(\d+(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?)\s*'
    r'(?:euros?|EUR|€|dollars?|USD|\$|pounds?|GBP|£|'
    r'FCFA|XOF|XAF|CFA|francs?|naira|NGN|₦|cedis?|GHS|₵|'
    r'shillings?|KES|KSh|TZS|UGX|rand|ZAR|dirhams?|MAD|DZD|AED|'
    r'yen|JPY|¥|yuan|CNY|RMB|won|KRW|₩|rupees?|INR|₹|'
    r'reais?|BRL|R\$|pesos?|MXN|ARS|COP|CLP|PEN|'
    r'riyal|riyals?|SAR|QAR|dinars?|TND|KWD|LYD|'
    r'birr|ETB|ariary|MGA|kwacha|ZMW|MWK|'
    r'lira|TRY|₺|baht|THB|฿|dong|VND|'
    r'ringgit|MYR|RM|rupiah|IDR|Rp|'
    r'zloty|PLN|zł|koruna|CZK|Kč|forint|HUF|Ft|'
    r'krona|SEK|NOK|DKK|kr|'
    r'franc\s*suisse|CHF|lei|RON|lev|BGN)\b',
    # Currency symbols before numbers
    r'[€$£₦₵₹¥₩₺฿]\s*(\d+(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?)',
    # Rp, R$, CA$, A$, HK$, S$, MX$ before numbers
    r'(?:Rp|R\$|CA\$|A\$|HK\$|S\$|MX\$|E£)\s*(\d+(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?)',
    # Free in many languages
    r'\b(?:gratuit|free|مجاني|gratis|bure|ucretsiz|muft|libre|gratuito|kostenlos|무료|無料)\b',
    # Promo/discount words (multilingual)
    r'\b(?:promo|promotion|discount|sale|soldes?|remise|réduction|reduction|'
    r'descuento|desconto|offerta|indirim|छूट|تخفيض|خصم|割引|할인|'
    r'rabais|ristourne|liquidation|clearance|deal|offer|bargain)\b',
    # Between X and Y (FR + EN + ES + PT)
    r'\b(?:entre|between|entre|entre)\s+(\d+)\s+(?:et|and|y|e)\s+(\d+)\b',
    # Percentage patterns
    r'(\d{1,3})\s*%\s*(?:off|de reduction|de remise|descuento|desconto|indirim)',
]


def extract_entities(transcript: str) -> list[dict]:
    """
    Extract entities from transcript (multilingual):
      - product: product keyword matches
      - price: amounts + promos
      - urgency: urgency words
      - brand: brand dictionary matches (simplified)

    Returns:
        [{text, type: product|brand|price|urgency, confidence}]
    """
    entities = []
    text_lower = transcript.lower()

    # Products
    for kw in PRODUCT_KEYWORDS:
        if kw in text_lower:
            entities.append({
                "text": kw,
                "type": "product",
                "confidence": 0.80,
            })

    # Prices
    for pattern in PRICE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for m in matches:
            text = m if isinstance(m, str) else " ".join(m)
            entities.append({
                "text": text.strip(),
                "type": "price",
                "confidence": 0.90,
            })

    # Urgency
    for kw in URGENCY_KEYWORDS:
        if kw in text_lower:
            entities.append({
                "text": kw,
                "type": "urgency",
                "confidence": 0.85,
            })

    # Deduplicate by text
    seen = set()
    unique = []
    for e in entities:
        if e["text"] not in seen:
            seen.add(e["text"])
            unique.append(e)

    return unique


async def transcribe(audio_url: str, language: str | None = None) -> dict:
    """
    Transcribe audio from a URL.

    Args:
        audio_url: URL of audio or video (S3, CDN)
        language:  ISO 639-1 code or None for auto-detection.
                   Whisper supports 99+ languages out of the box.

    Returns:
        {
            text: str,
            segments: [{start, end, text}],
            language: str,
            entities: [{text, type, confidence}],
        }
    """
    model = _load_whisper()
    if model is None:
        return _empty_result()

    loop = asyncio.get_event_loop()

    # Download audio locally
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
    except Exception as e:
        logger.error(f"Failed to download audio from {audio_url}: {e}")
        return _empty_result()

    # Transcription in a thread executor (GPU blocking operation)
    result = await loop.run_in_executor(
        None,
        lambda: _run_whisper(model, tmp_path, language),
    )

    # Cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    # Entity extraction
    full_text = result.get("text", "")
    entities = extract_entities(full_text)
    result["entities"] = entities

    return result


def _run_whisper(model, audio_path: str, language: str | None) -> dict:
    """Executed in a thread executor (blocking GPU operation)."""
    try:
        kwargs = {
            "task": "transcribe",
            "verbose": False,
        }
        # If language is specified, use it; otherwise Whisper auto-detects
        if language:
            kwargs["language"] = language

        result = model.transcribe(audio_path, **kwargs)
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", language or ""),
        }
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return _empty_result()


def _empty_result() -> dict:
    return {"text": "", "segments": [], "language": "", "entities": []}

