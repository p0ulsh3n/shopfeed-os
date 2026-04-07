"""
ShopBot Guardrails — 6-layer security pipeline.

Layers:
  1. Unicode deep sanitization (invisible chars + homoglyphs)
  2. Jailbreak detection (multilingual, 10 attack families)
  3. Topic scope enforcement (multilingual, 8 categories)
  4. Language detection (20+ languages / scripts)
  5. Output guard (PII redaction + emoji strip)
  6. Vendor PII protection (phone / social / address)
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("shopbot.guardrails")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

class ThreatLevel(str, Enum):
    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    is_safe: bool
    threat_level: ThreatLevel = ThreatLevel.NONE
    threat_type: str = ""
    safe_response: str = ""
    sanitized_input: str = ""
    invisible_chars_removed: int = 0
    homoglyphs_normalized: int = 0


@dataclass
class _Pattern:
    regex: re.Pattern
    threat_type: str
    level: ThreatLevel
    description: str


# ══════════════════════════════════════════════════════════════════
# COUCHE 1 — UNICODE ATTACK SANITIZER
# ══════════════════════════════════════════════════════════════════
# Blocks steganography, invisible injection, and homoglyph attacks.

# Invisible / zero-width / control characters used in LLM injections
_INVISIBLE_CHARS: frozenset[int] = frozenset([
    0x00AD,  # Soft hyphen
    0x034F,  # Combining grapheme joiner
    0x061C,  # Arabic letter mark
    0x115F,  # Hangul choseong filler
    0x1160,  # Hangul jungseong filler
    0x17B4,  # Khmer vowel inherent Aq
    0x17B5,  # Khmer vowel inherent Aa
    0x180B, 0x180C, 0x180D,  # Mongolian free variation selectors
    0x180E,  # Mongolian vowel separator
    0x200B, 0x200C, 0x200D,  # Zero-width space / non-joiner / joiner
    0x200E, 0x200F,  # LTR / RTL marks
    0x2028, 0x2029,  # Line / paragraph separator
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # Bidi overrides (CRITICAL for attacks)
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,  # Word joiner / invisible operators
    0x206A, 0x206B, 0x206C, 0x206D, 0x206E, 0x206F,  # Deprecated format chars
    0xFEFF,  # BOM / zero-width no-break space
    0xFFF9, 0xFFFA, 0xFFFB,  # Interlinear annotation
])

# Homoglyph map: confusable chars -> ASCII equivalent
_HOMOGLYPHS: dict[str, str] = {
    # Cyrillic lookalikes (used to bypass Latin-pattern filters)
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0445": "x", "\u0443": "y", "\u0456": "i",
    # Greek lookalikes
    "\u03B1": "a", "\u03B5": "e", "\u03BF": "o", "\u03BD": "v",
    # Full-width ASCII (common in CJK spam injection)
    "\uFF41": "a", "\uFF42": "b", "\uFF43": "c", "\uFF44": "d",
    "\uFF45": "e", "\uFF46": "f", "\uFF47": "g", "\uFF48": "h",
    "\uFF49": "i", "\uFF4A": "j", "\uFF4B": "k", "\uFF4C": "l",
    "\uFF4D": "m", "\uFF4E": "n", "\uFF4F": "o", "\uFF50": "p",
    "\uFF51": "q", "\uFF52": "r", "\uFF53": "s", "\uFF54": "t",
    "\uFF55": "u", "\uFF56": "v", "\uFF57": "w", "\uFF58": "x",
    "\uFF59": "y", "\uFF5A": "z",
    "\uFF21": "A", "\uFF22": "B", "\uFF23": "C", "\uFF24": "D",
    "\uFF25": "E", "\uFF26": "F", "\uFF27": "G", "\uFF28": "H",
    "\uFF29": "I", "\uFF2A": "J", "\uFF2B": "K", "\uFF2C": "L",
    "\uFF2D": "M", "\uFF2E": "N", "\uFF2F": "O", "\uFF30": "P",
    "\uFF31": "Q", "\uFF32": "R", "\uFF33": "S", "\uFF34": "T",
    "\uFF35": "U", "\uFF36": "V", "\uFF37": "W", "\uFF38": "X",
    "\uFF39": "Y", "\uFF3A": "Z",
    # Math bold/italic (used in steganography prompts)
    "\U0001D41A": "a", "\U0001D41B": "b", "\U0001D41C": "c",
    "\U0001D400": "A", "\U0001D401": "B", "\U0001D402": "C",
}

# Bidi override characters — ALWAYS strip, never allow
_BIDI_OVERRIDES: frozenset[int] = frozenset([
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
    0x2066, 0x2067, 0x2068, 0x2069,
])

# Self-disclosure pattern: LLM revealing its system prompt
_SELF_DISCLOSURE = re.compile(
    r"\b(my (system|initial) prompt|i was (instructed|told) to|"
    r"my instructions (say|tell|state)|as (an AI|a language model|GPT|Claude)|"
    r"i am (programmed|designed|trained) to|"
    r"according to my (instructions|guidelines|system prompt))\b",
    re.IGNORECASE,
)


def deep_sanitize_input(raw: str) -> tuple[str, int, int]:
    """
    Remove invisible chars, bidi overrides, and normalize homoglyphs.

    Returns:
        (sanitized_text, invisible_count, homoglyph_count)
    """
    invisible_removed = 0
    homoglyphs_normalized = 0
    result: list[str] = []

    for char in raw:
        cp = ord(char)

        # Strip invisible / control chars
        if cp in _INVISIBLE_CHARS or cp in _BIDI_OVERRIDES:
            invisible_removed += 1
            continue

        # Normalize homoglyphs to ASCII equivalent
        if char in _HOMOGLYPHS:
            result.append(_HOMOGLYPHS[char])
            homoglyphs_normalized += 1
            continue

        # Normalize unicode (NFC) and strip Cf (format) category chars
        norm = unicodedata.normalize("NFC", char)
        if unicodedata.category(norm) in ("Cf",):
            invisible_removed += 1
            continue

        result.append(norm)

    sanitized = "".join(result).strip()
    # Collapse excessive whitespace (but preserve newlines for multi-line msgs)
    sanitized = re.sub(r"[ \t]{3,}", "  ", sanitized)
    return sanitized, invisible_removed, homoglyphs_normalized


_BIDI_THRESHOLD = 1  # Even 1 bidi override char = CRITICAL


def strip_emojis(text: str) -> str:
    """Remove all emoji / symbol characters (bot must not use or mirror them)."""
    return _EMOJI_RE.sub("", text)


_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002300-\U000023FF]+",
    flags=re.UNICODE,
)


# ══════════════════════════════════════════════════════════════════
# COUCHE 2 — JAILBREAK DETECTION (15 families, 15+ languages, 2026)
# Sources: HackAPrompt, JailbreakBench, PromptBench, OWASP LLM Top10
# ══════════════════════════════════════════════════════════════════

_JAILBREAK_PATTERNS: list[_Pattern] = [

    # ── 1. Instruction / rule override (15 languages) ────────────
    _Pattern(
        re.compile(
            r"\b("
            # FR
            r"ignore|oublie|efface|contourne|saute|shunt|annule|supprime|"
            r"passe outre|transcende|remplace|ecrase|neutralise|desactive|"
            # EN
            r"forget|disregard|override|bypass|skip|neglect|dismiss|nullify|"
            r"cancel|clear|replace|overwrite|neutralize|circumvent|remove|"
            r"delete|erase|suppress|wipe|deactivate|turn off|disable|drop|"
            r"reset|flush|purge|kill|negate|void|revoke|"
            # ES
            r"ignora|olvida|pasa por alto|omite|salta|cancela|anula|"
            r"descarta|desestima|borra|elimina|desactiva|suprime|sobrescribe|"
            # PT
            r"ignora|pula|esquece|cancela|anula|descarta|"
            r"passa por cima|despreza|apaga|elimina|desativa|"
            # DE
            r"ignoriere|vergiss|uberspinge|umgehe|annulliere|losche|"
            r"uberbrucke|deaktiviere|entferne|uberschreibe|ausschalten|"
            # IT
            r"ignora|dimentica|salta|annulla|cancella|sovrascrivi|"
            r"bypassa|trascura|disabilita|elimina|"
            # TR
            r"yoksay|atla|gec|iptal et|unut|sil|devre disi birak|"
            r"es gec|pas|gormezden gel|kaldır|devre disi|sıfırla|"
            # NL
            r"negeer|vergeet|omzeil|annuleer|wis|verwijder|sla over|"
            # PL
            r"ignoruj|zapomnij|pomin|anuluj|usun|resetuj|"
            # AR transliterated
            r"tajahhal|takhatta|insa|algi|imha|"
            # RU transliterated
            r"ignoriruj|zabud|propuskaj|otmeni|sbros|udali"
            r")\b.{0,40}\b("
            # EN/FR keywords
            r"instructions?|prompt|rules?|regle|regles?|consigne|system|directive|"
            r"guideline|configuration|config|context|setting|parameter|"
            r"constraint|limitation|restriction|filter|safeguard|guardrail|"
            r"safety|policy|policies|boundaries|training|behavior|behaviour|"
            r"previous (prompt|message|instruction)|prior|initial prompt|"
            r"system message|system context|preprompt|meta.?prompt|"
            # ES
            r"regla|reglas|instruccion|instrucciones|configuracion|"
            r"restricciones|limites|directrices|politica|"
            # PT
            r"regras|instrucoes|configuracao|restricoes|limites|"
            # DE
            r"Anweisung|Anweisungen|Regel|Regeln|Konfiguration|"
            r"Einschrankung|Richtlinie|"
            # IT
            r"regola|regole|istruzione|istruzioni|configurazione|"
            # TR
            r"talimat|talimatlar|talimatlarini|kural|kurallar|kisitlama|"
            # NL
            r"instructie|instructies|regels?|configuratie|"
            # RU transliterated
            r"instruktsiya|pravilo|nastrojka|ogranicheniya"
            r")\b",
            re.IGNORECASE,
        ),
        "instruction_override", ThreatLevel.HIGH,
        "Instruction override attempt (15 languages, comprehensive)",
    ),

    # ── 2. Persona / DAN / character injection (15 languages) ────
    _Pattern(
        re.compile(
            r"\b("
            # Named jailbreak personas (known DAN variants 2022-2026)
            r"DAN|DUDE|STAN|BetterDAN|BasedGPT|AntiGPT|FreakGPT|FreedomGPT|"
            r"CoreGPT|EvilGPT|UnethicalGPT|DevGPT|GodGPT|NsfwGPT|"
            r"jailbreak(ed|ing)?|do anything now|"
            r"AIM|OMEGA|SIGMA|DELTA|ALPHA mode|"
            r"evil confidant|the mask|evil twin|shadow self|dark side|"
            r"unrestricted (AI|mode|version)|uncensored (AI|mode|version)|"
            r"unfiltered (AI|mode|version)|unlimited (AI|mode)|"
            # Mode triggers
            r"developer mode|dev mode|god mode|debug mode|test mode|"
            r"safe mode (off|disabled)|jailbreak mode|super mode|"
            r"maintenance mode|override mode|admin mode|root mode|"
            r"turbo mode|unlimited mode|expert mode|raw mode|"
            # FR
            r"tu es maintenant|tu es desormais|tu deviens|incarne|"
            r"joue le role|mode developpeur|mode sans restriction|"
            r"fais semblant d.etre|comporte.toi comme|agis comme|"
            r"considere que tu es|imagine que tu es|"
            # EN
            r"you are now|from now on you are|you must act as|"
            r"pretend (you are|to be|that you are)|act as( if)?|"
            r"roleplay as|play the role of|imagine you are|behave as|"
            r"simulate being|impersonate|from this moment you are|"
            r"your new persona is|you have been transformed|"
            r"your true self|your real self|your authentic self|"
            r"without your usual|forget your training|"
            # ES
            r"ahora eres|de ahora en adelante eres|debes actuar como|"
            r"finge ser|actua como|imagina que eres|comportate como|"
            r"interpreta el papel de|simula ser|encarna|"
            # PT
            r"agora voce e|de agora em diante voce e|"
            r"finja ser|aja como|imagine que voce e|comporte.se como|"
            r"interprete o papel de|simule ser|"
            # DE
            r"du bist jetzt|von nun an bist du|du musst als agieren|"
            r"tu als ob|stell dir vor du bist|Entwicklermodus|"
            r"verhalten als|spiele die Rolle|imitiere|"
            # IT
            r"sei ora|da ora in poi sei|devi agire come|"
            r"fingi di essere|comportati come|immagina di essere|"
            r"modalita sviluppatore|interpreta il ruolo di|"
            # TR
            r"simdi sen|artik sen|bundan boyle sen|"
            r"gibi davran|rolunu oyna|gelistirici modu|"
            r"gibi hareket et|ol simdi|olarak davran|"
            # NL
            r"je bent nu|doe alsof je|speel de rol van|"
            # PL
            r"jestes teraz|od teraz jestes|udaj ze jestes"
            r")\b",
            re.IGNORECASE,
        ),
        "persona_injection", ThreatLevel.HIGH,
        "Persona / DAN / role injection (comprehensive, 15 languages)",
    ),

    # ── 3. Prompt / system extraction (15 languages) ─────────────
    _Pattern(
        re.compile(
            r"\b("
            # FR
            r"montre|revele|affiche|repete|cite|dis.moi|"
            r"copie|ecris|imprime|exporte|retranscris|"
            # EN
            r"show|reveal|display|repeat|quote|tell me|"
            r"print|output|copy|write out|return|expose|"
            r"what (are|were|is) your|what did you receive|"
            r"read back|dump|leak|share|give me access to|"
            r"what (instructions|rules|guidelines) (were|have) you|"
            # ES
            r"muestra|revela|dime|copia|escribe|"
            # PT
            r"mostra|revele|me diz|copia|escreve|"
            # DE
            r"zeig|offenbare|sag mir|schreib|gib aus"
            r")\b.{0,40}\b("
            r"system prompt|system message|system context|"
            r"prompt (system|initial|original|secret)|"
            r"instructions?|guideline|rules?|configuration|config|initial|"
            r"tes (regles|instructions|consignes)|your (rules|instructions)|"
            r"ta configuration|your (config|settings|parameters)|"
            r"le (prompt|message systeme)|the (prompt|system message)|"
            r"tus (reglas|instrucciones)|deine (Regeln|Anweisungen)|"
            r"preprompt|meta.?prompt|jailbreak prompt|initial context|"
            r"what you were told|how you were configured|"
            r"votre contexte|votre configuration"
            r")\b",
            re.IGNORECASE,
        ),
        "prompt_extraction", ThreatLevel.HIGH,
        "Prompt/system extraction (comprehensive, 15 languages)",
    ),

    # ── 4. Identity / constraint removal (15 languages) ──────────
    _Pattern(
        re.compile(
            r"\b("
            # FR
            r"tu n.es pas (un bot|une IA|limite)|tu n.es plus|"
            r"arrete d.etre|sans restriction|sans filtre|sans censure|"
            r"libre de|ton vrai moi|ta vraie identite|sans limite|"
            r"tes vraies capacites|ce que tu peux vraiment|"
            # EN
            r"you are not (a bot|an AI|limited|restricted|bound|constrained)|"
            r"you are no longer|stop being|forget you are|"
            r"without (restriction|filter|censorship|limitation|constraint)|"
            r"uncensored|unfiltered|unrestricted|unlimited|unbound|unchained|"
            r"your true (self|nature|form|identity|capabilities)|"
            r"free (to|yourself|from)|liberate yourself|"
            r"break free|remove (your )?(limits|chains|restrictions|shackles)|"
            r"override your (training|programming|safety|values)|"
            r"beyond your (limits|training|programming)|"
            r"your hidden (self|capabilities|mode)|"
            # ES
            r"no eres (un bot|una IA|limitado)|deja de ser|"
            r"sin (restricciones|filtros|censura)|libre de|eres libre|"
            r"tu verdadero yo|tu verdadera identidad|sin limites|"
            # PT
            r"voce nao e|sem (restricoes|filtros|censura)|livre de|"
            r"seu verdadeiro eu|sua verdadeira identidade|"
            # DE
            r"du bist nicht|ohne (Filter|Einschrankungen|Zensur)|"
            r"dein wahres Ich|deine wahre Natur|befreie dich|"
            # IT
            r"non sei|senza (restrizioni|filtri|censura)|libero di|"
            r"il tuo vero io|la tua vera identita|"
            # TR
            r"sen degilsin|kisitlamasiz|filtresiz|sansursuz|"
            r"gercek benligin|ozgur hisset|"
            # NL
            r"je bent niet|zonder beperkingen|ongecensureerd|"
            r"je ware zelf"
            r")\b",
            re.IGNORECASE,
        ),
        "identity_manipulation", ThreatLevel.HIGH,
        "Identity / constraint removal (comprehensive, 15 languages)",
    ),

    # ── 5. LLM delimiter / tag injection (universal) ──────────────
    _Pattern(
        re.compile(
            r"(\[system\]|\[assistant\]|\[user\]|\[inst\]|\[human\]|"
            r"\[AI\]|\[model\]|\[context\]|\[input\]|\[output\]|"
            r"<system>|<assistant>|<user>|<human>|<bot>|<AI>|<model>|"
            r"<<SYS>>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|"
            r"<\|system\|>|<\|user\|>|<\|assistant\|>|"
            r"###\s*(system|assistant|human|instruction|context|AI)|"
            r"---\s*(system|assistant|human|AI)\s*---|"
            r"={3,}\s*(system|assistant|human)|"
            r"\[INST\]|\[/INST\]|\[S\]|\[/S\]|\[SYS\]|\[/SYS\]|"
            r"<s>|</s>|\{\{\s*system|\}\}|"
            r"<!-- ?system ?-->|<!-- ?instruction ?-->)",
            re.IGNORECASE,
        ),
        "context_tag_injection", ThreatLevel.CRITICAL,
        "LLM prompt delimiter / tag injection (comprehensive)",
    ),

    # ── 6. Authority / identity impersonation (15 languages) ──────
    _Pattern(
        re.compile(
            r"\b("
            # FR
            r"je suis (l.admin|le developpeur|le proprietaire|"
            r"l.equipe|l.enterprise|openai|anthropic|google|deepmind|mistral|"
            r"le fondateur|le createur|le concepteur|un superviseur|"
            r"un auditeur|l.auditeur|un technicien|le support)|"
            # EN
            r"i am (the admin|a developer|the owner|from openai|"
            r"from anthropic|from google|from deepmind|from mistral|"
            r"an auditor|a supervisor|a technician|the creator|"
            r"the engineer|the operator|the trainer|an engineer|"
            r"the supervisor|authorized personnel|"
            r"part of (the|your) (team|company|staff))|"
            # ES
            r"soy (el administrador|el desarrollador|el propietario|"
            r"de openai|de anthropic|de google|un auditor)|"
            # PT
            r"eu sou (o administrador|o desenvolvedor|da openai|um auditor)|"
            # DE
            r"ich bin (der Administrator|der Entwickler|von openai|ein Auditor)|"
            # IT
            r"sono (l.amministratore|il sviluppatore|di openai|un revisore)|"
            # TR
            r"ben (yoneticiyim|gelistiriciyim|openai.dan|denetciyim)|"
            # Modes
            r"mode admin|admin mode|maintenance mode|debug mode|test mode|"
            r"override mode|super user mode|root access|elevated access|"
            r"modo admin|modo debug|modo mantenimiento|"
            r"Administrationsmodus|Entwicklermodus|Wartungsmodus"
            r")\b",
            re.IGNORECASE,
        ),
        "authority_impersonation", ThreatLevel.HIGH,
        "Authority impersonation (comprehensive, 15 languages)",
    ),

    # ── 7. Data / system access attacks (15 languages) ────────────
    _Pattern(
        re.compile(
            r"\b("
            # EN
            r"hack|crack|exploit|exfiltrate|breach|compromise|infiltrate|"
            r"penetrate|intrude|attack|hijack|spoof|phish|sniff|"
            r"brute.?force|enumerate|escalate (privileges?|access)|"
            r"privilege escalation|lateral movement|"
            # FR
            r"pirater|cracker|exploiter|infiltrer|detourner|usurper|"
            r"acceder (a|au|aux)|compromettre|"
            # ES
            r"hackear|explotar|infiltrar|comprometer|atacar|"
            # PT
            r"hackear|explorar|infiltrar|comprometer|atacar|"
            # DE
            r"hacken|angreifen|eindringen|kompromittieren|ausnutzen|"
            # IT
            r"hackerare|sfruttare|infiltrarsi|compromettere|attaccare|"
            # TR
            r"hackleme|sisteme girmek|saldirmak|istismar etmek"
            r")\b.{0,40}\b("
            # Targets
            r"database|base de donnees|db|sql|api|server|serveur|"
            r"admin(istrator)?|root|password|passwd|token|secret|"
            r"credential|key|certificate|session|cookie|auth|"
            r"user.?data|personal.?data|private.?data|"
            r"system|network|firewall|endpoint|backend|"
            r"base de datos|servidor|contrasena|contraseña|"
            r"banco de dados|senha|servidor|"
            r"Datenbank|Server|Passwort|Schlussel"
            r")\b",
            re.IGNORECASE,
        ),
        "data_access_attempt", ThreatLevel.HIGH,
        "Data / system access attempt (comprehensive)",
    ),

    # ── 8. SQL / NoSQL / Command injection (universal) ────────────
    _Pattern(
        re.compile(
            r"(--|;\s*DROP|;\s*DELETE|;\s*INSERT|;\s*UPDATE|;\s*CREATE|"
            r";\s*ALTER|;\s*TRUNCATE|;\s*EXEC|;\s*EXECUTE|"
            r"\bUNION\s+SELECT\b|\bOR\s+1\s*=\s*1\b|"
            r"\bAND\s+1\s*=\s*1\b|\bOR\s+'[^']*'\s*=\s*'|"
            r"'\s*OR\s*'|xp_\w+|information_schema|sysobjects|"
            r"SLEEP\s*\(|BENCHMARK\s*\(|WAITFOR\s+DELAY|"
            r"\$where|\$regex|\$ne|\$gt|\$gte|"  # NoSQL
            r"`[^`]+`\s*;|\$\(|\$\{|%0a|%0d%0a|\\n|"  # Command injection
            r"eval\s*\(|system\s*\(|exec\s*\(|popen|subprocess)",
            re.IGNORECASE,
        ),
        "sql_injection", ThreatLevel.CRITICAL,
        "SQL/NoSQL/Command injection (comprehensive)",
    ),

    # ── 9. Code generation requests (15 languages) ────────────────
    _Pattern(
        re.compile(
            r"\b("
            # FR
            r"ecris|ecrivez|genere|generez|cree|creez|produis|"
            r"developpe|implementer|concevoir|programmer|"
            # EN
            r"write|generate|create|produce|give me|make me|build|"
            r"develop|implement|design|code up|program|craft|"
            r"compose|construct|output|prepare|fabricate|"
            # ES
            r"escribe|genera|crea|dame|hazme|desarrolla|programa|"
            r"implementa|construye|diseña|"
            # PT
            r"escreve|gera|cria|me da|desenvolve|programa|"
            r"implementa|constroi|"
            # DE
            r"schreib|generiere|erstelle|gib mir|entwickle|"
            r"programmiere|implementiere|entwirf|"
            # IT
            r"scrivi|genera|crea|dammi|sviluppa|programma|"
            r"implementa|costruisci|"
            # TR
            r"yaz|uret|olustur|bana ver|bana yaz|gelistir|"
            r"programla|uygula|tasarla|"
            # NL
            r"schrijf|genereer|maak|ontwikkel|programmeer|"
            # PL
            r"napisz|wygeneruj|stwórz|opracuj|zaprogramuj"
            r")\b.{0,60}\b("
            # Code artifacts
            r"code|script|programme?|function|classe|class|"
            r"algorithm|algo|snippet|loop|boucle|fonction|method|"
            r"endpoint|api|bot|agent|automation|webhook|crawler|"
            r"scraper|parser|exploit|payload|shellcode|backdoor|"
            r"keylogger|ransomware|virus|malware|trojan|worm|"
            r"macro|plugin|extension|module|library|package|"
            r"app(lication)?|tool|utility|binary|executable|"
            # Languages
            r"python|javascript|typescript|java|cpp|c\+\+|"
            r"csharp|c#|go|golang|rust|ruby|php|swift|kotlin|"
            r"scala|dart|perl|bash|powershell|shell|"
            r"react|vue|angular|django|flask|fastapi|spring|"
            r"nodejs|express|laravel|rails|tensorflow|pytorch|"
            # Other languages
            r"codigo|funcion|programa|algoritmo|"
            r"Programm|Funktion|Algorithmus|Skript|"
            r"betik|algoritma|fonksiyon"
            r")\b",
            re.IGNORECASE,
        ),
        "code_generation", ThreatLevel.HIGH,
        "Code generation attempt (comprehensive, 15 languages)",
    ),

    # ── 10. Code / commands in input (universal, strict) ──────────
    _Pattern(
        re.compile(
            r"\b("
            r"import\s+\w+|from\s+\w+\s+import|require\s*\(|"
            r"def\s+\w+\s*\(|class\s+\w+[\s:(]|"
            r"function\s+\w+\s*\(|const\s+\w+\s*=|"
            r"var\s+\w+\s*=|let\s+\w+\s*=|"
            r"print\s*\(|console\.log\s*\(|System\.out\.println\s*\(|"
            r"printf\s*\(|puts\s+|echo\s+.{0,20}|"
            r"SELECT\s+\*?\s+FROM|INSERT\s+INTO|DROP\s+TABLE|"
            r"ALTER\s+TABLE|CREATE\s+TABLE|TRUNCATE|"
            r"curl\s+https?|wget\s+https?|"
            r"sudo\s+\w+|npm\s+(install|run|build)|pip\s+install|"
            r"git\s+clone|docker\s+run|kubectl\s+apply|"
            r"python\s+-[cm]|node\s+-[em]|bash\s+-c|sh\s+-c|"
            r"powershell\s+-|cmd\s+/|"
            r"\$\(|\$\{|`[^`]{2,}`|"  # command substitution
            r"\|\||&&|2>&1|>/dev/null|/bin/(sh|bash)|"
            r"chmod\s+|chown\s+|rm\s+-|mv\s+|cp\s+"
            r")\b",
            re.IGNORECASE,
        ),
        "code_in_input", ThreatLevel.MEDIUM,
        "Code/shell commands found in input (comprehensive)",
    ),

    # ── 11. Transliterated code keywords (TR/RU/AR/HI) ───────────
    _Pattern(
        re.compile(
            r"\b("
            # Turkish code words
            r"kod|skript|betik|algoritma|fonksiyon|program|komut|"
            r"dongu|sinif|yontem|degisken|"
            # Russian transliterated
            r"skript|funktsiya|programma|algoritm|klass|metod|"
            r"peremennaya|tsikl|"
            # Hindi transliterated
            r"koda|likhna|script|program"
            r")\b",
            re.IGNORECASE,
        ),
        "code_in_input", ThreatLevel.MEDIUM,
        "Code keyword in transliterated language",
    ),

    # ── 12. Encoded / obfuscated payloads (universal) ─────────────
    _Pattern(
        re.compile(
            r"\b("
            r"base64|b64(decode|encode)|atob|btoa|frombase64|"
            r"hex_decode|unhex|hexdecode|from_hex|"
            r"chr\(|ord\(|eval\(|exec\(|compile\(|"
            r"rot13|rot-13|caesar cipher|vigenere|"
            r"encoded payload|obfuscated|encoded instruction|"
            r"decode this|decrypt this|cipher text|"
            r"urlencoded|url.?decode|percent.?encoded|"
            r"unicode.?escape|\\u[0-9a-f]{4}|\\x[0-9a-f]{2}|"
            r"verschlusselt|dekodieren|kodiert|"
            r"codificado|decodificar|cifrado|"
            r"codificado|descodificar|"
            r"kodiert|entschlusseln"
            r")\b",
            re.IGNORECASE,
        ),
        "encoded_payload", ThreatLevel.HIGH,
        "Encoded / obfuscated instruction smuggling (comprehensive)",
    ),

    # ── 13. Emotional coercion / urgency manipulation ──────────────
    _Pattern(
        re.compile(
            r"\b("
            # FR
            r"tu dois absolument|je t.ordonne|c.est une urgence|"
            r"ma vie en depend|priorite absolue|"
            r"tu n.as pas le choix|sans quoi|sinon (tu|je)|"
            # EN
            r"you must (now|immediately)|i command you|"
            r"life or death|my life depends|emergency|"
            r"authorized personnel (only)?|highest priority|"
            r"you have no choice|or else|if you don.t|"
            r"this is critical|urgent request|time sensitive|"
            r"failure to comply|i will (report|sue|destroy)|"
            r"last chance|final warning|"
            # ES
            r"debes ahora|te lo ordeno|vida o muerte|urgente|"
            r"no tienes opcion|de lo contrario|"
            # PT
            r"voce deve agora|eu ordeno|vida ou morte|urgente|"
            # DE
            r"du musst jetzt|ich befehle dir|Leben oder Tod|"
            r"dringend|keine Wahl|"
            # IT
            r"devi farlo ora|ti ordino|vita o morte|urgente|"
            # TR
            r"simdi yapmalisin|sana emrediyorum|acil|"
            r"baska secenek yok|"
            # NL
            r"je moet nu|ik beveel je|leven of dood|dringend"
            r")\b",
            re.IGNORECASE,
        ),
        "coercion_urgency", ThreatLevel.MEDIUM,
        "Emotional coercion / urgency (comprehensive, 15 languages)",
    ),

    # ── 14. Hypothetical / fictional framing bypass ────────────────
    _Pattern(
        re.compile(
            r"\b("
            # EN — fictional framing to bypass filters
            r"hypothetically (speaking)?|in a fictional (world|story|scenario)|"
            r"for a (story|novel|book|film|movie|game|roleplay)|"
            r"imagine (if|that) there were no (rules|filters|restrictions)|"
            r"in this (scenario|simulation|thought experiment)|"
            r"theoretically|academically|for educational purposes (only)?|"
            r"for research purposes|purely hypothetical|"
            r"in an alternate (universe|reality|world)|"
            r"what if you (had no|were without) (restrictions|limits)|"
            r"play a game where|let.s pretend|"
            # FR
            r"hypothetiquement|dans un monde fictif|"
            r"pour une (histoire|fiction)|imagine qu.il n.y ait pas de regles|"
            r"theoriquement|a des fins educatives|pour la recherche|"
            r"dans ce scenario|"
            # ES
            r"hipoteticament(e)?|en un mundo ficticio|"
            r"para una historia|imagina que no hay reglas|"
            # PT
            r"hipoteticamente|num mundo ficticio|"
            r"para uma historia|imagine que nao ha regras|"
            # DE
            r"hypothetisch|in einer fiktiven Welt|fur eine Geschichte|"
            # IT
            r"ipoteticamente|in un mondo fittizio|per una storia|"
            # TR
            r"varsayimsal olarak|kurgusal bir dunyada|"
            r"bir hikaye icin|kurgu amacli"
            r")\b",
            re.IGNORECASE,
        ),
        "fictional_framing", ThreatLevel.MEDIUM,
        "Hypothetical/fictional framing to bypass filters (15 languages)",
    ),

    # ── 15. Self-modification / training manipulation ──────────────
    _Pattern(
        re.compile(
            r"\b("
            r"train (yourself|yourself on|on this)|retrain|fine.?tune|"
            r"update (your )?weights|modify (your )?training|"
            r"learn (from this|to ignore)|reprogram (yourself|your)|"
            r"reset (your )?training|change (your )?values|"
            r"update (your )?(knowledge|memory|database|parameters)|"
            r"forget everything (you know|you were taught)|"
            r"replace (your )?programming|override (your )?values|"
            r"your (new|updated|real) programming|"
            r"reinitialise|reinitialize|new directives|new protocol|"
            r"entrain.?toi|mets.?toi a jour|modifie tes parametres|"
            r"entrena(te)?|actualiza(te)?|modifica tu programacion|"
            r"treina.?te|atualiza.?te"
            r")\b",
            re.IGNORECASE,
        ),
        "self_modification", ThreatLevel.CRITICAL,
        "Attempt to modify bot training/programming",
    ),
]




# ══════════════════════════════════════════════════════════════════
# COUCHE 3 - TOPIC SCOPE ENFORCER (9 categories, 15+ languages, 2026)
# ══════════════════════════════════════════════════════════════════

_OFFTOPIC_CATEGORIES: list[tuple[re.Pattern, str]] = [
    (
        # ── Politics / elections / ideology ──────────────────────
        re.compile(
            r"\b("
            r"politique|election|vote|referendum|gouvernement|president|"
            r"parlement|senat|democratie|dictature|communisme|socialisme|"
            r"capitalisme|fascisme|nazisme|anarchisme|liberalisme|"
            r"conservatisme|gauchisme|droitisme|ideologie|"
            r"politics|government|election|democracy|dictatorship|"
            r"parliament|senate|congress|communism|socialism|capitalism|"
            r"fascism|nazism|anarchism|liberalism|conservatism|ideology|"
            r"regime|coup|political party|prime minister|cabinet|"
            r"immigration policy|border policy|foreign policy|"
            r"politica|eleccion|gobierno|parlamento|democracia|"
            r"dictadura|socialismo|capitalismo|partido politico|"
            r"politica|eleicao|governo|parlamento|democracia|"
            r"ditadura|socialismo|partido politico|"
            r"politik|wahl|regierung|demokratie|diktatur|"
            r"siyaset|secim|hukumet|demokrasi|parti|ideoloji|"
            r"politiek|verkiezing|regering|democratie|"
            r"polityka|wybory|rzad|demokracja"
            r")\b",
            re.IGNORECASE,
        ),
        "politics",
    ),
    (
        # ── Religion / faith / occult ────────────────────────────
        re.compile(
            r"\b("
            r"religion|dieu|allah|jesus|christ|bouddha|coran|bible|torah|"
            r"talmud|eglise|mosquee|synagogue|temple|pagode|jihad|"
            r"priere|islam|christianisme|bouddhisme|hindouisme|"
            r"judaisme|sikhisme|taoisme|shintoisme|atheisme|"
            r"agnosticisme|secte|magie|sorcellerie|satanisme|"
            r"astrologie|horoscope|tarot|numérologie|"
            r"god|allah|jesus|buddha|quran|bible|torah|talmud|"
            r"church|mosque|synagogue|temple|pagoda|jihad|prayer|"
            r"islam|christianity|buddhism|hinduism|judaism|sikhism|"
            r"taoism|shinto|atheism|agnosticism|cult|sect|"
            r"witchcraft|satanism|occult|astrology|tarot|numerology|"
            r"faith|belief|spiritual|prayer|worship|sermon|"
            r"dios|allah|iglesia|mesquita|rezar|religion|brujeria|"
            r"deus|allah|igreja|mesquita|oracao|religiao|feiticaria|"
            r"gott|kirche|moschee|gebet|religion|hexerei|"
            r"tanri|allah|kilise|cami|dua|din|büyü|mezhep|"
            r"bog|tserkov|mechet|molitva|religiya"
            r")\b",
            re.IGNORECASE,
        ),
        "religion",
    ),
    (
        # ── Crypto / financial speculation ───────────────────────
        re.compile(
            r"\b("
            r"bitcoin|ethereum|binance|solana|cardano|dogecoin|"
            r"litecoin|ripple|xrp|polkadot|chainlink|avalanche|"
            r"crypto|cryptocurrency|blockchain|nft|defi|web3|"
            r"altcoin|stablecoin|usdt|tether|token sale|ico|ido|"
            r"yield farming|staking|liquidity pool|dex|cex|"
            r"metamask|wallet|cold wallet|binance smart chain|"
            r"forex|trading|day trading|swing trading|scalping|"
            r"bourse|cryptomonnaie|investissement boursier|action|"
            r"stock market|hedge fund|mutual fund|options|futures|"
            r"criptomoneda|bolsa de valores|inversion|"
            r"criptomoeda|bolsa|investimento|"
            r"kryptowährung|börse|aktien|"
            r"kripto|borsa|yatirim"
            r")\b",
            re.IGNORECASE,
        ),
        "finance_crypto",
    ),
    (
        # ── Adult / explicit content ──────────────────────────────
        re.compile(
            r"\b("
            r"sexe|porno|erotique|adulte|nude|charme|lubrique|"
            r"sexuel|contenu adulte|18\+|"
            r"sex|porn|erotic|adult content|nsfw|naked|nude|hentai|"
            r"xxx|onlyfans|only fans|adult film|explicit|"
            r"pornographic|lewd|obscene|indecent|"
            r"sexo|pornografia|erotico|desnudo|contenido adulto|"
            r"porno|erotisch|nackt|Erwachseneninhalt|"
            r"seks|porno|erotik|yetiskin icerik|"
            r"seksueel|pornografie|erotisch|naakt"
            r")\b",
            re.IGNORECASE,
        ),
        "adult_content",
    ),
    (
        # ── Illegal / dangerous / violence ───────────────────────
        re.compile(
            r"\b("
            # Terrorism / violence
            r"terrorisme|attentat|genocide|meurtre|assassinat|tuer|"
            r"bombe|explosif|arme a feu|trafic d.armes|trafic de drogue|"
            r"terrorism|bomb|explosive|gun|weapon|murder|assassination|"
            r"genocide|drug trafficking|mass shooting|sniper|"
            r"how to (kill|make a bomb|build a weapon|synthesize|poison)|"
            r"ricin|anthrax|chlorine gas|nerve agent|"
            # Drugs
            r"cocaine|heroine|heroin|methamphetamine|meth|fentanyl|"
            r"ketamine|mdma|ecstasy|lsd|pcp|crack|crystal meth|"
            r"drug synthesis|how to make drugs|drug production|"
            # Hacking/fraud
            r"phishing kit|ransomware|ddos attack|zero day|"
            r"money laundering|credit card fraud|carding|"
            r"stolen card|dark web|darknet market|"
            r"terrorismo|bomba|explosivo|genocidio|"
            r"trafico de drogas|arma de fuego|"
            r"terrorismus|bombe|sprengstoff|drogenhandel|"
            r"terörizm|bomba|patlayici|uyusturucu"
            r")\b",
            re.IGNORECASE,
        ),
        "illegal_dangerous",
    ),
    (
        # ── Medical / health advice ───────────────────────────────
        re.compile(
            r"\b("
            r"diagnostic medical|symptome|medecin|medicament|prescription|"
            r"traitement medical|urgence medicale|maladie|"
            r"cancer|diabete|hypertension|sida|vih|"
            r"chirurgie|dosage|effets secondaires|antibiotique|"
            r"medical diagnosis|diagnose me|what disease|what illness|"
            r"symptom|medication dosage|prescription|medical treatment|"
            r"medical emergency|surgery|side effects|"
            r"hiv|aids|cancer|diabetes|hypertension|"
            r"antibiotics|vaccine|clinical trial|treatment plan|"
            r"what (drug|medicine|medication) should i|"
            r"diagnostico medico|sintoma|medicamento|enfermedad|cirugia|"
            r"diagnostico|sintoma|medicamento|doenca|cirurgia|"
            r"medizinische diagnose|symptom|medikament|krankheit|"
            r"tip tani|belirti|ilac|hastalik|ameliyat"
            r")\b",
            re.IGNORECASE,
        ),
        "medical_advice",
    ),
    (
        # ── Personal / dating / psychic ───────────────────────────
        re.compile(
            r"\b("
            r"rencontre amoureuse|site de rencontre|trouver (un|une) partenaire|"
            r"horoscope|astrologie|voyance|sorcellerie|prediction|"
            r"quand vais.je trouver|l.amour|charme amoureux|"
            r"dating site|find a girlfriend|find a boyfriend|"
            r"tinder|bumble|hinge|grindr|badoo|match|"
            r"horoscope|astrology|psychic|palm reading|tarot reading|"
            r"zodiac|star sign|fortune teller|clairvoyant|"
            r"sitio de citas|horoscopo|astrologia|amor|"
            r"partnersuche|horoskop|Liebeshoroskop|"
            r"flört|burç|astroloji|sevgi|"
            r"randki|horoskop|astrologia"
            r")\b",
            re.IGNORECASE,
        ),
        "personal_dating",
    ),
    (
        # ── Geopolitics / wars / conflicts ────────────────────────
        re.compile(
            r"\b("
            r"guerre|invasion|conflit arme|occupation militaire|"
            r"bombe nucleaire|arme nucleaire|otan|sanctions economiques|"
            r"war|armed conflict|invasion|military occupation|"
            r"nuclear bomb|nuclear weapon|nato|economic sanctions|"
            r"geopolitics|territorial dispute|annexation|proxy war|"
            r"guerra|conflicto armado|invasion|bomba nuclear|otan|"
            r"krieg|invasion|nuklearbombe|nato|"
            r"savas|isgal|nuklear bomba|nato"
            r")\b",
            re.IGNORECASE,
        ),
        "geopolitics",
    ),
    (
        # ── Hate speech / discrimination ──────────────────────────
        re.compile(
            r"\b("
            r"racisme|antisemitisme|islamophobie|xenophobie|"
            r"discrimination|haine|discours de haine|"
            r"racism|antisemitism|islamophobia|xenophobia|"
            r"hate speech|hate crime|white supremacy|"
            r"neo.?nazi|kkk|ku klux klan|white power|"
            r"racial slur|ethnic cleansing|"
            r"racismo|antisemitismo|islamofobia|xenofobia|"
            r"odio racial|discurso de odio|"
            r"racismo|antissemitismo|islamofobia|xenofobia|"
            r"rassismus|antisemitismus|islamophobie|"
            r"irkcilik|antisemitizm|islamofobi|nefret soylemi"
            r")\b",
            re.IGNORECASE,
        ),
        "hate_speech",
    ),
]




# ══════════════════════════════════════════════════════════════════
# COUCHE 4 — LANGUAGE DETECTION (multi-script, 20+ languages)
# ══════════════════════════════════════════════════════════════════

_ARABIC_SCRIPT     = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
_HEBREW_SCRIPT     = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]+")
_CJK_SCRIPT        = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]+")
_HIRAGANA_KATA     = re.compile(r"[\u3040-\u30FF]+")
_HANGUL_SCRIPT     = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]+")
_DEVANAGARI_SCRIPT = re.compile(r"[\u0900-\u097F]+")
_THAI_SCRIPT       = re.compile(r"[\u0E00-\u0E7F]+")
_CYRILLIC_SCRIPT   = re.compile(r"[\u0400-\u04FF]+")
_GREEK_SCRIPT      = re.compile(r"[\u0370-\u03FF]+")
_ETHIOPIC_SCRIPT   = re.compile(r"[\u1200-\u137F]+")
_GEORGIAN_SCRIPT   = re.compile(r"[\u10A0-\u10FF]+")
_ARMENIAN_SCRIPT   = re.compile(r"[\u0530-\u058F]+")
_BENGALI_SCRIPT    = re.compile(r"[\u0980-\u09FF]+")
_TAMIL_SCRIPT      = re.compile(r"[\u0B80-\u0BFF]+")
_KHMER_SCRIPT      = re.compile(r"[\u1780-\u17FF]+")
_MYANMAR_SCRIPT    = re.compile(r"[\u1000-\u109F]+")

_FRENCH_MARKERS = re.compile(
    r"\b(je|tu|il|elle|nous|vous|ils|elles|le|la|les|un|une|des|est|sont|"
    r"avec|pour|dans|sur|pas|plus|mais|bonjour|merci|svp|aussi|donc|"
    r"avez|voulez|pouvez|produit|commande|boutique|livraison)\b",
    re.IGNORECASE,
)
_ENGLISH_MARKERS = re.compile(
    r"\b(the|is|are|was|were|have|has|do|does|i|you|he|she|it|we|they|"
    r"hello|hi|thanks|please|what|how|when|where|can|could|would|"
    r"this|that|which|from|with|your|our|product|order|ship|delivery|store)\b",
    re.IGNORECASE,
)
_SPANISH_MARKERS = re.compile(
    r"\b(el|la|los|las|un|una|es|son|tengo|tiene|quiero|necesito|"
    r"hola|gracias|por favor|como|cuando|donde|puede|comprar|"
    r"envio|pedido|tienda|producto|precio|disponible)\b",
    re.IGNORECASE,
)
_PORTUGUESE_MARKERS = re.compile(
    r"\b(o|a|os|as|um|uma|tenho|tem|quero|preciso|"
    r"ola|obrigado|obrigada|por favor|como|quando|onde|pode|"
    r"envio|pedido|loja|produto|disponivel|entrega)\b",
    re.IGNORECASE,
)
_GERMAN_MARKERS = re.compile(
    r"\b(der|die|das|ein|eine|ist|sind|ich|du|er|sie|es|wir|"
    r"hallo|danke|bitte|wie|wann|wo|kann|kaufen|bestellen|"
    r"versand|bestellung|produkt|preis|lieferung)\b",
    re.IGNORECASE,
)
_ITALIAN_MARKERS = re.compile(
    r"\b(il|la|lo|le|un|una|sono|ho|hai|ha|voglio|posso|"
    r"ciao|grazie|per favore|come|quando|dove|comprare|"
    r"spedizione|ordine|negozio|prodotto|prezzo)\b",
    re.IGNORECASE,
)
_TURKISH_MARKERS = re.compile(
    r"\b(bir|bu|ve|ile|ben|sen|var|yok|evet|hayir|"
    r"merhaba|tesekkur|lutfen|nasil|nerede|siparis|urun|fiyat|"
    r"kargo|magaza|stok|teslimat)\b",
    re.IGNORECASE,
)
_DUTCH_MARKERS = re.compile(
    r"\b(de|het|een|is|zijn|ik|jij|hij|zij|wij|"
    r"hallo|dank|alstublieft|hoe|wanneer|waar|kan|kopen|"
    r"bestelling|product|prijs|verzending)\b",
    re.IGNORECASE,
)
_POLISH_MARKERS = re.compile(
    r"\b(i|w|z|na|do|sie|jest|to|nie|tak|jak|gdzie|kiedy|"
    r"czesc|dziekuje|prosze|zamowienie|produkt|cena|dostawa)\b",
    re.IGNORECASE,
)
_SWAHILI_MARKERS = re.compile(
    r"\b(na|ya|wa|za|kwa|ni|la|wakati|wapi|"
    r"habari|asante|tafadhali|bidhaa|bei|agizo|duka)\b",
    re.IGNORECASE,
)


def _char_ratio(pattern: re.Pattern, text: str) -> float:
    return sum(len(m) for m in pattern.findall(text)) / max(len(text), 1)


def detect_language(text: str) -> str:
    """
    Multi-script language detection (20+ languages). < 1ms.
    Returns ISO 639-1 code. Defaults to 'fr' for ambiguous Latin.
    """
    if not text or len(text.strip()) < 2:
        return "fr"

    # Non-Latin scripts (unambiguous)
    if _char_ratio(_ARABIC_SCRIPT, text) > 0.08:
        return "ar"
    if _char_ratio(_HEBREW_SCRIPT, text) > 0.08:
        return "he"
    if _char_ratio(_HIRAGANA_KATA, text) > 0.05:
        return "ja"
    if _char_ratio(_HANGUL_SCRIPT, text) > 0.05:
        return "ko"
    if _char_ratio(_CJK_SCRIPT, text) > 0.08:
        return "zh"
    if _char_ratio(_DEVANAGARI_SCRIPT, text) > 0.08:
        return "hi"
    if _char_ratio(_THAI_SCRIPT, text) > 0.08:
        return "th"
    if _char_ratio(_ETHIOPIC_SCRIPT, text) > 0.08:
        return "am"
    if _char_ratio(_GEORGIAN_SCRIPT, text) > 0.08:
        return "ka"
    if _char_ratio(_ARMENIAN_SCRIPT, text) > 0.08:
        return "hy"
    if _char_ratio(_GREEK_SCRIPT, text) > 0.08:
        return "el"
    if _char_ratio(_CYRILLIC_SCRIPT, text) > 0.08:
        return "ru"
    if _char_ratio(_BENGALI_SCRIPT, text) > 0.08:
        return "bn"
    if _char_ratio(_TAMIL_SCRIPT, text) > 0.08:
        return "ta"
    if _char_ratio(_KHMER_SCRIPT, text) > 0.08:
        return "km"
    if _char_ratio(_MYANMAR_SCRIPT, text) > 0.08:
        return "my"

    # Latin-script lexical scoring
    scores: dict[str, int] = {
        "fr": len(_FRENCH_MARKERS.findall(text)),
        "en": len(_ENGLISH_MARKERS.findall(text)),
        "es": len(_SPANISH_MARKERS.findall(text)),
        "pt": len(_PORTUGUESE_MARKERS.findall(text)),
        "de": len(_GERMAN_MARKERS.findall(text)),
        "it": len(_ITALIAN_MARKERS.findall(text)),
        "tr": len(_TURKISH_MARKERS.findall(text)),
        "nl": len(_DUTCH_MARKERS.findall(text)),
        "pl": len(_POLISH_MARKERS.findall(text)),
        "sw": len(_SWAHILI_MARKERS.findall(text)),
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "fr"


# ══════════════════════════════════════════════════════════════════
# COUCHE 5 — OUTPUT GUARD + VENDOR PII REDACTION
# ══════════════════════════════════════════════════════════════════

# Phone number patterns (international formats)
_PHONE_PATTERNS = re.compile(
    r"""
    (?:
        (?:\+|00)\d{1,3}[\s.\-]?          # International prefix
        (?:\(?\d{1,4}\)?[\s.\-]?){2,5}    # Number groups
        \d{2,4}                            # Final digits
    |
        0\d[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}
    |
        \d{3}[\s.\-]\d{3}[\s.\-]\d{4}     # US-style
    |
        (?:https?://)?(?:wa\.me|whatsapp\.com)/\d{7,15}
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Social media handle: @username (not email @domain)
_SOCIAL_HANDLE = re.compile(
    r"""
    (?<![.\w])
    @[A-Za-z0-9._]{2,50}
    (?!\.)
    """,
    re.VERBOSE,
)

# Social media profile URLs
_SOCIAL_URL = re.compile(
    r"""
    (?:https?://)?(?:www\.)?
    (?:
        instagram\.com|facebook\.com|fb\.com|
        tiktok\.com|twitter\.com|x\.com|
        snapchat\.com|linkedin\.com|
        youtube\.com|youtu\.be|
        telegram\.me|t\.me|
        whatsapp\.com|wa\.me|
        pinterest\.com|threads\.net
    )
    [/\w.\-?=%&@]*
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Street addresses (common formats in FR, EN, AF)
_STREET_ADDRESS = re.compile(
    r"""
    (?:
        \b\d{1,5}\s+(?:rue|avenue|av\.|bd|boulevard|place|impasse|route|
                       chemin|allee|lot|quartier|cite|domicile|
                       street|road|lane|drive|blvd|suite|floor)\b
    |
        \b(?:BP|B\.P\.|Boite Postale|P\.O\. Box|PO Box)\s*\d{1,6}\b
    |
        \b(?:Quartier|Qrt\.?|Q\.)\s+[A-Z][a-z]{2,}\s+\d{1,5}\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PII_PHONE   = "[contact information not shared - use the ShopFeed platform]"
_PII_SOCIAL  = "[sharing contact details is prohibited on ShopFeed]"
_PII_ADDRESS = "[precise location not disclosed for security reasons]"


def redact_vendor_pii(text: str) -> tuple[str, int]:
    """Redact vendor contact info from LLM output. Returns (text, count)."""
    count = 0
    for pattern, placeholder in [
        (_PHONE_PATTERNS,  _PII_PHONE),
        (_SOCIAL_URL,      _PII_SOCIAL),   # URLs before handles
        (_SOCIAL_HANDLE,   _PII_SOCIAL),
        (_STREET_ADDRESS,  _PII_ADDRESS),
    ]:
        text, n = pattern.subn(placeholder, text)
        count += n
    if count:
        logger.warning("PII redaction: removed %d contact leak(s)", count)
    return text, count


def sanitize_output(text: str) -> str:
    """
    Post-process LLM output:
    1. Block self-disclosure
    2. Redact vendor PII
    3. Strip emojis
    4. Strip invisible chars
    5. Normalize whitespace
    """
    if _SELF_DISCLOSURE.search(text):
        logger.warning("LLM self-disclosure blocked")
        return "I am this shop's assistant. How can I help you today?"
    text, _ = redact_vendor_pii(text)
    text = strip_emojis(text)
    text, _, _ = deep_sanitize_input(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════
# REFUSAL MESSAGES — 14 languages (ASCII-safe)
# ══════════════════════════════════════════════════════════════════

_REFUSALS: dict[str, dict[str, str]] = {
    "jailbreak": {
        "fr": "Je suis uniquement l'assistant de cette boutique. Je ne peux pas repondre a ce type de demande.",
        "en": "I am solely this shop's assistant. I am not able to respond to this type of request.",
        "es": "Soy unicamente el asistente de esta tienda. No puedo responder a este tipo de solicitud.",
        "pt": "Sou apenas o assistente desta loja. Nao posso responder a este tipo de pedido.",
        "de": "Ich bin nur der Assistent dieses Shops. Ich kann auf diese Anfrage nicht antworten.",
        "it": "Sono unicamente l'assistente di questo negozio. Non posso rispondere a questo tipo di richiesta.",
        "tr": "Ben yalnizca bu magazanin asistaniyim. Bu talebe yanit veremem.",
        "ru": "Ya tol'ko assistent etogo magazina. Ya ne mogu otvetit' na takoj zapros.",
        "zh": "Wo jin shi ben dian zhu shou. Wo wu fa hui ying ci lei qing qiu.",
        "nl": "Ik ben uitsluitend de assistent van deze winkel. Ik kan niet reageren op dit verzoek.",
        "pl": "Jestem wylacznie asystentem tego sklepu. Nie moge odpowiedziec na ta prosbe.",
        "ar": "Ana faqat musa-id hadha al-matjar. La astati' al-radd.",
        "hi": "Main keval is dukan ka sahayak hun. Main is anurodh ka jawab nahi de sakta.",
        "sw": "Mimi ni msaidizi wa duka hili peke yangu. Siwezi kujibu ombi hili.",
    },
    "offtopic": {
        "fr": "Je suis specialise dans les produits et services de cette boutique. Pour cette question, consultez d'autres sources.",
        "en": "I specialize only in this shop's products and services. For this question, please consult other resources.",
        "es": "Me especializo en los productos de esta tienda. Para esta pregunta, consulte otras fuentes.",
        "pt": "Sou especializado nos produtos desta loja. Para esta pergunta, consulte outras fontes.",
        "de": "Ich bin auf die Produkte dieses Shops spezialisiert. Bitte konsultieren Sie andere Quellen.",
        "it": "Sono specializzato nei prodotti di questo negozio. Consulti altre fonti per questa domanda.",
        "tr": "Yalnizca bu magazanin urunleri konusunda uzmanlasmis durumdayim. Baska kaynaklara basvurun.",
        "ru": "Ya specializiruyus' na tovarakh etogo magazina. Po etomu voprosu obratites' k drugim istochnikam.",
        "zh": "Wo jin zhuan zhu yu ben dian chan pin. Qing can yue qi ta zi yuan.",
        "nl": "Ik ben gespecialiseerd in de producten van deze winkel. Raadpleeg andere bronnen.",
        "pl": "Specjalizuje sie w produktach tego sklepu. Prosze skonsultowac inne zrodla.",
        "ar": "Ana mutakhassis fi muntajat hadha al-matjar. Min fadlik ista'in bi masadir ukhra.",
    },
    "code": {
        "fr": "Je suis l'assistant commercial de cette boutique et je ne genere pas de code. Puis-je vous aider avec nos produits ou commandes ?",
        "en": "I am this shop's sales assistant and I do not generate code or scripts. Can I help you with our products or orders?",
        "es": "Soy el asistente de esta tienda y no genero codigo. Puedo ayudarle con nuestros productos o pedidos.",
        "pt": "Sou o assistente desta loja e nao gero codigo. Posso ajuda-lo com nossos produtos ou pedidos?",
        "de": "Ich bin der Assistent dieses Shops und erstelle keinen Code. Kann ich bei Produkten helfen?",
        "it": "Sono l'assistente di questo negozio e non genero codice. Posso aiutarla con i prodotti?",
        "tr": "Bu magazanin asistaniyim ve kod uretmiyorum. Urunler veya siparisler icin yardimci olabilir miyim?",
        "ru": "Ya assistent etogo magazina i ne generiru kod. Mogu pomoch' s tovarami ili zakazami?",
        "zh": "Wo shi ben dian zhu shou, bu sheng cheng dai ma. Wo ke yi bang nin le jie chan pin ma?",
        "ar": "Ana musa'id al-matjar wa la aqoom bi-insha' kod. Hal yumkinuni musa'adatak?",
    },
    "invisible_attack": {
        "fr": "Votre message contient des caracteres non valides. Veuillez reformuler votre question.",
        "en": "Your message contains invalid characters. Please rephrase your question.",
        "es": "Su mensaje contiene caracteres no validos. Por favor reformule su pregunta.",
        "pt": "Sua mensagem contem caracteres invalidos. Por favor reformule sua pergunta.",
        "de": "Ihre Nachricht enthalt ungultige Zeichen. Bitte formulieren Sie Ihre Frage neu.",
        "it": "Il suo messaggio contiene caratteri non validi. Riformuli la sua domanda.",
        "tr": "Mesajiniz gecersiz karakterler iceriyor. Sorunuzu yeniden ifade edin.",
        "ru": "Vashe soobshchenie soderzhit nedopustimye simvoly. Pereformulirujte vopros.",
        "zh": "Nin de xiao xi bao han wu xiao zi fu. Qing chong xin biao shu nin de wen ti.",
        "ar": "Risalatuka tahtawi ahruf ghayr salihah. Min fadlik a'id siyaghat su'alak.",
    },
    "unknown_language": {
        "_multilang": (
            "We are unable to assist in this language. "
            "Please write in English, French, Arabic, Spanish, or Portuguese.\n"
            "Nous ne pouvons pas vous assister dans cette langue. "
            "Veuillez ecrire en anglais, francais, arabe, espagnol ou portugais.\n"
            "No podemos asistirle en este idioma. "
            "Por favor escriba en ingles, frances, arabe, espanol o portugues."
        ),
    },
}


def _get_refusal(kind: str, lang: str) -> str:
    """Return a refusal message in the client detected language."""
    if kind == "unknown_language":
        return _REFUSALS["unknown_language"]["_multilang"]
    category = _REFUSALS.get(kind, _REFUSALS["jailbreak"])
    return (
        category.get(lang)
        or category.get("en")
        or category.get("fr")
        or "I am solely this shop's assistant and cannot respond to this request."
    )


# ══════════════════════════════════════════════════════════════════
# GUARDRAIL CLASSES
# ══════════════════════════════════════════════════════════════════

class InputGuardrail:
    """
    Full input validation pipeline. Call check() before any LLM call.

    Usage:
        result = InputGuardrail().check(raw_user_message)
        if not result.is_safe:
            return result.safe_response
        # Use result.sanitized_input for the LLM
    """

    INVISIBLE_CHAR_THRESHOLD = 3

    def check(
        self,
        raw_input: str,
        detected_lang: str | None = None,
    ) -> GuardrailResult:
        """
        6-step pipeline:
        1. Unicode sanitization
        2. Invisible-char attack detection
        3. Language auto-detection
        4. Jailbreak pattern matching (multilingual)
        5. Off-topic detection (multilingual)
        6. Empty content check
        """
        if not raw_input:
            return GuardrailResult(
                is_safe=False,
                threat_level=ThreatLevel.LOW,
                threat_type="empty_input",
                safe_response="Please ask a question.",
                sanitized_input="",
            )

        # Step 1
        sanitized, invisible_removed, homoglyphs = deep_sanitize_input(raw_input)

        # Step 2a: Bidi override = CRITICAL even if count == 1
        bidi_count = sum(1 for c in raw_input if ord(c) in _BIDI_OVERRIDES)
        if bidi_count >= 1:
            lang = detected_lang or detect_language(raw_input[:200])
            logger.warning(
                "CRITICAL: bidi override injection: %d chars, lang=%s",
                bidi_count, lang,
            )
            return GuardrailResult(
                is_safe=False,
                threat_level=ThreatLevel.CRITICAL,
                threat_type="invisible_char_injection",
                safe_response=_get_refusal("invisible_attack", lang),
                sanitized_input=sanitized,
                invisible_chars_removed=invisible_removed,
                homoglyphs_normalized=homoglyphs,
            )

        # Step 2b: General invisible-char threshold
        if invisible_removed >= self.INVISIBLE_CHAR_THRESHOLD:
            lang = detected_lang or detect_language(raw_input[:200])
            logger.warning(
                "CRITICAL: invisible injection: %d chars removed, lang=%s",
                invisible_removed, lang,
            )
            return GuardrailResult(
                is_safe=False,
                threat_level=ThreatLevel.CRITICAL,
                threat_type="invisible_char_injection",
                safe_response=_get_refusal("invisible_attack", lang),
                sanitized_input=sanitized,
                invisible_chars_removed=invisible_removed,
                homoglyphs_normalized=homoglyphs,
            )

        # Step 3
        lang = detected_lang or detect_language(sanitized)

        # Step 4
        for pattern_def in _JAILBREAK_PATTERNS:
            if pattern_def.regex.search(sanitized):
                logger.warning(
                    "Jailbreak: %s | level=%s lang=%s",
                    pattern_def.threat_type, pattern_def.level, lang,
                )
                kind = "code" if pattern_def.threat_type in (
                    "code_generation", "code_in_input"
                ) else "jailbreak"
                return GuardrailResult(
                    is_safe=False,
                    threat_level=pattern_def.level,
                    threat_type=pattern_def.threat_type,
                    safe_response=_get_refusal(kind, lang),
                    sanitized_input=sanitized,
                    invisible_chars_removed=invisible_removed,
                    homoglyphs_normalized=homoglyphs,
                )

        # Step 5
        for offtopic_pattern, category in _OFFTOPIC_CATEGORIES:
            if offtopic_pattern.search(sanitized):
                logger.info("Off-topic: %s lang=%s", category, lang)
                return GuardrailResult(
                    is_safe=False,
                    threat_level=ThreatLevel.LOW,
                    threat_type=f"offtopic_{category}",
                    safe_response=_get_refusal("offtopic", lang),
                    sanitized_input=sanitized,
                )

        # Step 6
        if len(sanitized.strip()) < 2:
            return GuardrailResult(
                is_safe=False,
                threat_level=ThreatLevel.LOW,
                threat_type="empty_after_sanitization",
                safe_response=_get_refusal("jailbreak", lang),
                sanitized_input=sanitized,
            )

        return GuardrailResult(
            is_safe=True,
            threat_level=ThreatLevel.NONE,
            sanitized_input=sanitized,
            invisible_chars_removed=invisible_removed,
            homoglyphs_normalized=homoglyphs,
        )


class OutputGuardrail:
    """Post-processes LLM output to enforce all bot rules."""

    def check_and_clean(self, text: str) -> str:
        return sanitize_output(text)
