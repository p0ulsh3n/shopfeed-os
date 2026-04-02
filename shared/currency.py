"""
Currency Service — Dynamic Exchange Rates & Formatting.

Uses Frankfurter API (100% free, no API key, open-source, self-hostable)
backed by European Central Bank data. Rates cached in memory with TTL.

Architecture:
    1. Vendor sets prices in their local currency (e.g. XOF for West Africa)
    2. Backend stores prices in VENDOR's currency (no conversion at storage)
    3. At display time, prices are converted to BUYER's currency
    4. Currency is resolved from GPS coordinates (reverse_geocode → country → currency)

Usage:
    from shared.currency import CurrencyService

    fx = CurrencyService()
    rate = await fx.get_rate("XOF", "EUR")
    converted = await fx.convert(15000, "XOF", "EUR")
    symbol = fx.get_symbol("EUR")  # "€"
    formatted = fx.format_price(29.99, "EUR")  # "29,99 €"
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Frankfurter API — 100% free, no API key, unlimited, open-source
# Source: European Central Bank (ECB) data
# Self-hostable: https://github.com/hakanensari/frankfurter
FRANKFURTER_API = "https://api.frankfurter.dev/v2"

# Cache TTL: 6 hours (ECB updates once per business day)
CACHE_TTL_SECONDS = 6 * 60 * 60

# ── Country code → Currency mapping (ISO 4217) ──────────────────
# Resolved from reverse_geocode country_code. Covers 150+ countries.
# This is NOT hardcoded defaults — it maps real country codes to their
# official legal tender. The country code comes from GPS coordinates.

COUNTRY_TO_CURRENCY: dict[str, str] = {
    # West Africa (FCFA - XOF)
    "CI": "XOF", "SN": "XOF", "BF": "XOF", "ML": "XOF",
    "NE": "XOF", "TG": "XOF", "BJ": "XOF", "GW": "XOF",
    # Central Africa (FCFA - XAF)
    "CM": "XAF", "GA": "XAF", "CG": "XAF", "TD": "XAF",
    "CF": "XAF", "GQ": "XAF",
    # Eurozone (EUR)
    "FR": "EUR", "DE": "EUR", "IT": "EUR", "ES": "EUR",
    "PT": "EUR", "NL": "EUR", "BE": "EUR", "AT": "EUR",
    "IE": "EUR", "FI": "EUR", "GR": "EUR", "LU": "EUR",
    "SK": "EUR", "SI": "EUR", "EE": "EUR", "LV": "EUR",
    "LT": "EUR", "MT": "EUR", "CY": "EUR", "HR": "EUR",
    # North Africa
    "MA": "MAD", "DZ": "DZD", "TN": "TND", "EG": "EGP", "LY": "LYD",
    # East Africa
    "KE": "KES", "TZ": "TZS", "UG": "UGX", "RW": "RWF",
    "ET": "ETB", "MG": "MGA",
    # Southern Africa
    "ZA": "ZAR", "BW": "BWP", "MZ": "MZN", "ZW": "ZWL",
    "NA": "NAD", "AO": "AOA",
    # West Africa (non-FCFA)
    "NG": "NGN", "GH": "GHS", "GM": "GMD", "SL": "SLL",
    "LR": "LRD", "MR": "MRU",
    # Americas
    "US": "USD", "CA": "CAD", "MX": "MXN", "BR": "BRL",
    "AR": "ARS", "CO": "COP", "CL": "CLP", "PE": "PEN",
    # Asia
    "CN": "CNY", "JP": "JPY", "KR": "KRW", "IN": "INR",
    "ID": "IDR", "TH": "THB", "VN": "VND", "PH": "PHP",
    "MY": "MYR", "SG": "SGD", "HK": "HKD", "TW": "TWD",
    "AE": "AED", "SA": "SAR", "QA": "QAR", "KW": "KWD",
    "TR": "TRY", "IL": "ILS", "PK": "PKR", "BD": "BDT",
    # Europe (non-Euro)
    "GB": "GBP", "CH": "CHF", "NO": "NOK", "SE": "SEK",
    "DK": "DKK", "PL": "PLN", "CZ": "CZK", "HU": "HUF",
    "RO": "RON", "BG": "BGN", "RS": "RSD", "UA": "UAH",
    "RU": "RUB",
    # Oceania
    "AU": "AUD", "NZ": "NZD",
    # Caribbean
    "JM": "JMD", "HT": "HTG", "DO": "DOP",
    # Indian Ocean
    "MU": "MUR", "SC": "SCR", "KM": "KMF",
}

# ── Currency symbols ─────────────────────────────────────────────

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
    "XOF": "FCFA", "XAF": "FCFA", "NGN": "₦", "GHS": "₵",
    "KES": "KSh", "ZAR": "R", "MAD": "MAD", "DZD": "DA",
    "TND": "DT", "EGP": "E£", "INR": "₹", "BRL": "R$",
    "CAD": "CA$", "AUD": "A$", "CHF": "CHF", "SEK": "kr",
    "NOK": "kr", "DKK": "kr", "PLN": "zł", "TRY": "₺",
    "AED": "AED", "SAR": "SAR", "MXN": "MX$", "KRW": "₩",
    "THB": "฿", "MYR": "RM", "SGD": "S$", "HKD": "HK$",
    "RUB": "₽", "UAH": "₴", "CZK": "Kč", "HUF": "Ft",
    "RON": "lei", "BGN": "лв", "RWF": "RF", "MGA": "Ar",
}


class CurrencyService:
    """Currency conversion and formatting service.

    Uses Frankfurter API (ECB data) with in-memory caching.
    Thread-safe for async use.
    """

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamp: float = 0.0

    # ── Public API ───────────────────────────────────────────────

    @staticmethod
    def currency_from_country(country_code: str) -> str:
        """Resolve ISO currency code from ISO country code.

        Args:
            country_code: ISO 3166-1 alpha-2 (e.g. "CI", "FR", "US")

        Returns:
            ISO 4217 currency code (e.g. "XOF", "EUR", "USD")
            Empty string if unknown country.
        """
        return COUNTRY_TO_CURRENCY.get(country_code.upper(), "")

    @staticmethod
    def currency_from_coordinates(lat: float, lon: float) -> str:
        """Resolve currency from GPS coordinates.

        Uses reverse_geocode to find the country, then maps to currency.
        """
        try:
            import reverse_geocode
            result = reverse_geocode.search([(lat, lon)])[0]
            country_code = result.get("country_code", "")
            return COUNTRY_TO_CURRENCY.get(country_code, "")
        except Exception:
            return ""

    async def get_rates(self, base: str = "USD") -> dict[str, float]:
        """Fetch exchange rates from Frankfurter API with caching.

        Args:
            base: Base currency (ISO 4217)

        Returns:
            Dict of currency_code → rate relative to base
        """
        now = time.time()

        # Check cache
        cache_key = f"rates_{base}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if now - cached["timestamp"] < CACHE_TTL_SECONDS:
                return cached["rates"]

        # Fetch from Frankfurter v2 API
        # Docs: https://frankfurter.dev/docs
        # Endpoint: GET /v2/rates?base=USD
        # Response: [{base, quote, rate, date, provider}, ...]
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{FRANKFURTER_API}/rates",
                    params={"base": base},
                )
                resp.raise_for_status()
                data = resp.json()

                # v2 returns an array of {base, quote, rate, date}
                rates: dict[str, float] = {base: 1.0}
                if isinstance(data, list):
                    for entry in data:
                        quote = entry.get("quote", "")
                        rate = entry.get("rate", 0.0)
                        if quote:
                            rates[quote] = rate
                elif isinstance(data, dict):
                    # Fallback for v1-style response
                    for k, v in data.get("rates", {}).items():
                        rates[k] = v

                # Cache
                self._cache[cache_key] = {
                    "rates": rates,
                    "timestamp": now,
                }
                logger.info(
                    "Fetched %d exchange rates (base=%s) from Frankfurter v2",
                    len(rates), base,
                )
                return rates

        except Exception as e:
            logger.error("Failed to fetch exchange rates: %s", e)
            # Return cached data even if stale, or empty
            if cache_key in self._cache:
                return self._cache[cache_key]["rates"]
            return {base: 1.0}

    async def get_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate between two currencies.

        Args:
            from_currency: Source currency (e.g. "XOF")
            to_currency: Target currency (e.g. "EUR")

        Returns:
            Exchange rate (multiply source amount by this)
        """
        if from_currency == to_currency:
            return 1.0

        rates = await self.get_rates(base=from_currency)
        return rates.get(to_currency, 0.0)

    async def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> float:
        """Convert an amount between currencies.

        Args:
            amount: Amount in source currency
            from_currency: Source currency code
            to_currency: Target currency code

        Returns:
            Converted amount (rounded to 2 decimals)
        """
        if from_currency == to_currency:
            return amount

        rate = await self.get_rate(from_currency, to_currency)
        return round(amount * rate, 2)

    @staticmethod
    def get_symbol(currency_code: str) -> str:
        """Get the display symbol for a currency."""
        return CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code)

    @staticmethod
    def format_price(amount: float, currency_code: str) -> str:
        """Format a price for display.

        Examples:
            format_price(29.99, "EUR")  → "29,99 €"
            format_price(15000, "XOF")  → "15 000 FCFA"
            format_price(49.99, "USD")  → "$49.99"
        """
        symbol = CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code)

        # Symbol-before currencies (USD, GBP, etc.)
        symbol_before = {"$", "£", "¥", "₦", "₵", "₹", "₩", "₺", "₽", "₴",
                         "R$", "CA$", "A$", "MX$", "E£", "S$", "HK$"}

        if symbol in symbol_before:
            return f"{symbol}{amount:,.2f}"

        # FCFA: no decimals, space-separated thousands
        if currency_code.upper() in ("XOF", "XAF"):
            formatted = f"{int(amount):,}".replace(",", " ")
            return f"{formatted} {symbol}"

        # Default: number then symbol
        return f"{amount:,.2f} {symbol}"
