"""Value cleaners and normalizers for raw CSV/XLSX data."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Union

import pandas as pd


_NUMBER_RE = re.compile(r"-?\d+(?:[\.,]\d+)?")


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def clean_rent(value) -> Optional[int]:
    """Return rent in dollars/month as an int.

    Examples
    --------
    >>> clean_rent("$2,500")
    2500
    >>> clean_rent("2500/mo")
    2500
    >>> clean_rent("Contact for price") is None
    True
    """
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return int(round(value))
    text = str(value).strip()
    text = text.replace(",", "")
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return int(round(float(match.group(0))))
    except ValueError:
        return None


def clean_bedrooms(value) -> Optional[int]:
    """Return bedroom count, mapping ``"Studio"`` -> 0."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return max(0, int(value))
    text = str(value).strip().lower()
    if "studio" in text or text in ("0", "0br", "0 bed"):
        return 0
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))
    return None


def clean_bathrooms(value) -> Optional[float]:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value).strip().lower()
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    return None


def clean_numeric(value) -> Optional[float]:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def clean_integer(value) -> Optional[int]:
    num = clean_numeric(value)
    if num is None:
        return None
    return int(round(num))


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
)


def clean_date(value) -> Optional[str]:
    """Return ISO ``YYYY-MM-DD`` or ``None``.

    Falls back to ``pandas.to_datetime`` when the explicit formats miss.
    """
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        parsed = None
    if parsed is None or pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def standardize_text(value) -> Optional[str]:
    if _is_missing(value):
        return None
    return " ".join(str(value).strip().split()).strip()


_BOROUGH_NORMALIZATION: dict[str, str] = {
    "mn": "Manhattan",
    "manhattan": "Manhattan",
    "new york": "Manhattan",
    "new york county": "Manhattan",
    "ny": "Manhattan",
    "bk": "Brooklyn",
    "brooklyn": "Brooklyn",
    "kings": "Brooklyn",
    "kings county": "Brooklyn",
    "qn": "Queens",
    "queens": "Queens",
    "queens county": "Queens",
    "bx": "Bronx",
    "bronx": "Bronx",
    "bronx county": "Bronx",
    "the bronx": "Bronx",
    "si": "Staten Island",
    "staten island": "Staten Island",
    "richmond": "Staten Island",
    "richmond county": "Staten Island",
}


def normalize_borough_name(value) -> Optional[str]:
    """Map the many borough spellings to one of MN/BK/QN/BX/SI display names."""
    if _is_missing(value):
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in _BOROUGH_NORMALIZATION:
        return _BOROUGH_NORMALIZATION[raw]
    # Numeric borocode (1=MN, 2=BX, 3=BK, 4=QN, 5=SI).
    code_map = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}
    if raw in code_map:
        return code_map[raw]
    # Sometimes "BROOKLYN, NY" -> take leftmost token.
    head = raw.split(",", 1)[0].strip()
    return _BOROUGH_NORMALIZATION.get(head, str(value).strip().title())
