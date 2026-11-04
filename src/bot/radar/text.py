"""Sanitizing untrusted post text. Never eval/exec anything from a post."""
from __future__ import annotations

import hashlib
import html
import re

MAX_TEXT = 1000

_BLOCK = re.compile(r"<(script|style|noscript|iframe|svg)\b.*?</\1\s*>", re.I | re.S)
_TAG = re.compile(r"<[^>]{0,2000}>", re.S)
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁠-⁤﻿]")
_WS = re.compile(r"\s+")


def clean_text(raw: str | None, cap: int = MAX_TEXT) -> str:
    """Strip HTML (twice, to catch entity-escaped markup), control/bidi chars; collapse whitespace; cap length."""
    if not raw:
        return ""
    s = str(raw)[: cap * 20]  # bound the work on huge inputs
    for _ in range(2):
        s = _BLOCK.sub(" ", s)
        s = _TAG.sub(" ", s)
        s = html.unescape(s)
    s = _CTRL.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s[:cap]


def text_hash(title: str, text: str) -> str:
    """Hash of the normalized content, so syndicated copies with tiny differences dedupe."""
    norm = re.sub(r"[^a-z0-9]+", " ", f"{title} {text}".lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()
