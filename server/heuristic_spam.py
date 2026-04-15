"""Lightweight promo/urgency cues the TF–IDF model often misses (Unicode subjects, ₹ prices, etc.)."""

from __future__ import annotations

import re


def heuristic_spam_mass(text: str) -> float:
    """
    Returns extra spam probability mass in [0, ~0.48], capped, to add on top of ML score.
    Tuned so obvious marketing stays below invoice-style ham that lacks these cues.
    """
    tl = text.lower()
    mass = 0.0

    if re.search(r"p\.?\s*s\.", tl):
        mass += 0.1
    if re.search(r"won['\u2019]?t last", tl):
        mass += 0.12
    if re.search(r"price(s)? go(es)? up", tl):
        mass += 0.1
    if re.search(r"\b(re)?open(ing)?\s+enroll(ment|ing)?\b", tl):
        mass += 0.1
    if re.search(r"launch price|introductory offer|early bird", tl):
        mass += 0.1
    if re.search(r"hit reply|reply to this (e-?mail|message)\b", tl):
        mass += 0.08
    if re.search(r"₹|rs\.?\s*[0-9]{3,}|\$\s*[0-9]{3,}", text, re.I):
        mass += 0.12
    if re.search(r"\b(limited time|act now|while supplies last)\b", tl):
        mass += 0.1
    if re.search(r"\b(viagra|cialis|lottery winner)\b", tl):
        mass += 0.15

    return min(0.48, mass)
