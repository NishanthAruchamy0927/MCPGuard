"""Text normalisation and lexical analysis for tool descriptions."""
from __future__ import annotations

import re
import string
import unicodedata

# Compact English stop-word list (kept local to avoid an NLTK download at runtime).
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves
""".split())

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Lexicons used by the engineered risk features. Each is documented in docs/FEATURES.md.
VAGUE_TERMS = frozenset("""
various etc something stuff things thing data info information handle handles handling process processes manage
manages appropriate appropriately relevant some certain misc miscellaneous general generic utility helper perform
performs operation operations action actions needed necessary whatever specified given related accordingly may might
possibly usually typically
""".split())
INSTRUCTION_TERMS = frozenset("""
use must should provide specify return returns call pass include ensure set required requires require only always
never avoid do don't make select choose enter supply give
""".split())
CONSTRAINT_TERMS = frozenset("""
format must between maximum minimum max min only exactly required range limit limits allowed valid one of either
default iso yyyy mm dd hh utc integer number string boolean array list characters chars length pattern regex enum
""".split())
EXAMPLE_PATTERNS = re.compile(r"\b(e\.g\.|eg\.|for example|example|such as|i\.e\.|like ')", re.I)
FORMAT_HINT_PATTERNS = re.compile(
    r"(\bformat\b|yyyy|mm/dd|hh:mm|iso ?8601|\bunits?\b|\bin (?:celsius|fahrenheit|meters|km|miles|usd|seconds|minutes)\b|"
    r"'[^']{1,40}'|\"[^\"]{1,40}\"|e\.g\.|for example|such as|one of|either|allowed values|\bregex\b|pattern)", re.I)


def normalize_text(text: object) -> str:
    """Unicode NFKC + whitespace normalisation. Non-strings become ''."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("​", "")
    return _WS.sub(" ", s).strip()


def split_identifier(name: str) -> str:
    """``getUserProfile`` / ``get_user.profile`` -> ``get user profile``."""
    name = normalize_text(name)
    name = _CAMEL.sub(" ", name)
    return _WS.sub(" ", re.sub(r"[_.\-/:]+", " ", name)).strip().lower()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text or "") if s.strip()]


def lexical_stats(text: str) -> dict[str, float]:
    """Interpretable lexical statistics of a description."""
    text = normalize_text(text)
    toks = tokenize(text)
    n_tok = len(toks)
    content = [t for t in toks if t not in STOPWORDS]
    n_char = len(text)
    punct = sum(ch in string.punctuation for ch in text)
    special = sum((not ch.isalnum()) and (not ch.isspace()) and ch not in ".,;:'\"()-" for ch in text)
    upper = sum(ch.isupper() for ch in text)
    sents = sentences(text)
    return {
        "desc_char_len": float(n_char),
        "desc_token_count": float(n_tok),
        "desc_sentence_count": float(len(sents)),
        "desc_avg_token_len": float(sum(map(len, toks)) / n_tok) if n_tok else 0.0,
        "desc_vocab_size": float(len(set(toks))),
        "desc_lexical_diversity": float(len(set(toks)) / n_tok) if n_tok else 0.0,
        "desc_stopword_ratio": float(1 - len(content) / n_tok) if n_tok else 0.0,
        "desc_punct_density": punct / n_char if n_char else 0.0,
        "desc_special_char_density": special / n_char if n_char else 0.0,
        "desc_uppercase_ratio": upper / n_char if n_char else 0.0,
        "desc_digit_count": float(sum(ch.isdigit() for ch in text)),
        "vague_term_count": float(sum(t in VAGUE_TERMS for t in toks)),
        "instruction_term_count": float(sum(t in INSTRUCTION_TERMS for t in toks)),
        "constraint_term_count": float(sum(t in CONSTRAINT_TERMS for t in toks)),
        "example_mentions": float(len(EXAMPLE_PATTERNS.findall(text))),
        "format_hint_mentions": float(len(FORMAT_HINT_PATTERNS.findall(text))),
    }
