"""Unicode-safe lyric normalization and reproducible character diagnostics.

UTF-8 is the file encoding. ISO-8859-14 (Latin-8) compatibility is only a
diagnostic, never a language detector or a reason to discard a song.
"""

from collections import Counter
from functools import lru_cache
import re
import unicodedata as ud


LATIN8_CHARACTERS = frozenset(bytes(range(256)).decode("iso8859_14"))
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_MOJIBAKE = re.compile(r"â[€\x80-\x9f]|Ã[\x80-\xbf]|Â[\x80-\xbf]|ï»¿|ï¿½|ðŸ")
_PUNCTUATION = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "…": "...",
})
# Keep ZWJ/ZWNJ: these have linguistic and emoji meaning. A zero-width
# space marks a boundary; BOM, soft hyphens and directional marks do not.
_FORMAT = str.maketrans({
    "\u200b": " ", "\ufeff": "", "\u00ad": "", "\u2060": "",
    **{chr(cp): "" for cp in (0x200E, 0x200F, *range(0x202A, 0x202F),
                              *range(0x2066, 0x206A))},
})
# This corpus overwhelmingly contains Cyrillic IE inside otherwise Latin words.
# Deliberately do not generalize this to transliteration or all confusables.
_CYRILLIC_IE = str.maketrans({"е": "e", "Е": "E"})
_C1 = {}
for _cp in range(0x80, 0xA0):
    try:
        _C1[_cp] = bytes([_cp]).decode("cp1252")
    except UnicodeDecodeError:
        _C1[_cp] = " "
# NEL is a genuine Unicode line ending; do not reinterpret it as an ellipsis.
_C1[0x85] = "\n"


@lru_cache(maxsize=8192)
def character_kind(char):
    """Return a diagnostic bucket using Unicode names/categories, not language."""
    category = ud.category(char)
    if char in "\ufffd\ufffc":
        return "replacement_or_object"
    if char.isspace():
        return "whitespace"
    if category.startswith("L"):
        name = ud.name(char, "")
        if "LATIN" in name:
            return "latin_letter"
        if "CYRILLIC" in name:
            return "cyrillic_letter"
        return "other_script_letter"
    return {"P": "punctuation", "M": "combining_mark", "S": "symbol",
            "N": "number", "C": "control_or_format"}.get(category[0], "other")


def _repair_mixed_word(match):
    word = match.group()
    if "е" not in word and "Е" not in word:
        return word
    latin = sum(character_kind(c) == "latin_letter" for c in word)
    # Only IE may be non-Latin, and at least half of the word must be Latin.
    if latin and latin * 2 >= len(word) and all(
        character_kind(c) == "latin_letter" or c in "еЕ" for c in word
    ):
        return word.translate(_CYRILLIC_IE)
    return word


def normalize_lyrics(text, *, repair_mixed_script=False, preserve_lines=True):
    """Normalize text without ASCII conversion, language filtering or case changes.

    Uses NFC, regular spaces, consistent quotes/dashes and LF line endings.
    Known Windows-1252 C1 punctuation is recovered before other control codes
    become spaces. Suspected mojibake and U+FFFD remain visible for review.
    Accents, non-Latin scripts, emoji and meaningful joiners are preserved.

    ``repair_mixed_script=True`` opts into a corpus-specific heuristic: Cyrillic
    е/Е becomes Latin e/E in majority/half-Latin words, only in documents with
    at least 90% Latin letters. It is not language detection. Default is off.
    None/NaN/pd.NA pass through; non-missing non-strings raise TypeError.
    """
    if not isinstance(text, str):
        import pandas as pd
        if pd.api.types.is_scalar(text) and pd.isna(text):
            return text
        raise TypeError("lyrics must be a string or a missing scalar")
    text = ud.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    # These code points are a recognizable legacy artifact, not a file re-decode.
    text = text.translate(_C1).translate(_PUNCTUATION)
    # In this scrape, zero-width spaces also occur inside English words. Remove
    # only those inside Latin-letter runs; preserve a boundary in other contexts.
    def inside_latin_word(match):
        left, right = match.group(1), match.string[match.end()]
        if character_kind(left) == character_kind(right) == "latin_letter":
            return left
        return left + " "
    text = re.sub(r"([^\W\d_])\u200b(?=[^\W\d_])", inside_latin_word, text)
    text = text.translate(_FORMAT)
    if repair_mixed_script and ("е" in text or "Е" in text):
        letters = Counter(character_kind(c) for c in text if c.isalpha())
        if letters["latin_letter"] >= 0.9 * sum(letters.values()):
            text = _WORD.sub(_repair_mixed_word, text)
    replacements = {}
    for c in set(text):
        if c in "\n\u2028\u2029\x85":
            replacements[ord(c)] = "\n" if preserve_lines else " "
        elif c.isspace() or ud.category(c) == "Cc" or c == "\ufffc":
            replacements[ord(c)] = " "
    text = text.translate(replacements)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return ud.normalize("NFC", text.strip())


def audit_lyrics(lyrics):
    """Return (per-row diagnostics, non-ASCII character inventory) for a Series.

    Character row counts count a character once per song. Diagnostic flags
    overlap. Script ratios use Unicode letter names/categories as a heuristic,
    not the complete Unicode Script property, and do not detect English.
    No input is modified. Output keeps the input index, including duplicate labels.
    """
    import pandas as pd

    records, occurrences, row_counts = [], Counter(), Counter()
    for text in lyrics:
        if not isinstance(text, str):
            if not (pd.api.types.is_scalar(text) and pd.isna(text)):
                raise TypeError("lyrics must contain strings or missing scalars")
            records.append({"is_text": False})
            continue
        counts = Counter(text)
        non_ascii = {c: n for c, n in counts.items() if ord(c) > 127}
        occurrences.update(non_ascii)
        row_counts.update(non_ascii.keys())
        letters = Counter()
        for c, n in counts.items():
            if c.isalpha():
                letters[character_kind(c)] += n
        non_latin = letters["cyrillic_letter"] + letters["other_script_letter"]
        total_letters = sum(letters.values())
        mixed_words = sum(
            any(character_kind(c) == "latin_letter" for c in word)
            and any(character_kind(c) in ("cyrillic_letter", "other_script_letter") for c in word)
            for word in _WORD.findall(text) if not word.isascii()
        )
        records.append({
            "is_text": True,
            "latin8_compatible": all(c in LATIN8_CHARACTERS for c in counts),
            "has_non_ascii": bool(non_ascii),
            "has_unicode_punctuation": any(character_kind(c) == "punctuation" for c in non_ascii),
            "has_unicode_whitespace": any(c.isspace() for c in non_ascii),
            "has_format_controls": any(ud.category(c) == "Cf" for c in counts),
            "has_c1_controls": any("\x80" <= c <= "\x9f" for c in counts),
            "has_unexpected_controls": any(ud.category(c) == "Cc" and c not in "\n\r\t" for c in counts),
            "has_replacement": "\ufffd" in counts,
            "has_object_placeholder": "\ufffc" in counts,
            "mojibake_suspected": bool(_MOJIBAKE.search(text)),
            "has_non_latin_letters": bool(non_latin),
            "has_other_script_letters": bool(letters["other_script_letter"]),
            "has_cyrillic_letters": bool(letters["cyrillic_letter"]),
            "mixed_script_words": mixed_words,
            "latin_letters": letters["latin_letter"],
            "non_latin_letters": non_latin,
            "letter_count": total_letters,
            "non_latin_letter_ratio": non_latin / total_letters if total_letters else 0.0,
        })
    row_columns = ["is_text", "latin8_compatible", "has_non_ascii",
                   "has_unicode_punctuation", "has_unicode_whitespace",
                   "has_format_controls", "has_c1_controls", "has_unexpected_controls", "has_replacement",
                   "has_object_placeholder", "mojibake_suspected", "has_non_latin_letters",
                   "has_other_script_letters", "has_cyrillic_letters", "mixed_script_words",
                   "latin_letters", "non_latin_letters", "letter_count", "non_latin_letter_ratio"]
    rows = pd.DataFrame(records, index=lyrics.index, columns=row_columns)
    for col in row_columns:
        if col == "is_text" or col == "latin8_compatible" or col.startswith("has_") or col == "mojibake_suspected":
            rows[col] = rows[col].astype("boolean")
    character_columns = ["character", "escaped", "codepoint", "name", "category",
                         "kind", "latin8_compatible", "occurrences", "rows"]
    characters = pd.DataFrame([
        {"character": c, "escaped": ascii(c), "codepoint": f"U+{ord(c):04X}",
         "name": ud.name(c, "UNNAMED"), "category": ud.category(c),
         "kind": character_kind(c), "latin8_compatible": c in LATIN8_CHARACTERS,
         "occurrences": n, "rows": row_counts[c]}
        for c, n in occurrences.items()
    ], columns=character_columns)
    characters = characters.sort_values(["rows", "codepoint"], ascending=[False, True]).reset_index(drop=True)
    return rows, characters
