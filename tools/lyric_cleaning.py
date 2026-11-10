import re
import unicodedata


_SECTION_NAME = (
    r"(?:pre[ -]?(?:chorus|hook|drop)|post[ -]?(?:chorus|hook)|"
    r"verse|chorus|hook|refrain|bridge|intro|outro|interlude|"
    r"instrumental(?:[ -](?:break|intro|outro|bridge))?|"
    r"guitar[ -]solo|solo|breakdown|break|drop|spoken(?:[ -]word)?)"
)
_SECTION = re.compile(
    rf"(?P<header>{_SECTION_NAME}(?:\s*(?:\d+|one|two|three|four|five|six|[ivx]+))?"
    r"(?:\s*(?:[x×]\s*\d+|\d+\s*[x×]))?)(?:\s*:\s*(?P<payload>.*))?",
    re.IGNORECASE,
)
_PRODUCTION = re.compile(
    r"(?:produced\s+by|prod\.?\s+by|written\s+by|lyrics\s+by|music\s+by|"
    r"translated\s+by|performed\s+by|vocals\s+by|"
    r"(?:producer|production|songwriter|lyrics|music|vocals)\s*:)\s*\S.*",
    re.IGNORECASE,
)
_UNKNOWN = re.compile(r"(?:[?*.…]+|inaudible|unintelligible|unknown)", re.IGNORECASE)
_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
_JOINERS = "\u200c\u200d"
_APOSTROPHES = "'’"
_LINE_ENDINGS = "\r\n\v\f\x85\u2028\u2029"


def _require_text(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")


def _performer_credit(text):
    words = text.split()
    if not words or len(words) > 12 or len(text) > 120:
        return False
    if not re.fullmatch(r"[\w\s.,'’&/+()–—-]+", text):
        return False
    connectors = {"and", "with", "the", "of", "de", "del", "la", "feat.", "ft.", "&", "/", "+"}
    names = [word for word in words if word.casefold() not in connectors]
    return bool(names) and all(
        any(char.isupper() for char in word) or word.casefold() in {"all", "both", "choir", "company"}
        for word in names
    )


def clean_brackets(
    text,
    remove_section_headers=False,
    remove_performer_names=True,
    remove_production_credits=True,
    remove_unknown_markers=True,
    remove_bracket_symbols=False,
    remove_sung_brackets=False,
):
    _require_text(text)

    def replace(match):
        original = match.group()
        content = match.group(1).strip()
        if not content or _UNKNOWN.fullmatch(content):
            return " " if remove_unknown_markers else original
        if any(char in content for char in _LINE_ENDINGS):
            return " " if remove_sung_brackets else original
        if _PRODUCTION.fullmatch(content):
            return " " if remove_production_credits else original
        section = _SECTION.fullmatch(content)
        if section is None:
            return " " if remove_sung_brackets else original
        header, payload = section.group("header", "payload")
        if payload and header.casefold().startswith("spoken"):
            return " " if remove_sung_brackets else original
        performer = bool(payload) and _performer_credit(payload)
        if payload and not performer:
            return f"[{payload}]" if remove_section_headers else original
        kept_header = "" if remove_section_headers else header
        kept_payload = "" if remove_performer_names else (payload or "")
        if not remove_section_headers and not (performer and remove_performer_names):
            return original
        retained = ": ".join(part for part in (kept_header, kept_payload) if part)
        return f"[{retained}]" if retained else " "

    result = _BRACKETS.sub(replace, text)
    if remove_bracket_symbols:
        result = result.replace("[", "").replace("]", "")
    return result.strip() if result != text else text


def clean_unbracketed_headers(
    text,
    remove_section_headers=False,
    remove_production_credits=True,
):
    _require_text(text)
    output = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip(_LINE_ENDINGS)
        ending = line[len(body):]
        content = body.strip()
        if remove_production_credits and _PRODUCTION.fullmatch(content):
            continue
        section = _SECTION.fullmatch(content) if remove_section_headers else None
        if section is None:
            output.append(line)
        elif section.group("payload"):
            output.append(section.group("payload") + ending)
    result = "".join(output)
    return "" if result != text and not result.strip() else result


def deduplicate_lines(text):
    _require_text(text)
    output, seen = [], set()
    for line in text.splitlines(keepends=True):
        key = " ".join(unicodedata.normalize("NFC", line).split()).casefold()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        output.append(line)
    return "".join(output)


def _word_character(char):
    return char.isalnum() or unicodedata.category(char).startswith("M")


def _internal_connector(text, position):
    return (
        0 < position < len(text) - 1
        and _word_character(text[position - 1])
        and _word_character(text[position + 1])
    )


def tokenize_text(text):
    _require_text(text)
    tokens, current = [], []
    for position, char in enumerate(text):
        if (_word_character(char) and (char.isalnum() or current)) or (
            char in _APOSTROPHES + _JOINERS and _internal_connector(text, position)
        ):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def remove_punctuation(text):
    _require_text(text)
    output = []
    for position, char in enumerate(text):
        if char in _APOSTROPHES + _JOINERS:
            output.append(char if _internal_connector(text, position) else " ")
        elif unicodedata.category(char).startswith(("P", "S")):
            output.append(" ")
        elif unicodedata.category(char).startswith("M") and (
            not output or not (_word_character(output[-1]) or output[-1] in _JOINERS)
        ):
            output.append(" ")
        else:
            output.append(char)
    result = re.sub(r"[ \t]+", " ", "".join(output))
    result = re.sub(r" *([\r\n]) *", r"\1", result)
    return result.strip(" ")
