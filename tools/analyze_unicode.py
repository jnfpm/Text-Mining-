"""Reproduce the train.csv Unicode investigation without modifying the dataset.

Run from the project root: python -m tools.analyze_unicode --input data/train.csv --output reports
The audit uses all rows, not a sample. Output contains aggregate metrics,
character counts, short review examples and a diagnostic figure, not new lyrics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

import pandas as pd

from tools.text_quality import LATIN8_CHARACTERS, audit_lyrics, normalize_lyrics


def latin8_mask(lyrics):
    return lyrics.map(lambda text: all(c in LATIN8_CHARACTERS for c in set(text)))


def analyze(df):
    """Return JSON-compatible results and derived columns, preserving raw data."""
    raw = df["lyrics"]
    print("Auditing all original lyrics...", flush=True)
    audit, characters = audit_lyrics(raw)
    if not audit["is_text"].all():
        raise ValueError("Training analysis expects non-missing string lyrics")
    failed = ~audit["latin8_compatible"]
    typography_only = raw.map(lambda s: all(
        c in LATIN8_CHARACTERS or unicodedata.category(c)[0] in "PZ"
        for c in set(s)
    )) & failed
    print("Normalizing typography, then testing the optional mixed-script repair...", flush=True)
    normalized = raw.map(normalize_lyrics)
    repaired = normalized.map(lambda s: normalize_lyrics(s, repair_mixed_script=True))
    ordinary_mask, repaired_mask = latin8_mask(normalized), latin8_mask(repaired)
    # Idempotence matters: preprocessing must not keep changing on each run.
    print("Checking idempotence and remaining Unicode...", flush=True)
    stable = all(normalize_lyrics(s, repair_mixed_script=True) == s for s in repaired)
    remaining_audit, _ = audit_lyrics(repaired.loc[~repaired_mask])
    n = len(df)
    genre = pd.DataFrame({"tag": df["tag"], "raw_latin8_failure": failed,
                          "normalized_latin8_failure": ~ordinary_mask,
                          "repaired_latin8_failure": ~repaired_mask})
    genre = genre.groupby("tag", observed=True).agg(
        rows=("raw_latin8_failure", "size"),
        raw_latin8_failures=("raw_latin8_failure", "sum"),
        normalized_latin8_failures=("normalized_latin8_failure", "sum"),
        repaired_latin8_failures=("repaired_latin8_failure", "sum"),
    )
    genre["pct_lost_if_dropped"] = 100 * genre["raw_latin8_failures"] / genre["rows"]
    genre["original_share_pct"] = 100 * genre["rows"] / n
    kept = genre["rows"] - genre["raw_latin8_failures"]
    genre["share_after_dropping_pct"] = 100 * kept / kept.sum()

    review_flags = audit["mojibake_suspected"] | audit["has_replacement"]
    # A review threshold, not a language label; includes code-switching.
    substantial_other_script = (audit["non_latin_letter_ratio"] >= 0.05) & (audit["non_latin_letters"] >= 20)
    examples = {}
    selections = {
        "typography_only": typography_only,
        "mixed_script_repaired": normalized.ne(repaired),
        "mojibake_suspected": audit["mojibake_suspected"],
        "replacement_character": audit["has_replacement"],
        "substantial_non_latin_script": substantial_other_script,
        "remaining_latin8_incompatible": ~repaired_mask,
    }
    for label, mask in selections.items():
        examples[label] = []
        for idx in df.index[mask][:8]:
            text = raw.loc[idx]
            if label == "mixed_script_repaired":
                position = next((i for i, c in enumerate(text) if c in "еЕ"), 0)
            elif label == "replacement_character":
                position = text.index("\ufffd")
            elif label == "mojibake_suspected":
                match = re.search(r"â[€\x80-\x9f]|Ã[\x80-\xbf]|Â[\x80-\xbf]|ï»¿|ï¿½|ðŸ", text)
                position = match.start()
            elif label == "substantial_non_latin_script":
                position = next((i for i, c in enumerate(text)
                                 if c.isalpha() and "LATIN" not in unicodedata.name(c, "")), 0)
            elif label == "remaining_latin8_incompatible":
                position = next((i for i, c in enumerate(text)
                                 if c not in LATIN8_CHARACTERS
                                 and unicodedata.category(c)[0] not in "PZ"), 0)
            else:
                position = next((i for i, c in enumerate(text) if ord(c) > 127), 0)
            excerpt = text[max(0, position - 35):position + 100]
            # Display text from the actual full-document result: an excerpt's
            # Latin-letter ratio may differ from that of the original song.
            # Prefix normalization estimates the corresponding character offset;
            # optional IE repair changes no string lengths.
            normalized_position = len(normalize_lyrics(text[:position]))
            examples[label].append({
                "id": int(df.at[idx, "id"]), "title": df.at[idx, "title"],
                "tag": df.at[idx, "tag"], "raw_excerpt": excerpt,
                "normalized_excerpt": repaired.loc[idx][max(0, normalized_position - 35):normalized_position + 100],
                "non_ascii_characters": ascii("".join(sorted({c for c in text if ord(c) > 127})))[:250],
            })
    duplicate_raw = int(raw.duplicated().sum())
    duplicate_normalized = int(repaired.duplicated().sum())
    duplicate_groups = pd.DataFrame({"lyrics": repaired, "tag": df["tag"]})
    duplicate_groups = duplicate_groups.loc[repaired.duplicated(keep=False)]
    conflicting_groups = int((duplicate_groups.groupby("lyrics")["tag"].nunique() > 1).sum())
    results = {
        "rows": n, "unicode_database_version": unicodedata.unidata_version,
        "missing_lyrics": int(raw.isna().sum()),
        "non_ascii_rows": int(audit["has_non_ascii"].sum()),
        "latin1_incompatible_rows": int(raw.map(lambda s: not all(ord(c) < 256 for c in set(s))).sum()),
        "latin8_incompatible_rows": int(failed.sum()),
        "latin8_incompatible_pct": 100 * int(failed.sum()) / n,
        "latin8_failures_only_punctuation_or_separators": int(typography_only.sum()),
        "latin8_ignored_character_loss": int(characters.loc[~characters["latin8_compatible"], "occurrences"].sum()),
        "nfc_changed_rows": int(raw.map(lambda s: unicodedata.normalize("NFC", s) != s).sum()),
        "row_flags": {c: int(audit[c].sum()) for c in audit if c.startswith("has_") or c == "mojibake_suspected"},
        "mixed_script_rows": int(audit["mixed_script_words"].gt(0).sum()),
        "substantial_non_latin_script_review_rows": int(substantial_other_script.sum()),
        "corruption_review_rows": int(review_flags.sum()),
        "normalization_stages": [
            {"stage": "original", "latin8_incompatible_rows": int(failed.sum()), "changed_rows": 0},
            {"stage": "unicode_normalization", "latin8_incompatible_rows": int((~ordinary_mask).sum()), "changed_rows": int(raw.ne(normalized).sum())},
            {"stage": "optional_contextual_ie_repair", "latin8_incompatible_rows": int((~repaired_mask).sum()), "changed_rows": int(raw.ne(repaired).sum())},
        ],
        "optional_repair_changed_rows": int(normalized.ne(repaired).sum()),
        "remaining_row_flags": {c: int(remaining_audit[c].sum()) for c in remaining_audit if c.startswith("has_") or c == "mojibake_suspected"},
        "duplicates": {"raw_extra_rows": duplicate_raw, "normalized_extra_rows": duplicate_normalized,
                       "additional_extra_rows": duplicate_normalized - duplicate_raw,
                       "normalized_groups_with_conflicting_tags": conflicting_groups},
        "checks": {"normalizer_idempotent_all_rows": stable, "rows_retained": n,
                   "newly_empty_lyrics": int((repaired.str.strip().eq("") & raw.str.strip().ne("")).sum()),
                   "all_normalized_nfc": all(unicodedata.is_normalized("NFC", s) for s in repaired)},
        "genre_impact": genre.reset_index().to_dict(orient="records"),
        "characters": characters.to_dict(orient="records"),
        "examples": examples,
    }
    return results, normalized, repaired, audit, characters


def plot_results(results, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    genres = pd.DataFrame(results["genre_impact"]).sort_values("pct_lost_if_dropped")
    axes[0].barh(genres["tag"], genres["pct_lost_if_dropped"], color="#ca5288")
    axes[0].set(xlabel="% of genre removed", title="Dropping Latin-8 failures biases genres", xlim=(0, 65))
    for i, pct in enumerate(genres["pct_lost_if_dropped"]):
        axes[0].text(pct + .6, i, f"{pct:.1f}%", va="center", fontsize=10)
    stages = results["normalization_stages"]
    counts = [s["latin8_incompatible_rows"] for s in stages]
    axes[1].bar(["Original", "Normalize\nUnicode", "+ optional\nе → e repair"], counts,
                color=["#ca5288", "#8d76b5", "#537d93"])
    axes[1].set(ylabel="Lyrics incompatible with Latin-8", title="Compatibility is a diagnostic, not a target", ylim=(0, max(counts) * 1.17))
    for i, count in enumerate(counts):
        axes[1].text(i, count + 900, f"{count:,}", ha="center", fontsize=10)
    if path is not None:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    started = time.monotonic()
    digest = hashlib.sha256()
    with args.input.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    df = pd.read_csv(args.input, encoding="utf-8", encoding_errors="strict")
    results, *_ = analyze(df)
    results["source"] = {"path": str(args.input), "bytes": args.input.stat().st_size,
                         "sha256": digest.hexdigest(), "strict_utf8_decode": True}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "unicode_audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_results(results, args.output / "unicode_audit.png")
    print(json.dumps({k: v for k, v in results.items() if k not in ("characters", "examples")}, indent=2))
    print(f"Completed in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
