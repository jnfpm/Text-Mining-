# Text Mining — song genre classification

The dataset exploration is in [EDA.ipynb](EDA.ipynb).

## Unicode lyrics

`data/train.csv` is valid UTF-8. **55,494 of 134,967 lyrics** cannot be
represented in Latin-8 (ISO-8859-14), which is a limited legacy repertoire.
That is not evidence that those songs are corrupt or non-English. In 40,724 of
those rows, the incompatible characters are only punctuation or separators.
Dropping all incompatible rows would remove 31.2% of pop but 52.8% of rap.

[text_quality.py](text_quality.py) provides the standalone normalizer and audit:

```python
from text_quality import normalize_lyrics, audit_lyrics

# Preserve source lyrics, accents, other scripts, emoji, case and verse breaks.
df_train['lyrics_normalized'] = df_train['lyrics'].map(normalize_lyrics)
row_diagnostics, character_inventory = audit_lyrics(df_train['lyrics'])

# Optional, corpus-specific: Cyrillic е inside otherwise Latin words.
# Requires >=90% Latin letters in the document and >=50% in the word.
df_train['lyrics_normalized'] = df_train['lyrics'].map(
    lambda text: normalize_lyrics(text, repair_mixed_script=True)
)
```

The default uses NFC and explicit punctuation/spacing rules. It removes selected
scraping controls, preserves meaningful joiners, and recovers recognizable C1
punctuation. U+0085 is treated as a Unicode line ending; legacy ellipsis usage is
ambiguous. It leaves suspected mojibake and U+FFFD visible for review. Missing
scalars pass through; other non-string inputs raise `TypeError`. The optional
repair defaults to **off**; genuine Cyrillic words are preserved. The EDA enables
the heuristic explicitly in its derived preprocessing column.

`utils.clean_lyrics()` now preserves Unicode letters, marks and apostrophes;
other punctuation becomes word boundaries. It still removes bracketed section
labels and lowercases for word models. `tokenize_lyrics()` retains ambiguous
possessive `'s`. Its NLTK dependencies/resources remain required.

Reproduce the full audit without modifying the data:

```bash
python3 analyze_unicode.py --input data/train.csv --output reports
python3 -m unittest discover -s tests -v
```

The audit produces [aggregate results and character inventory](reports/unicode_audit.json)
and a [diagnostic figure](reports/unicode_audit.png). It checks every row for
idempotence, NFC, new empty lyrics and normalization-induced duplicates, and
includes short review examples. Runtime is about two minutes on this machine.

| Policy | Lyrics still incompatible with Latin-8 |
| --- | ---: |
| Original | 55,494 |
| Unicode normalization | 14,648 |
| Plus optional contextual `е` repair | 1,763 |

Remaining Unicode is retained intentionally. Corruption flags are heuristics;
script ratios use Unicode character names rather than full Script properties
and do not determine language. No rows are removed. Normalization
reveals 30 additional duplicate rows and 24 normalized lyric groups with
conflicting genres; review these and group duplicate lyrics when splitting for
validation. Model accuracy has not been evaluated for either policy. Use the
same chosen normalization for training, validation, test and inference.

References: [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html),
[Unicode normalization guidance](https://www.unicode.org/faq/normalization.html),
[Unicode confusable characters](https://www.unicode.org/reports/tr39/).
