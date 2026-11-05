I inspected all **134,967 training rows**. The most useful next steps are **more precise annotation handling, reviewing non-song records, and controlling duplicates**. This was a read-only analysis.

These are my recommendations, in priority order.

1. **Replace blanket bracket removal with selective annotation handling.**

   **62,255 lyrics contain brackets**, but their contents serve different purposes:

   | Example | Recommended treatment |
   |---|---|
   | `[Verse 1: Drake]` | Remove the performer name; retain generic section information |
   | `[Produced by …]` | Remove the credit from the lyric text |
   | `[?]` | Remove the marker and record an uncertainty flag |
   | `I was so stressed [Stressed]` | Preserve the sung response: `I was so stressed Stressed` |

   The last example actually occurs in your dataset, in **“Legend In Town The Comeback,” ID 492245**.

   Your current cleaner also flattens line breaks before deleting brackets. This can erase entire bracketed spoken passages. For example, **“Wimmin,” ID 4631890**, loses its substantive content under that approach.

   **Decision:** preserve line boundaries until annotations have been classified. Remove recognized metadata; preserve plausible sung or spoken words.

2. **Handle section labels that have no brackets.**

   **4,928 rows** contain standalone labels such as `Chorus`, `Verse 1:`, or `Intro`, which your current cleaner misses.

   Another **746 rows** contain a label and lyrics together:

   ```text
   Chorus: Walk it, talk it, hold it, solid
   ```

   Remove only the label prefix and preserve the words following it.

   **Decision:** recognize labels at line boundaries. Retain section counts or generic section markers if useful, while removing artist identities from credits.

3. **Review advertisements, promotional posts, and prose assigned to song genres.**

   These deserve attention before smaller punctuation changes:

   | ID | Current genre | Actual content |
   |---|---|---|
   | 7472449 | country | Addiction treatment advertisement |
   | 3086034 | country | Architectural modeling services article |
   | 670717 | rb | Weight loss promotion |
   | 465884 | rap | Request for feedback on a music video |
   | 90663 | rap | Charles Darwin’s *Variation Under Nature* chapter |

   **156 rows contain URLs**, but a URL alone does not establish that a record is bad: some genuine lyrics have website footers.

   **Decision:** flag suspicious records for inspection, then exclude confirmed non-song contamination. Removing the URL from an advertisement does not turn it into lyrics.

   Treat `misc` separately: novels, court opinions, and scripts may be intentional members of that class. Removing them would change the classification task.

4. **Track incomplete transcriptions explicitly.**

   I found:

   - **224 rows** beginning with a “Lyrics from … snippet” heading.
   - **5,739 rows** containing `[?]`-style unknown-word markers.

   Add diagnostic columns such as:

   ```python
   is_snippet
   unknown_span_count
   lyric_word_count
   ```

   Remove transcription notices from model text, preserve the available lyrics, and leave missing words unresolved.

   **Decision:** compare performance with and without incomplete records. A short lyric alone is insufficient evidence for exclusion.

5. **Group duplicate and closely related lyrics before validation splitting.**

   There are **178 repeated rows beyond the first occurrence of each raw lyric**, and **13 raw duplicate groups contain conflicting genre labels**.

   For example, Feist’s **“Honey Honey”** and its **BBC session** have identical lyrics but different labels: `rock` and `pop`.

   A restricted search also found **789 nonidentical candidate pairs with at least 90% character similarity**. This is a review list, not an exhaustive duplicate count.

   **Decision:** keep identical and reviewed near-identical lyrics together when splitting. Review conflicting labels rather than automatically choosing the majority genre. Different performances can legitimately have different genres despite sharing lyrics.

6. **Preserve repetition, but test how strongly it influences the model.**

   **11,643 songs** have more than half their lyric lines repeated, among songs with at least ten nonempty lines. This includes **6,934 pop songs**.

   Repetition is often chorus structure. Removing repeated lines changes the song substantially.

   For TF-IDF, start by testing:

   ```python
   TfidfVectorizer(sublinear_tf=True)
   ```

   This reduces the influence of repeated word occurrences without deleting them. Separately retain features such as line count, repeated-line ratio, and unique-word ratio. [TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

   **Decision:** preserve repetitions in the baseline; evaluate repetition caps as an experiment.

7. **Use restrained word-level cleaning.**

   After removing single-line bracket annotations:

   - **127,365 rows** still contain contraction-like forms.
   - **58,633 rows** contain informal forms such as `gonna`, `wanna`, `finna`, `tryna`, or `ain’t`.
   - **34,639 rows** still contain digits.

   These can carry lyrical style. My baseline would lowercase for TF-IDF while preserving contractions, slang, negation, profanity, and numbers.

   Treat stopword removal, spelling correction, stemming, and lemmatization as separate experiments. Scikit-learn specifically notes that common words can be informative for style classification and that stopword lists need task-specific consideration. [Stopword guidance](https://scikit-learn.org/stable/modules/feature_extraction.html#using-stop-words).

   Also review your utility functions before using them: `lemmatize_all()` returns five alternative lemmas per token, while `stemming_all()` returns a nested list and iterates over characters if given a single word.

8. **Use length thresholds for review rather than automatic deletion.**

   **570 records exceed 3,000 Unicode letter-sequence tokens; 536 are `misc`.** However, genuine long songs exist: Canibus’s **“Poet Laureate Infinity,” ID 52375**, exceeds 10,000 tokens.

   **Decision:** inspect extreme length together with repetition, vocabulary diversity, and document type. A fixed word limit would remove valid examples.

I also ran an exploratory comparison on a **30,000-row stratified sample**, using the same **5,998 validation rows**, TF-IDF, and a balanced LinearSVC. Canonically identical lyrics were grouped together.

| Processing variant | Macro F1 |
|---|---:|
| Unicode normalization; retain annotations | 0.4845 |
| Anonymize section credits; retain structure | 0.4835 |
| Remove all bracket spans | 0.4773 |
| Remove brackets and duplicate lines | 0.4764 |
| Remove brackets and English stopwords | 0.4656 |

These are single-split exploratory results, with artist overlap still present. They support starting conservatively; they do not establish a final winning pipeline.

My recommended order is **Unicode normalization → line-aware annotation processing → quality flags and reviewed exclusions → duplicate grouping → validation split → TF-IDF fitting**. Fit vocabulary, document-frequency thresholds, and IDF weights on the training portion only. [Preprocessing and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
