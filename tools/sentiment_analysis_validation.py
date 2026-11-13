"""Validate VADER and TextBlob song labels against an LLM-labelled reference.

The LLM labels are a *silver standard*: a consistent, reasonably informed
reference, not human ground truth. Use them to compare the two tools with each
other and with a reference, and spot-check the LLM itself against a small
human-labelled subset (see ``export_for_human_review``).

Requires: ``pip install anthropic`` and the ANTHROPIC_API_KEY environment variable.
Place this file next to ``sentiment_analysis.py`` in ``tools/``.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from tools.sentiment_analysis import (
    PRIMARY_COLOR,
    RANDOM_STATE,
    SENTIMENT_ORDER,
    classify_polarity,
    get_lyric_lines,
)

LLM_MODEL = "claude-sonnet-5-5"  # any Claude model string works
MAX_LYRIC_CHARS = 6000
LLM_LABELS = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive",
              "mixed": "Neutral"}  # "mixed" is kept in llm_label_raw for analysis

SYSTEM_PROMPT = """You are an annotator for a song-lyrics sentiment study.
Rate the overall emotional valence of the lyrics as expressed by the words.

Definitions:
- Positive: the lyrics overall convey positive emotion (joy, love, hope, pride,
  celebration, confidence).
- Negative: the lyrics overall convey negative emotion (sadness, anger, fear,
  hatred, despair, bitterness, violence).
- Neutral: little emotional valence, or purely descriptive/factual content.
- Mixed: substantial positive AND negative emotion that roughly balance out.

Guidelines:
- Judge the emotion the lyrics express, not whether the topic is pleasant.
  Sarcasm, irony and slang should be interpreted by meaning, not surface words.
  Profanity or aggressive style alone does not make a song negative.
- Judge only the text provided. Do not use outside knowledge about the song or artist.

Reply with ONE JSON object and nothing else, in this exact form:
{"rationale": "<one short sentence>", "label": "positive|negative|neutral|mixed",
 "score": <number from -1.0 (very negative) to 1.0 (very positive)>}"""


# --------------------------------------------------------------------------- #
# 1. Sample selection
# --------------------------------------------------------------------------- #
def select_reference_sample(df_song_primary, n_songs=300, random_state=RANDOM_STATE):
    """Draw a genre-balanced random subset, blind to the VADER/TextBlob scores.

    Selecting only by genre (never by score) keeps the reference sample from being
    biased toward songs where the tools agree or disagree.
    """
    genres = df_song_primary["tag"].unique()
    per_genre = max(1, n_songs // len(genres))
    parts = [
        group.sample(min(len(group), per_genre), random_state=random_state)
        for _, group in df_song_primary.groupby("tag")
    ]
    return pd.concat(parts).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. LLM labelling (cached, parallel, repeated runs)
# --------------------------------------------------------------------------- #
def _lyrics_for_llm(lyrics_clean):
    """Same deduplicated text the tools scored, truncated for cost control."""
    text = "\n".join(get_lyric_lines(lyrics_clean, deduplicate=True))
    return text[:MAX_LYRIC_CHARS]


def _parse_llm_reply(text):
    """Extract and validate the JSON object from the model reply."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in reply: {text[:120]!r}")
    data = json.loads(match.group(0))
    raw_label = str(data["label"]).strip().lower()
    if raw_label not in LLM_LABELS:
        raise ValueError(f"Unexpected label: {raw_label!r}")
    score = float(np.clip(float(data["score"]), -1.0, 1.0))
    return {
        "llm_label_raw": raw_label,
        "llm_label": LLM_LABELS[raw_label],
        "llm_score": score,
        "llm_rationale": str(data.get("rationale", ""))[:300],
    }


def _label_one(client, model, lyrics, temperature, max_retries):
    kwargs = {} if temperature is None else {"temperature": temperature}
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Lyrics:\n\n{lyrics}"}],
                **kwargs,
            )
            return _parse_llm_reply(response.content[0].text)
        except Exception as error:  # rate limits, malformed JSON, network...
            if attempt == max_retries - 1:
                return {"llm_error": str(error)[:200]}
            time.sleep(2 ** attempt)


def label_with_llm(
    df_ref,
    df_sample,
    cache_path="llm_labels_cache.jsonl",
    n_runs=3,
    model=LLM_MODEL,
    temperature=None,
    max_workers=8,
    max_retries=4,
    client=None,
):
    """Label every song n_runs times. Results are cached so reruns cost nothing.

    Each (song_key, run) is appended to ``cache_path`` as soon as it finishes, so an
    interrupted run resumes where it stopped. Songs are labelled from the lyrics only
    (no title/artist, no tool scores) so the LLM stays blind to everything else.
    ``temperature=None`` uses the API default, which lets repeated runs vary and so
    measures the LLM's own consistency; pass 0 for near-deterministic labels.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    cache_path = Path(cache_path)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "llm_error" not in record and record.get("model") == model:
                done[(record["song_key"], record["run"])] = record

    lyrics_by_song = df_sample.set_index("song_key")["lyrics_clean"].map(_lyrics_for_llm)
    jobs = [
        (song_key, run)
        for song_key in df_ref["song_key"]
        for run in range(n_runs)
        if (song_key, run) not in done
    ]
    print(f"{len(done)} cached, {len(jobs)} to label with {model}.")

    with cache_path.open("a", encoding="utf-8") as cache, ThreadPoolExecutor(max_workers) as pool:
        futures = {
            pool.submit(_label_one, client, model, lyrics_by_song[song_key],
                        temperature, max_retries): (song_key, run)
            for song_key, run in jobs
        }
        for future in as_completed(futures):
            song_key, run = futures[future]
            result = future.result()
            record = {"song_key": int(song_key), "run": run, "model": model, **result}
            if "llm_error" not in result:
                done[(song_key, run)] = record
            cache.write(json.dumps(record) + "\n")
            cache.flush()

    failed = [j for j in ((s, r) for s in df_ref["song_key"] for r in range(n_runs))
              if j not in done]
    if failed:
        print(f"Warning: {len(failed)} labelling calls failed; rerun to retry them.")
    return pd.DataFrame(done.values())


def aggregate_llm_runs(df_runs, tie_threshold=0.2):
    """Collapse repeated runs into one label per song (majority vote).

    Ties (e.g. three different labels) fall back to the mean score. Also returns the
    run-to-run agreement, which is a practical ceiling for how much any tool can
    agree with this reference.
    """
    def vote(group):
        counts = group["llm_label"].value_counts()
        mean_score = group["llm_score"].mean()
        top = counts[counts.eq(counts.max())].index
        label = top[0] if len(top) == 1 else classify_polarity(mean_score, tie_threshold)
        return pd.Series({
            "llm_label": label,
            "llm_score": mean_score,
            "llm_votes_agree": counts.max() / len(group),
            "llm_mixed_share": group["llm_label_raw"].eq("mixed").mean(),
            "llm_runs": len(group),
        })

    df_ref = df_runs.groupby("song_key").apply(vote).reset_index()

    pivot = df_runs.pivot(index="song_key", columns="run", values="llm_label").dropna()
    kappas = [
        cohen_kappa_score(pivot[a], pivot[b], labels=SENTIMENT_ORDER)
        for a, b in combinations(pivot.columns, 2)
    ]
    consistency = {
        "mean_pairwise_kappa": float(np.mean(kappas)) if kappas else np.nan,
        "unanimous_share": float(df_ref["llm_votes_agree"].eq(1).mean()),
        "mixed_share": float(df_ref["llm_mixed_share"].gt(0.5).mean()),
    }
    return df_ref, consistency


def build_reference(df_song_primary, df_runs):
    """Join aggregated LLM labels onto the tools' song-level scores."""
    df_llm, consistency = aggregate_llm_runs(df_runs)
    df_ref = df_song_primary.merge(df_llm, on="song_key", how="inner", validate="one_to_one")
    return df_ref, consistency


# --------------------------------------------------------------------------- #
# 3. Evaluation against the reference
# --------------------------------------------------------------------------- #
TOOL_COLUMNS = {
    "VADER": ("vader_label", "vader_compound"),
    "TextBlob": ("textblob_label", "textblob_polarity"),
}


def _bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        values.append(metric(y_true[idx], y_pred[idx]))
    return tuple(np.nanpercentile(values, [2.5, 97.5]))


def evaluate_against_reference(df_ref, reference_col="llm_label",
                               reference_score_col="llm_score", tools=None):
    """Accuracy, macro-F1, kappa and rank correlation of each tool vs the reference.

    Works for any reference (LLM labels now, human labels later). 95% bootstrap
    intervals are included because the reference sample is small.
    """
    tools = tools or TOOL_COLUMNS
    macro_f1 = lambda t, p: f1_score(t, p, labels=SENTIMENT_ORDER, average="macro",
                                     zero_division=0)
    weighted_kappa = lambda t, p: cohen_kappa_score(t, p, labels=SENTIMENT_ORDER,
                                                    weights="linear")
    y_true = df_ref[reference_col]
    rows, matrices, reports = {}, {}, {}
    for name, (label_col, score_col) in tools.items():
        y_pred = df_ref[label_col]
        f1_low, f1_high = _bootstrap_ci(y_true, y_pred, macro_f1)
        rows[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "macro_f1_ci_low": f1_low,
            "macro_f1_ci_high": f1_high,
            "weighted_kappa": weighted_kappa(y_true, y_pred),
            "spearman_vs_llm_score": spearmanr(df_ref[score_col],
                                               df_ref[reference_score_col]).statistic,
        }
        matrices[name] = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=SENTIMENT_ORDER),
            index=[f"ref {l}" for l in SENTIMENT_ORDER],
            columns=[f"pred {l}" for l in SENTIMENT_ORDER],
        )
        reports[name] = pd.DataFrame(
            classification_report(y_true, y_pred, labels=SENTIMENT_ORDER,
                                  output_dict=True, zero_division=0)
        ).T.loc[SENTIMENT_ORDER, ["precision", "recall", "f1-score", "support"]]
    return pd.DataFrame(rows).T, matrices, reports


def compare_tools_mcnemar(df_ref, reference_col="llm_label", tools=None):
    """Exact McNemar test: is one tool right more often than the other?"""
    tools = tools or TOOL_COLUMNS
    (name_a, (col_a, _)), (name_b, (col_b, _)) = list(tools.items())[:2]
    a_right = df_ref[col_a].eq(df_ref[reference_col])
    b_right = df_ref[col_b].eq(df_ref[reference_col])
    only_a, only_b = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {f"only_{name_a}_correct": only_a, f"only_{name_b}_correct": only_b,
            "p_value": p_value}


def evaluate_by_genre(df_ref, reference_col="llm_label", tools=None):
    """Macro-F1 per genre and tool (small groups: read as indicative only)."""
    tools = tools or TOOL_COLUMNS
    rows = {}
    for genre, group in df_ref.groupby("tag"):
        rows[genre] = {"songs": len(group)}
        for name, (label_col, _) in tools.items():
            rows[genre][f"{name}_accuracy"] = accuracy_score(group[reference_col],
                                                             group[label_col])
    return pd.DataFrame(rows).T


def cross_validated_threshold(df_ref, score_col, reference_col="llm_label",
                              grid=None, n_splits=5):
    """Honest threshold tuning: choose the neutral band on train folds, score on test folds.

    The notebook's fixed ±0.05 band is arbitrary, and line-averaged scores shrink
    toward zero. Picking the best threshold on the same songs it is scored on would be
    optimistic, so predictions are produced out-of-fold.
    """
    grid = np.round(grid if grid is not None else np.arange(0.0, 0.31, 0.01), 3)
    y = df_ref[reference_col].to_numpy()
    scores = df_ref[score_col].to_numpy()
    macro = lambda t, p: f1_score(t, p, labels=SENTIMENT_ORDER, average="macro",
                                  zero_division=0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out_of_fold = np.empty(len(y), dtype=object)
    chosen = []
    for train, test in cv.split(scores, y):
        best = max(grid, key=lambda t: macro(
            y[train], [classify_polarity(s, t) for s in scores[train]]))
        chosen.append(float(best))
        out_of_fold[test] = [classify_polarity(s, best) for s in scores[test]]
    return {
        "fixed_0.05_macro_f1": macro(y, [classify_polarity(s, 0.05) for s in scores]),
        "tuned_cv_macro_f1": macro(y, out_of_fold),
        "chosen_thresholds": chosen,
    }


# --------------------------------------------------------------------------- #
# 4. Plots and human spot-check
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(matrices):
    """Row-normalised confusion matrices (reference on rows) for each tool."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.5))
    for ax, (name, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        shares = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
        sns.heatmap(shares, annot=matrix, fmt="d", cmap=sns.light_palette(PRIMARY_COLOR, as_cmap=True),
                    vmin=0, vmax=1, cbar=False, ax=ax)
        ax.set_title(f"{name} vs LLM reference")
        ax.set_xlabel("Tool prediction")
        ax.set_ylabel("LLM reference")
    plt.tight_layout()
    plt.show()


def plot_score_vs_llm(df_ref, tools=None):
    """Tool score against the LLM score, to see calibration and compression."""
    tools = tools or TOOL_COLUMNS
    fig, axes = plt.subplots(1, len(tools), figsize=(5.5 * len(tools), 4.5), sharey=True)
    for ax, (name, (_, score_col)) in zip(np.atleast_1d(axes), tools.items()):
        sns.regplot(data=df_ref, x="llm_score", y=score_col, ax=ax,
                    scatter_kws={"alpha": 0.35, "color": PRIMARY_COLOR}, line_kws={"color": "red"})
        ax.set_title(name)
        ax.set_xlabel("LLM score")
        ax.set_ylabel(f"{name} score")
    plt.tight_layout()
    plt.show()


def export_for_human_review(df_ref, df_sample, path="human_review.csv", n=50,
                            random_state=RANDOM_STATE):
    """Write a blind CSV (no LLM/tool labels) to hand-label and check the LLM itself."""
    subset = df_ref.sample(min(n, len(df_ref)), random_state=random_state)
    export = subset[["song_key"]].merge(df_sample[["song_key", "lyrics_clean"]], on="song_key")
    export["lyrics_excerpt"] = export["lyrics_clean"].str.slice(0, 1200)
    export["human_label"] = ""  # fill with Negative / Neutral / Positive
    export[["song_key", "lyrics_excerpt", "human_label"]].to_csv(path, index=False)
    return Path(path).resolve()


def evaluate_llm_vs_human(df_ref, human_path="human_review.csv"):
    """Agreement of the LLM reference with your own labels on the spot-check subset."""
    human = pd.read_csv(human_path)[["song_key", "human_label"]].dropna()
    human["human_label"] = human["human_label"].str.strip().str.capitalize()
    merged = df_ref.merge(human, on="song_key", validate="one_to_one")
    return {
        "songs": len(merged),
        "accuracy": accuracy_score(merged["human_label"], merged["llm_label"]),
        "cohen_kappa": cohen_kappa_score(merged["human_label"], merged["llm_label"],
                                         labels=SENTIMENT_ORDER),
        "merged": merged,
    }