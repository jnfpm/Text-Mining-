"""Helper functions for the lyrics sentiment analysis notebook."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from nrclex import NRCLex
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


RANDOM_STATE = 42
LANGUAGE_CONFIDENCE = 0.80
PRIMARY_COLOR = "#4166c4"
SECONDARY_COLOR = "#597edb"
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
EMOTION_NAMES = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

DetectorFactory.seed = RANDOM_STATE
plt.rcParams.update(
    {"axes.titlecolor": SECONDARY_COLOR, "axes.titleweight": "normal"}
)

SECTION_TAG = re.compile(r"^\s*\[[^\]]+\]\s*$")
SCRAPE_NOISE = re.compile(
    r"^\s*(?:(?:\d+\s+)?embed|lyrics(?:\s+by\b.*)?|you might also like|"
    r"contributors?\b.*|written by\b.*|produced by\b.*|"
    r"translated by\b.*|songwriters?\s*:.*|(?:see|read) more\b.*)\s*$",
    flags=re.IGNORECASE,
)


def load_data(data_path):
    """Load lyrics while preserving the original text."""
    df_train = pd.read_csv(data_path, encoding="utf-8")
    required_columns = {"id", "title", "tag", "artist", "year", "lyrics"}
    missing_columns = required_columns.difference(df_train.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    return df_train


def clean_lyrics(text):
    """Remove section labels and common scraping noise from lyrics."""
    if pd.isna(text):
        return ""

    cleaned_lines = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or SECTION_TAG.fullmatch(line) or SCRAPE_NOISE.fullmatch(line):
            continue
        if not re.search(r"\w", line, flags=re.UNICODE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def detect_lyric_language(text):
    """Return the most likely language and its confidence."""
    if len(re.findall(r"\w+", text)) < 5:
        return "unknown", 0.0
    try:
        prediction = detect_langs(text)[0]
        return prediction.lang, float(prediction.prob)
    except LangDetectException:
        return "unknown", 0.0


def get_lyric_lines(text, deduplicate=False):
    """Split lyrics into lines, optionally keeping only the first repeated line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not deduplicate:
        return lines

    unique_lines = []
    seen = set()
    for line in lines:
        normalized = line.casefold()
        if normalized not in seen:
            seen.add(normalized)
            unique_lines.append(line)
    return unique_lines


def prepare_sample(
    df_train,
    sample_fraction=0.02,
    random_state=RANDOM_STATE,
    language_confidence=LANGUAGE_CONFIDENCE,
):
    """Sample by genre, clean lyrics, and retain confident English predictions."""
    df_candidates = (
        df_train.groupby("tag", group_keys=False)
        .sample(frac=sample_fraction, random_state=random_state)
        .copy()
    )
    df_candidates["lyrics_clean"] = df_candidates["lyrics"].map(clean_lyrics)
    df_candidates = df_candidates[
        df_candidates["lyrics_clean"].str.strip().ne("")
    ].copy()

    predictions = df_candidates["lyrics_clean"].map(detect_lyric_language)
    df_candidates[["language", "language_confidence"]] = pd.DataFrame(
        predictions.tolist(), index=df_candidates.index
    )
    language_counts = (
        df_candidates["language"]
        .value_counts(dropna=False)
        .rename_axis("language")
        .to_frame("songs")
    )
    df_sample = df_candidates[
        df_candidates["language"].eq("en")
        & df_candidates["language_confidence"].ge(language_confidence)
    ].copy()
    if df_sample.empty:
        raise ValueError(
            "No high-confidence English lyrics remain; inspect language predictions."
        )

    df_sample = df_sample.reset_index(drop=True)
    df_sample["song_key"] = np.arange(len(df_sample))
    return df_candidates, df_sample, language_counts


def classify_polarity(score, threshold=0.05):
    """Convert a polarity score to a negative, neutral, or positive label."""
    if score <= -threshold:
        return "Negative"
    if score >= threshold:
        return "Positive"
    return "Neutral"


def score_lyric_lines(df_sample):
    """Score each lyric line with VADER and TextBlob for both line versions."""
    line_records = []
    for row in df_sample.itertuples(index=False):
        for version, deduplicate in (
            ("all_lines", False),
            ("deduplicated", True),
        ):
            lines = get_lyric_lines(row.lyrics_clean, deduplicate=deduplicate)
            line_count = len(lines)
            for line_index, line_text in enumerate(lines):
                line_records.append(
                    {
                        "song_key": row.song_key,
                        "line_version": version,
                        "line_index": line_index,
                        "line_position": line_index / max(line_count - 1, 1),
                        "line_text": line_text,
                    }
                )

    df_lines = pd.DataFrame.from_records(line_records)
    if df_lines.empty:
        raise ValueError("No lyric lines were available for scoring.")

    sentiment_analyzer = SentimentIntensityAnalyzer()
    vader_results = df_lines["line_text"].map(
        sentiment_analyzer.polarity_scores
    )
    df_lines[["vader_neg", "vader_neu", "vader_pos", "vader_compound"]] = (
        pd.DataFrame(vader_results.tolist(), index=df_lines.index)
    )
    textblob_results = df_lines["line_text"].map(
        lambda text: TextBlob(text).sentiment
    )
    df_lines[["textblob_polarity", "textblob_subjectivity"]] = pd.DataFrame(
        [(result.polarity, result.subjectivity) for result in textblob_results],
        index=df_lines.index,
    )
    df_lines["vader_label"] = df_lines["vader_compound"].map(classify_polarity)
    df_lines["textblob_label"] = df_lines["textblob_polarity"].map(
        classify_polarity
    )

    song_scores = (
        df_lines.groupby(["song_key", "line_version"], as_index=False)
        .agg(
            line_count=("line_text", "size"),
            vader_compound=("vader_compound", "mean"),
            textblob_polarity=("textblob_polarity", "mean"),
            textblob_subjectivity=("textblob_subjectivity", "mean"),
        )
    )
    metadata_columns = ["song_key", "id", "title", "tag", "artist", "year"]
    df_song = song_scores.merge(
        df_sample[metadata_columns],
        on="song_key",
        how="left",
        validate="many_to_one",
    )
    df_song["vader_label"] = df_song["vader_compound"].map(classify_polarity)
    df_song["textblob_label"] = df_song["textblob_polarity"].map(
        classify_polarity
    )
    return df_lines, df_song


def get_sentiment_agreement(df_song_primary):
    """Compare song-level distributions, scores, and sentiment labels."""
    distribution = pd.concat(
        {
            "VADER": df_song_primary["vader_label"].value_counts(normalize=True),
            "TextBlob": df_song_primary["textblob_label"].value_counts(
                normalize=True
            ),
        },
        axis=1,
    ).reindex(SENTIMENT_ORDER, fill_value=0)

    agreement = {
        "spearman_correlation": df_song_primary[
            ["vader_compound", "textblob_polarity"]
        ]
        .corr(method="spearman")
        .iloc[0, 1],
        "label_agreement": df_song_primary["vader_label"]
        .eq(df_song_primary["textblob_label"])
        .mean(),
        "cohen_kappa": cohen_kappa_score(
            df_song_primary["vader_label"],
            df_song_primary["textblob_label"],
            labels=SENTIMENT_ORDER,
        ),
    }
    return distribution, agreement


def summarize_metadata(df_song_primary):
    """Return genre, artist, and decade sentiment summaries."""
    genre_summary = (
        df_song_primary.groupby("tag")
        .agg(
            song_count=("song_key", "nunique"),
            vader_mean=("vader_compound", "mean"),
            textblob_mean=("textblob_polarity", "mean"),
            positive_share=(
                "vader_label",
                lambda values: values.eq("Positive").mean(),
            ),
        )
        .sort_values("vader_mean")
    )

    artist_summary = (
        df_song_primary.groupby("artist")
        .agg(
            song_count=("song_key", "nunique"),
            vader_mean=("vader_compound", "mean"),
            textblob_mean=("textblob_polarity", "mean"),
        )
    )
    artist_summary = artist_summary[artist_summary["song_count"] >= 3]

    valid_years = df_song_primary[
        df_song_primary["year"].between(1900, 2024)
    ].copy()
    valid_years["decade"] = (valid_years["year"] // 10) * 10
    decade_summary = (
        valid_years.groupby("decade")
        .agg(
            song_count=("song_key", "nunique"),
            vader_median=("vader_compound", "median"),
            textblob_median=("textblob_polarity", "median"),
        )
    )
    decade_summary = decade_summary[decade_summary["song_count"] >= 10]
    return genre_summary, artist_summary, decade_summary


def summarize_song_arc(df_lines):
    """Average sentiment by song position, weighting every song equally."""
    arc_lines = df_lines[df_lines["line_version"].eq("deduplicated")].copy()
    arc_lines["position_bin"] = pd.cut(
        arc_lines["line_position"],
        bins=np.linspace(0, 1, 11),
        labels=np.arange(10),
        include_lowest=True,
    )
    song_arc = (
        arc_lines.groupby(["song_key", "position_bin"], observed=True)
        .agg(
            vader_compound=("vader_compound", "mean"),
            textblob_polarity=("textblob_polarity", "mean"),
        )
        .reset_index()
    )
    return song_arc.groupby("position_bin", observed=True)[
        ["vader_compound", "textblob_polarity"]
    ].mean()


def summarize_sentiment_over_time(df_song_primary, min_songs=8):
    """Summarize valid release years and smooth over calendar years."""
    valid_years = df_song_primary[
        df_song_primary["year"].between(1900, 2024)
    ]
    year_summary_all = (
        valid_years.groupby("year")
        .agg(
            song_count=("song_key", "nunique"),
            vader_median=("vader_compound", "median"),
            textblob_median=("textblob_polarity", "median"),
        )
        .reindex(range(1900, 2025))
    )
    adequately_sampled = year_summary_all["song_count"].ge(min_songs)
    year_summary = year_summary_all[adequately_sampled].copy()
    rolling_median = (
        year_summary_all["vader_median"]
        .where(adequately_sampled)
        .rolling(window=5, center=True, min_periods=1)
        .median()
    )
    year_summary["vader_rolling_median"] = rolling_median.loc[
        year_summary.index
    ]
    excluded_years = len(df_song_primary) - len(valid_years)
    return year_summary, excluded_years


def summarize_themes_and_emotions(df_sample, df_song_primary):
    """Find TF-IDF topics and summarize NRC emotions for sampled songs."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_features=3000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    tfidf = vectorizer.fit_transform(df_sample["lyrics_clean"])
    topic_count = min(6, max(2, len(df_sample) // 20))
    nmf = NMF(
        n_components=topic_count,
        random_state=RANDOM_STATE,
        max_iter=300,
    )
    topic_weights = nmf.fit_transform(tfidf)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    topic_terms = {
        topic_id: ", ".join(feature_names[np.argsort(component)[-8:][::-1]])
        for topic_id, component in enumerate(nmf.components_)
    }

    df_themes = df_song_primary.copy()
    df_themes["topic_id"] = topic_weights.argmax(axis=1)
    df_themes["topic_terms"] = df_themes["topic_id"].map(topic_terms)
    theme_summary = (
        df_themes.groupby(["topic_id", "topic_terms"])
        .agg(
            song_count=("song_key", "nunique"),
            vader_mean=("vader_compound", "mean"),
            textblob_mean=("textblob_polarity", "mean"),
        )
        .sort_values("song_count", ascending=False)
    )

    emotion_reader = NRCLex()
    emotion_rows = []
    for text in df_sample["lyrics_clean"]:
        emotion_reader.load_raw_text(text)
        emotion_rows.append(emotion_reader.affect_frequencies.copy())
    emotion_scores = pd.DataFrame(
        emotion_rows, index=df_sample["song_key"]
    ).reindex(columns=EMOTION_NAMES, fill_value=0)
    emotion_scores.index.name = "song_key"
    emotion_scores = emotion_scores.join(
        df_sample.set_index("song_key")[["tag"]], how="left"
    )
    emotion_summary = emotion_scores.groupby("tag")[EMOTION_NAMES].mean()
    return theme_summary, emotion_summary


def summarize_subjectivity(df_song_primary):
    """Return descriptive statistics for TextBlob subjectivity."""
    return df_song_primary["textblob_subjectivity"].describe(
        percentiles=[0.25, 0.5, 0.75]
    )


def validate_cleaning():
    """Check that cleaning removes scrape noise but preserves sentiment cues."""
    example = clean_lyrics(
        "[Chorus]\nI DON'T want to go!!!\nWritten by Sample Writer\n"
        "[Verse 1]\nMaybe tomorrow...\nEmbed"
    )
    assert "[Chorus]" not in example
    assert "Written by" not in example
    assert "DON'T" in example and "!!!" in example
    assert get_lyric_lines(
        "Stay!\nStay!\nGo.", deduplicate=True
    ) == ["Stay!", "Go."]
    return example


def score_validation_examples():
    """Score short example lines to inspect both tools' polarity behavior."""
    examples = [
        "I love this song!",
        "I do not love this song.",
        "I NEVER said I hated you!!!",
        "I'm gonna be alright...",
    ]
    sentiment_analyzer = SentimentIntensityAnalyzer()
    rows = [
        {
            "line": line,
            "vader_compound": sentiment_analyzer.polarity_scores(line)["compound"],
            "textblob_polarity": TextBlob(line).sentiment.polarity,
        }
        for line in examples
    ]
    return pd.DataFrame(rows)


def summarize_chorus_sensitivity(df_song):
    """Measure how retaining repeated lines changes scores and labels."""
    sensitivity = df_song.pivot(
        index="song_key",
        columns="line_version",
        values="vader_compound",
    ).dropna()
    sensitivity["absolute_score_change"] = (
        sensitivity["all_lines"] - sensitivity["deduplicated"]
    ).abs()
    sensitivity["label_changed"] = (
        sensitivity["all_lines"].map(classify_polarity)
        != sensitivity["deduplicated"].map(classify_polarity)
    )
    return sensitivity


def plot_sentiment_distribution(distribution):
    """Plot song-level VADER and TextBlob sentiment shares."""
    distribution.plot(
        kind="bar",
        figsize=(8, 4),
        color=[PRIMARY_COLOR, SECONDARY_COLOR],
    )
    plt.title("Song-level sentiment distribution by tool")
    plt.ylabel("Share of songs")
    plt.xlabel("Sentiment label")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_genre_sentiment(genre_summary):
    """Plot mean song-level VADER sentiment by genre."""
    sns.barplot(
        data=genre_summary.reset_index(),
        x="vader_mean",
        y="tag",
        color=SECONDARY_COLOR,
    )
    plt.title("Mean song sentiment by genre")
    plt.xlabel("Mean VADER compound score")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.show()


def plot_song_arc(arc_summary):
    """Plot average sentiment from the first to the last lyric line."""
    arc_summary.plot(marker="o", figsize=(9, 4))
    plt.title("Average sentiment across the song")
    plt.xlabel("Relative line position (first to last)")
    plt.ylabel("Mean sentiment score")
    plt.xticks(
        range(10),
        [f"{i * 10}-{(i + 1) * 10}%" for i in range(10)],
        rotation=35,
    )
    plt.tight_layout()
    plt.show()


def plot_sentiment_over_time(year_summary):
    """Plot annual and smoothed VADER sentiment."""
    year_summary[["vader_median", "vader_rolling_median"]].plot(
        figsize=(10, 4),
        alpha=0.8,
    )
    plt.title("VADER sentiment over time")
    plt.xlabel("Release year")
    plt.ylabel("Median compound score")
    plt.tight_layout()
    plt.show()


def plot_emotions(emotion_summary):
    """Plot average NRC emotion proportions across genres."""
    emotion_summary.mean().sort_values().plot(
        kind="barh",
        color=SECONDARY_COLOR,
        figsize=(7, 4),
    )
    plt.title("Mean NRC emotion proportions across sampled songs")
    plt.xlabel("Mean lexicon proportion")
    plt.tight_layout()
    plt.show()


def plot_subjectivity(df_song_primary):
    """Plot song-level TextBlob subjectivity."""
    sns.histplot(
        df_song_primary["textblob_subjectivity"],
        bins=30,
        color=SECONDARY_COLOR,
    )
    plt.title("Song-level TextBlob subjectivity")
    plt.xlabel("Mean line subjectivity")
    plt.ylabel("Number of songs")
    plt.tight_layout()
    plt.show()
