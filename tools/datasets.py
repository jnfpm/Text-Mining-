from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tools.lyric_cleaning import (
    clean_brackets,
    clean_unbracketed_headers,
    deduplicate_lines,
    remove_punctuation,
)
from tools.text_quality import normalize_lyrics


def _require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}.")


def _as_series(values, name):
    if isinstance(values, (str, bytes)) or pd.api.types.is_scalar(values):
        raise ValueError(f"{name} must be one-dimensional.")
    try:
        result = pd.Series(values).reset_index(drop=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be one-dimensional.") from error
    if not result.map(pd.api.types.is_scalar).all():
        raise ValueError(f"{name} must contain scalar values only.")
    return result


def _validate_ids(ids, name):
    values = _as_series(ids, f"{name} IDs")
    blank = values.map(lambda value: isinstance(value, str) and not value.strip())
    if values.isna().any() or blank.any() or values.duplicated().any():
        raise ValueError(f"{name} must have unique, non-missing IDs.")
    return values


def _validate_labels(labels, name):
    valid = labels.map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not valid.all():
        raise ValueError(f"{name} must contain non-missing, non-empty string labels.")


def load_datasets(data_dir):
    data_dir = Path(data_dir)
    frames = [
        pd.read_csv(data_dir / filename, encoding="utf-8", encoding_errors="strict")
        for filename in ("train.csv", "test.csv", "sample_submission.csv")
    ]
    train, test, template = frames
    for frame, columns, name in (
        (train, ("id", "lyrics", "tag"), "train.csv"),
        (test, ("id", "lyrics"), "test.csv"),
        (template, ("id", "tag"), "sample_submission.csv"),
    ):
        _require_columns(frame, columns, name)
        _validate_ids(frame["id"], name)
        if frame.empty:
            raise ValueError(f"{name} must contain at least one row.")
    _validate_labels(train["tag"], "train.csv tag")
    if set(test["id"]) != set(template["id"]):
        raise ValueError("Test IDs and submission-template IDs must match exactly.")
    return train, test.drop(columns="tag", errors="ignore"), template


def duplicate_keys(frame):
    _require_columns(frame, ("id", "lyrics"), "lyrics data")
    ids = _validate_ids(frame["id"], "lyrics data")
    keys = []
    for song_id, lyrics in zip(ids, frame["lyrics"]):
        text = normalize_lyrics(lyrics, repair_mixed_script=True)
        if isinstance(text, str):
            text = clean_brackets(text, remove_section_headers=True)
            text = clean_unbracketed_headers(text, remove_section_headers=True)
            text = remove_punctuation(text.casefold())
            text = deduplicate_lines(text)
            text = " ".join(text.split())
        if isinstance(text, str) and text:
            keys.append("lyrics:" + text)
        else:
            keys.append(f"empty:{type(song_id).__name__}:{song_id!r}")
    return pd.Series(keys, index=frame.index, name="duplicate_key", dtype="string")


def split_training_data(train, validation_size=0.2, random_state=39):
    _require_columns(train, ("id", "lyrics", "tag"), "training data")
    _validate_labels(train["tag"], "Training tags")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    groups = duplicate_keys(train)
    classes = set(train["tag"])
    if len(classes) < 2:
        raise ValueError("Training data must contain at least two classes.")
    n_splits = max(2, round(1 / min(validation_size, 1 - validation_size)))
    if groups.nunique() < n_splits:
        raise ValueError(
            f"This validation_size requires at least {n_splits} independent lyric "
            "groups. Add more groups or choose a larger held-out fraction."
        )
    class_groups = pd.DataFrame({
        "tag": train["tag"].to_numpy(),
        "group": groups.to_numpy(),
    }).groupby("tag")["group"].nunique()
    if class_groups.lt(2).any():
        rare = ", ".join(map(str, class_groups[class_groups.lt(2)].index))
        raise ValueError(
            f"Each class needs at least two independent lyric groups to appear "
            f"in both partitions. Insufficient groups for: {rare}."
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    candidates = []
    try:
        for training, validation in splitter.split(train, train["tag"], groups):
            if validation_size > 0.5:
                training, validation = validation, training
            if (
                set(train.iloc[training]["tag"]) == classes
                and set(train.iloc[validation]["tag"]) == classes
            ):
                distance = abs(len(validation) / len(train) - validation_size)
                candidates.append((distance, training, validation))
    except ValueError as error:
        raise ValueError(
            "Cannot create this grouped split. Add more examples per class or "
            "choose a larger held-out fraction."
        ) from error
    if not candidates:
        raise ValueError(
            "Cannot place every class in both partitions while keeping duplicate "
            "lyrics together. Add independent lyric groups or change validation_size."
        )
    _, training, validation = min(candidates, key=lambda candidate: candidate[0])
    return train.iloc[training].copy(), train.iloc[validation].copy()


def training_keep_mask(prepared_frame, y, remove_duplicates=True):
    column = "lyrics_processed" if "lyrics_processed" in prepared_frame else "lyrics"
    _require_columns(prepared_frame, (column,), "prepared training data")
    labels = _as_series(y, "Training targets")
    if len(labels) != len(prepared_frame):
        raise ValueError("Training targets and prepared rows must have equal length.")
    if labels.isna().any():
        raise ValueError("Training targets must not be missing.")
    keep, seen = [], set()
    for text, label in zip(prepared_frame[column], labels):
        if not isinstance(text, str):
            if pd.api.types.is_scalar(text) and pd.isna(text):
                keep.append(True)
                continue
            raise TypeError("Prepared lyrics must be strings or missing values.")
        if not text.strip() or not remove_duplicates:
            keep.append(True)
            continue
        key = (text, label)
        keep.append(key not in seen)
        seen.add(key)
    return pd.Series(keep, dtype=bool).to_numpy()


def build_submission(test_ids, predictions, template):
    _require_columns(template, ("id", "tag"), "submission template")
    ids = _validate_ids(test_ids, "Test data")
    template_ids = _validate_ids(template["id"], "Submission template")
    predicted = _as_series(predictions, "Predictions")
    if len(predicted) != len(ids):
        raise ValueError("Predictions and test IDs must have equal length.")
    blank = predicted.map(lambda value: isinstance(value, str) and not value.strip())
    if predicted.isna().any() or blank.any():
        raise ValueError("Predictions must not contain missing or empty labels.")
    if set(ids) != set(template_ids):
        raise ValueError("Test IDs and submission-template IDs must match exactly.")
    by_id = pd.Series(predicted.to_numpy(), index=ids.to_numpy())
    result = template[["id"]].reset_index(drop=True).copy()
    result["tag"] = result["id"].map(by_id)
    return result
