


import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean-only preprocessing for dataset structure.

    Keep metadata corrections like artist/year imputation for a later stage,
    after the split, because they depend on external information from Genius.
    """
    prepared = data.copy()

    # 1) Remove obvious duplicate/alternative versions that are not original tracks.
    if "title" in prepared.columns:
        variations_pattern = r"(?i)\b(remix|acoustic|live|version|edit|instrumental|cover|stripp?ed|mix)\b"
        prepared = prepared[~prepared["title"].str.contains(variations_pattern, na=False, regex=True)]

    # 2) Remove exact duplicate lyrics, keeping the first occurrence.
    if "lyrics" in prepared.columns:
        prepared = prepared.drop_duplicates(subset=["lyrics"], keep="first")

    # 3) Normalize string columns without altering semantic metadata values.
    for column in ("title", "artist", "tag", "features"):
        if column in prepared.columns:
            values = prepared[column].astype("string").str.strip()
            if column == "tag":
                values = values.str.lower()
            prepared[column] = values

    return prepared
