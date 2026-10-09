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

from utils import get_info_batch, artist_from_title


def fix_metadata(df):
    df = df.copy()

    if "song_id" in df.columns:
        # 1) corrigir artistas do tipo Genius English Translations
        if "artist" in df.columns and "title" in df.columns:
            mask = df["artist"].fillna("").str.contains("Genius|Translation", case=False, na=False)

            ids = df.loc[mask, "song_id"].dropna().unique()
            if len(ids) > 0:
                meta = get_info_batch(ids, artist=True, release_date=True).reset_index()
                meta = meta.rename(columns={"id": "song_id"})
                df = df.merge(meta, on="song_id", how="left", suffixes=("", "_api"))

                df.loc[mask, "artist"] = df.loc[mask, "artist_api"].combine_first(
                    df.loc[mask, "title"].map(artist_from_title)
                )

    # 2) corrigir anos inferiores a 1900
    if "year" in df.columns and "song_id" in df.columns:
        invalid_year_mask = pd.to_numeric(df["year"], errors="coerce") < 1900

        ids = df.loc[invalid_year_mask, "song_id"].dropna().unique()
        if len(ids) > 0:
            meta = get_info_batch(ids, release_date=True).reset_index()
            meta = meta.rename(columns={"id": "song_id"})
            df = df.merge(meta, on="song_id", how="left", suffixes=("", "_api"))

            release_year = pd.to_datetime(df["release_date_api"], errors="coerce").dt.year
            df.loc[invalid_year_mask, "year"] = release_year.loc[invalid_year_mask]

    return df
