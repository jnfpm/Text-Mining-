import pandas as pd



def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # already-separated data split
    prepared = data.copy()

 """   # Impute missing titles with 'unknown'
    if "title" in prepared.columns:
        prepared["title"] = prepared["title"].fillna("unknown")

    # Handle missing values: drop rows where essential columns are missing
    essential_cols = [col for col in ["lyrics", "tag"] if col in prepared.columns]
    prepared = prepared.dropna(subset=essential_cols) """

    # Drop variations (remixes, acoustics, live, etc.)
    if "title" in prepared.columns:
        variations_pattern = r"(?i)\b(remix|acoustic|live|version|edit|instrumental|cover|stripp?ed|mix)\b"
        prepared = prepared[~prepared["title"].str.contains(variations_pattern, na=False, regex=True)]

    if "lyrics" in prepared.columns: #just in case
        # Drop duplicates based on lyrics to keep only the original
        prepared = prepared.drop_duplicates(subset=["lyrics"], keep="first")

    # Normalize text formatting while leaving blank values available for missingness checks.
    for column in ("title", "artist", "tag", "features"):
        if column in prepared.columns:
            values = prepared[column].astype("string").str.strip()
            if column == "tag":
                values = values.str.lower()
            prepared[column] = values

    if "year" in prepared.columns:
        # Parse numeric or date-like years; future and non-integral years become missing.
        year_text = prepared["year"].astype("string").str.strip()
        numeric_year = pd.to_numeric(year_text, errors="coerce")
        date_year = pd.to_datetime(year_text, errors="coerce").dt.year
        parsed_year = numeric_year.fillna(date_year)
        valid_year = parsed_year.ge(1900) & parsed_year.le(2026) & parsed_year.mod(1).eq(0)
        prepared["year"] = parsed_year.where(valid_year).astype("Int64")

    if "views" in prepared.columns:
        # View counts cannot be negative or non-numeric.
        views = pd.to_numeric(prepared["views"], errors="coerce")
        prepared["views"] = views.where(views >= 0)

    return prepared
