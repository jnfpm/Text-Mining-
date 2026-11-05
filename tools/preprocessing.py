


import re

import pandas as pd
from tools.utils import artist_from_title, get_info_batch, median_years


#_____ BASE PREPROCESSING FUNCTIONS/ PIPELINE _____#

##Functions that fix inconsistencies

def fix_all_translation_artists(df):
    """Return a copy with Genius translation-account artists inferred from titles.

    Rows whose artist cannot be extracted from the title are left unchanged.
    """
    result = df.copy()
    translation_artist_mask = result['artist'].astype('string').str.contains(
        r'\bgenius\b.*\b(?:translat\w*|trad\w*|romaniz\w*|romanis\w*|transliter\w*)\b',
        case=False,
        na=False,
        regex=True,
    )
    inferred_artists = result.loc[translation_artist_mask, 'title'].fillna('').map(
        artist_from_title
    )
    replace_mask = inferred_artists.notna() & inferred_artists.ne('')
    result.loc[inferred_artists.index[replace_mask], 'artist'] = inferred_artists.loc[
        replace_mask
    ]

    return result


def correct_pre_years(treshold, data, split_name, suspicious_year_threshold=1800):
    """Correct pre-threshold years and classify Genius lookup results by ID.

    Years below ``suspicious_year_threshold`` and rows with no usable API year
    are set to ``pd.NA``. Suspicious and not-found IDs are returned separately.
    """
    data = data.copy()
    numeric_year = pd.to_numeric(data['year'], errors='coerce')
    mask = numeric_year < treshold
    ids = data.loc[mask, 'id']

    print(f"{split_name}: {len(ids)} rows with years before {treshold}")
    if ids.empty:
        return data, [], [], []

    info = get_info_batch(ids, release_date=True)
    fetched_years = pd.to_numeric(
        info['release_date'].astype('string').str.extract(r'(\d{4})', expand=False),
        errors='coerce'
    )

    replacements = data.loc[mask, 'id'].map(fetched_years)
    has_year = replacements.notna()
    suspicious = has_year & replacements.lt(suspicious_year_threshold)
    resolved = has_year & ~suspicious
    not_found = ~has_year

    data.loc[replacements.index[resolved], 'year'] = replacements[resolved].astype(int)
    clear_year = suspicious | not_found
    if clear_year.any():
        data['year'] = data['year'].astype('object')
        data.loc[replacements.index[clear_year], 'year'] = pd.NA

    resolved_ids = data.loc[replacements.index[resolved], 'id'].tolist()
    suspicious_ids = data.loc[replacements.index[suspicious], 'id'].tolist()
    not_found_ids = data.loc[replacements.index[not_found], 'id'].tolist()

    print(f"{split_name}: accepted {resolved.sum()} years; "
          f"{suspicious.sum()} suspicious; {not_found.sum()} not found")
    return data, resolved_ids, suspicious_ids, not_found_ids

## Duplicates

def remove_duplicated_lyrics(x_train, y_train, x_test, y_test):
    """Remove duplicate lyrics from train while leaving test unchanged.

    Within a repeated train-lyrics group, prefer a title without version clues
    such as ``remix``, ``cover``, ``instrumental`` or ``live``. When candidates
    have the same title score, prefer the earliest known year, then the first
    row. This is a metadata heuristic and cannot identify every original
    recording reliably.

    After within-train deduplication, remove train rows whose lyrics also occur
    in test. Inputs and outputs retain their original row indexes and target
    types; targets remain aligned with their feature rows.
    """
    for name, features, target in (
        ('train', x_train, y_train),
        ('test', x_test, y_test),
    ):
        if 'lyrics' not in features.columns:
            raise ValueError(f"x_{name} must contain a 'lyrics' column.")
        if len(features) != len(target):
            raise ValueError(
                f"x_{name} and y_{name} must have the same number of rows."
            )

    variant_cues = (
        (re.compile(r'\b(?:cover|tribute|karaoke|instrumental)\b', re.I), 100),
        (re.compile(r'\b(?:remix|re[\s-]?mix|remixed|nightcore|sped[\s-]?up|slowed(?:\s+down)?)\b', re.I), 90),
        (re.compile(r'\b(?:live|acoustic|demo|alternate\s+take|re[\s-]?recorded|re[\s-]?recording)\b', re.I), 60),
        (re.compile(r'\b(?:session|piano|orchestral)\s+version\b', re.I), 40),
        (re.compile(r'\b(?:radio|clean|extended|club|dance)\s+(?:edit|mix|version)\b', re.I), 25),
        (re.compile(r'\b(?:remaster(?:ed)?|edit|mix)\b', re.I), 10),
    )

    def lyric_key(value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    def title_score(title):
        title = title if isinstance(title, str) else ''
        return sum(weight for pattern, weight in variant_cues if pattern.search(title))

    train_groups = {}
    for position, lyric in enumerate(x_train['lyrics'].tolist()):
        key = lyric_key(lyric)
        if key is not None:
            train_groups.setdefault(key, []).append(position)

    keep_train = [True] * len(x_train)
    years = (
        pd.to_numeric(x_train['year'], errors='coerce').tolist()
        if 'year' in x_train.columns
        else [float('nan')] * len(x_train)
    )

    for positions in train_groups.values():
        if len(positions) < 2:
            continue

        def retention_rank(position):
            year = years[position]
            return (
                title_score(x_train.iloc[position].get('title')),
                year if pd.notna(year) else float('inf'),
                position,
            )

        retained_position = min(positions, key=retention_rank)
        for position in positions:
            if position != retained_position:
                keep_train[position] = False

    test_lyrics = {
        key for key in (lyric_key(value) for value in x_test['lyrics'].tolist())
        if key is not None
    }
    for position, lyric in enumerate(x_train['lyrics'].tolist()):
        if keep_train[position] and lyric_key(lyric) in test_lyrics:
            keep_train[position] = False

    retained_positions = [
        position for position, keep in enumerate(keep_train) if keep
    ]
    return (
        x_train.iloc[retained_positions].copy(),
        y_train.iloc[retained_positions].copy(),
        x_test.copy(),
        y_test.copy(),
    )

##Filling in Nan

def imputing_years(dataset, training_dataset):
    """Impute missing years using training artist medians and an overall fallback.

    Missing or unknown artist medians fall back to the overall median year from
    ``training_dataset``. If training data contains no usable years, unresolved
    values remain missing. The input dataset is not modified.
    """
    required_columns = {'artist', 'year'}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"dataset must contain columns: {', '.join(sorted(missing_columns))}."
        )

    artist_medians = median_years(training_dataset)
    training_years = pd.to_numeric(training_dataset['year'], errors='coerce')
    overall_median = training_years.median()

    result = dataset.copy()
    artists = result['artist'].astype('string').str.strip().replace('', pd.NA)
    years = pd.to_numeric(result['year'], errors='coerce')

    artist_imputations = artists.map(artist_medians)
    result['year'] = years.fillna(artist_imputations).fillna(overall_median)
    return result


#_____ PREPROCESSING FOR GENRE CLASSIFICATION _____#

import os
import time
import nltk
import pandas as pd
import requests
from dotenv import load_dotenv
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer

def ensure_nltk_data():
    """Download the corpora used by the text-preprocessing functions if missing."""
    required = ['punkt', 'stopwords', 'wordnet']
    for resource in required:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)


ensure_nltk_data()

load_dotenv(override=True)   # override=True: re-read .env even if the variable was already loaded earlier in this kernel
token = os.getenv('GENIUS_API_KEY')

# basic functions that will probably be needed to clean/preprocess the Lyrics

def removing_stopwords(lyrics):
    '''Remove stopwords from the lyrics'''
    stopwords = nltk.corpus.stopwords.words('english') #defines a list of stopwords in english
    lyrics = [word for word in lyrics if word not in stopwords]
    return lyrics

def clean_lyrics(lyrics):
    '''Remove unwanted characters and patterns from the lyrics
    Not removing words between parentheses because they usually represent choruses example: "I need you (need you)".'''
    
    lyrics = unidecode(lyrics)  # Convert to ASCII (basically takes accentuation)
    lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove text within brackets
    lyrics = lyrics.lower() #ower case all the lyrics
    lyrics = re.sub(r'[^a-zA-Z0-9\s]', '', lyrics)  # Remove special characters
    lyrics = re.sub(r'\s+', ' ', lyrics)  # replace multiple spaces with a single space
    return lyrics.strip()  # remove leading and trailing whitespace

def tokenize_lyrics(lyrics):
    '''Tokenize the lyrics and remove unwanted characters like abreviations and contractions'''
    tokenized_text = nltk.tokenize.word_tokenize(clean_lyrics(lyrics)) #separate the lyrics into tokens

    tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
    tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
    tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]
    return tokenized_text

# turning tokens into lemmas (using literally the function from class notebooks)
def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    """Apply WordNet lemmatization for each requested part-of-speech tag.

    Inputs are one token and an ordered list of WordNet POS tags. The function
    repeatedly updates the token with each lemmatization pass and returns the
    final lemma.
    """
    wordnet_lem = nltk.stem.WordNetLemmatizer() 

    return [wordnet_lem.lemmatize(token, arg_1) for arg_1 in list_pos] #only difference: comprehension instead of for loop, but same result

#in case we need steemming instead of lemmatization
def stemming_all(token):
    """Apply Porter stemming to a token."""
    porter_stemmer = nltk.stem.PorterStemmer()
    return [[porter_stemmer.stem(token) for token in token]]

#______ PREPROCESSING FOR SENTIMENT ANALYSIS _____#
