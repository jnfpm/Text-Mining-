import re

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from text_quality import normalize_lyrics
from utils import artist_from_title, get_info_batch

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



#_____ PREPROCESSING FOR GENRE CLASSIFICATION _____#

def ensure_nltk_data():
    """Download the corpora used by the text-preprocessing functions if missing."""
    required = {'corpora/stopwords': 'stopwords', 'corpora/wordnet': 'wordnet'}
    for resource, package in required.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package)

# basic functions that will probably be needed to clean/preprocess the Lyrics

def removing_stopwords(lyrics):
    '''Remove stopwords from the lyrics'''
    stopwords = nltk.corpus.stopwords.words('english') #defines a list of stopwords in english
    lyrics = [word for word in lyrics if word not in stopwords]
    return lyrics

def clean_lyrics(lyrics):
    """Normalize lyrics and remove section annotations, keeping Unicode words."""
    lyrics = normalize_lyrics(lyrics)
    if not isinstance(lyrics, str):
        return ''
    lyrics = re.sub(r'\[[^\]]*\]', ' ', lyrics)
    lyrics = re.sub(r"[^\w\s']", ' ', lyrics.lower())
    return re.sub(r'\s+', ' ', lyrics).strip()

def tokenize_lyrics(lyrics):
    """Return lowercase word tokens, preserving contractions without downloads."""
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", clean_lyrics(lyrics))


def tokenize_corpus(lyrics):
    """Tokenize a collection of lyric strings."""
    return [tokenize_lyrics(text) for text in lyrics]


class MeanWord2Vec(TransformerMixin, BaseEstimator):
    """Learn word embeddings and average known word vectors for each lyric."""

    def __init__(self, vector_size=25, window=5, min_count=2, epochs=5,
                 random_state=39):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, epochs=self.epochs, sg=0,
            workers=1, seed=self.random_state,
        )
        self.model_.build_vocab(X)
        if not len(self.model_.wv):
            raise ValueError('No words remain; provide more lyrics or lower min_count.')
        self.model_.train(X, total_examples=self.model_.corpus_count,
                          epochs=self.epochs)
        return self

    def transform(self, X):
        check_is_fitted(self, 'model_')
        vectors = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, tokens in enumerate(X):
            known = [word for word in tokens if word in self.model_.wv]
            if known:
                vectors[i] = self.model_.wv[known].mean(axis=0)
        return vectors


def normalize_lyrics_inplace(frame):
    """Apply the project's conservative normalization to a frame's lyrics."""
    frame['lyrics'] = frame['lyrics'].map(normalize_lyrics)

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
