import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from tools.lyric_cleaning import tokenize_text

# Words with inner apostrophes ("don't", "i'm") stay one token.
TOKEN_PATTERN = r"(?u)\b\w+(?:'\w+)*\b"


class TextModel:
    """Shared count / TF-IDF settings of the classical models."""

    def __init__(self, representation='tfidf', ngram_range=(1, 2), min_df=2,
                 max_df=1.0, max_features=20_000, stop_words=None, sublinear_tf=True):
        self.representation = representation
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf

    def build_features(self):
        options = dict(ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
                       max_features=self.max_features, stop_words=self.stop_words,
                       lowercase=False, token_pattern=TOKEN_PATTERN)
        if self.representation == 'tfidf':
            return TfidfVectorizer(**options, sublinear_tf=self.sublinear_tf, dtype=np.float32)
        if self.representation == 'count':
            return CountVectorizer(**options)
        raise ValueError("representation must be 'tfidf' or 'count' (word2vec only for tfidf model).")


def tokenize_corpus(texts):
    return [tokenize_text(text) for text in texts]


class MeanWord2Vec(TransformerMixin, BaseEstimator):
    """Trains Word2Vec on the training lyrics; a song is the mean of its word vectors."""

    def __init__(self, vector_size=100, window=5, min_count=2, epochs=5, random_state=39):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y=None):
        from gensim.models import Word2Vec  # optional dependency

        sentences = [list(tokens) for tokens in X]
        self.model_ = Word2Vec(vector_size=self.vector_size, window=self.window,
                               min_count=self.min_count, workers=1, seed=self.random_state)
        self.model_.build_vocab(sentences)
        if not len(self.model_.wv):
            raise ValueError('No words remain; provide more text or lower min_count.')
        self.model_.train(sentences, total_examples=len(sentences), epochs=self.epochs)
        return self

    def transform(self, X):
        check_is_fitted(self, 'model_')
        sentences = list(X)
        vectors = np.zeros((len(sentences), self.vector_size), dtype=np.float32)
        for index, tokens in enumerate(sentences):
            known = [token for token in tokens if token in self.model_.wv]
            if known:
                vectors[index] = self.model_.wv[known].mean(axis=0)
        return vectors


def build_word2vec_features(vector_size, window, min_count, epochs, random_state):
    return Pipeline([
        ('tokens', FunctionTransformer(tokenize_corpus, validate=False)),
        ('embeddings', MeanWord2Vec(vector_size, window, min_count, epochs, random_state)),
        ('scaler', StandardScaler()),
    ])
