"""CPU genre baselines using TF-IDF or mean Word2Vec features."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    multilabel_confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from preprocessing import MeanWord2Vec, tokenize_corpus, tokenize_lyrics


def build_model(random_state=39):
    """Build the first TF-IDF baseline for scalar genres or multi-hot targets."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_lyrics, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), max_features=20_000,
        )),
        ('classifier', OneVsRestClassifier(LogisticRegression(
            max_iter=1000, random_state=random_state,
        ))),
    ])


def build_word2vec_model(vector_size=25, epochs=5, min_count=2, random_state=39):
    """Accept scalar genres now, or a multi-hot target matrix for multilabel data."""
    return Pipeline([
        ('tokens', FunctionTransformer(tokenize_corpus)),
        ('embeddings', MeanWord2Vec(
            vector_size=vector_size, epochs=epochs, min_count=min_count,
            random_state=random_state,
        )),
        ('scaler', StandardScaler()),
        ('classifier', OneVsRestClassifier(LogisticRegression(
            max_iter=1000, random_state=random_state,
        ))),
    ])


def split_lyrics_data(frame, random_state=39):
    """Make an approximately 80/20 split, keeping identical tokens together."""
    groups = frame['lyrics'].map(tokenize_lyrics).str.join(' ')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                   random_state=random_state)
    train_idx, valid_idx = next(splitter.split(frame, frame['tag'], groups))
    return frame.iloc[train_idx].copy(), frame.iloc[valid_idx].copy()


def evaluate_model(model, lyrics, y_true, label_names=None, threshold=0.5):
    """Return the professor's classification metrics and per-category details."""
    y_true = np.asarray(y_true)
    scores = model.predict_proba(lyrics)
    multilabel = model.named_steps['classifier'].multilabel_
    if multilabel:
        y_pred = (scores >= threshold).astype(int)
        indicators = y_true
    else:
        y_pred = model.classes_[scores.argmax(axis=1)]
        indicators = (y_true[:, None] == model.classes_).astype(int)
    names = list(model.classes_ if label_names is None else label_names)
    if len(names) != scores.shape[1]:
        raise ValueError('label_names must follow the model probability columns.')

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted',
                                              zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted',
                                       zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
    }
    support = indicators.sum(axis=0)
    auc = np.full(scores.shape[1], np.nan)
    for i in range(scores.shape[1]):
        if 0 < support[i] < len(indicators):
            auc[i] = roc_auc_score(indicators[:, i], scores[:, i])
    eligible = np.isfinite(auc)
    metrics['roc_auc_weighted'] = (
        np.average(auc[eligible], weights=support[eligible])
        if eligible.any() else np.nan
    )
    report = classification_report(
        y_true, y_pred, labels=None if multilabel else model.classes_,
        target_names=[str(name) for name in names], output_dict=True,
        zero_division=0,
    )
    return {
        'metrics': pd.Series(metrics, name='validation'),
        'classification_report': pd.DataFrame(report).T,
        'roc_auc_by_class': pd.DataFrame({'roc_auc': auc, 'support': support},
                                        index=names),
        'auc_undefined_labels': [names[i] for i in np.flatnonzero(~eligible)],
        'confusion_matrices': multilabel_confusion_matrix(
            y_true, y_pred, labels=None if multilabel else model.classes_,
        ),
        'predictions': y_pred,
    }
