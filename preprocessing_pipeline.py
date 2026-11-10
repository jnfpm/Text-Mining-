from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tools.lyric_cleaning import (
    clean_brackets,
    clean_unbracketed_headers,
    deduplicate_lines,
    remove_punctuation,
    tokenize_text,
)
from tools.text_quality import normalize_lyrics

# Song attributes written in front of the lyrics, e.g. "artist: drake | year: 2015".
INPUT_FIELDS = {
    'lyrics': (),
    'lyrics_artist': ('artist',),
    'lyrics_artist_year': ('artist', 'year'),
    'all': ('title', 'artist', 'features', 'year'),
}


def flatten_lines(text):
    return ' '.join(text.split())


def select_processed_text(frame, column='lyrics_processed'):
    return frame[column]


def format_field(field, value):
    if pd.isna(value):
        return ''
    if field == 'year':
        return str(int(value))
    text = ' '.join(str(value).split())
    if field == 'features':
        text = text.strip('{}').replace('"', '').replace("'", '')
    return text.strip()


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    def __init__(self, normalize_unicode=True, repair_mixed_script=False,
                 remove_section_headers=False, remove_performer_names=True,
                 remove_production_credits=True, remove_unknown_markers=True,
                 remove_bracket_symbols=False, remove_sung_brackets=False,
                 remove_unbracketed_headers=True, remove_repeated_lines=False,
                 lowercase=True, strip_punctuation=False, preserve_lines=True,
                 input_fields='lyrics', n_jobs=1,
                 text_column='lyrics', output_column='lyrics_processed'):
        self.normalize_unicode = normalize_unicode
        self.repair_mixed_script = repair_mixed_script
        self.remove_section_headers = remove_section_headers
        self.remove_performer_names = remove_performer_names
        self.remove_production_credits = remove_production_credits
        self.remove_unknown_markers = remove_unknown_markers
        self.remove_bracket_symbols = remove_bracket_symbols
        self.remove_sung_brackets = remove_sung_brackets
        self.remove_unbracketed_headers = remove_unbracketed_headers
        self.remove_repeated_lines = remove_repeated_lines
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.preserve_lines = preserve_lines
        self.input_fields = input_fields
        self.n_jobs = n_jobs
        self.text_column = text_column
        self.output_column = output_column

    def operations(self):
        bracket_options = {
            'remove_section_headers': self.remove_section_headers,
            'remove_performer_names': self.remove_performer_names,
            'remove_production_credits': self.remove_production_credits,
            'remove_unknown_markers': self.remove_unknown_markers,
            'remove_bracket_symbols': self.remove_bracket_symbols,
            'remove_sung_brackets': self.remove_sung_brackets,
        }
        return [
            (self.normalize_unicode, partial(
                normalize_lyrics, repair_mixed_script=self.repair_mixed_script,
                preserve_lines=True,
            )),
            (any(bracket_options.values()), partial(clean_brackets, **bracket_options)),
            (self.remove_unbracketed_headers, partial(
                clean_unbracketed_headers,
                remove_section_headers=self.remove_section_headers,
                remove_production_credits=self.remove_production_credits,
            )),
            (self.remove_repeated_lines, deduplicate_lines),
            (self.strip_punctuation, remove_punctuation),
            (self.lowercase, str.lower),
            (not self.preserve_lines, flatten_lines),
        ]

    def clean_text(self, text):
        for enabled, operation in self.operations():
            if enabled:
                text = operation(text)
        return text

    def clean_many(self, texts):
        # Cleaning is slow (~4 min for the training set on one core), so split big batches over processes.
        if self.n_jobs == 1 or len(texts) < 2_000:
            return [self.clean_text(text) for text in texts]
        workers = None if self.n_jobs == -1 else self.n_jobs
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.clean_text, texts, chunksize=500))

    def metadata_prefixes(self, frame):
        fields = INPUT_FIELDS[self.input_fields]
        if self.preserve_lines:
            separator = '\n'
        else:
            separator = ' '
        prefixes = []
        for _, row in frame.iterrows():
            parts = []
            for field in fields:
                text = format_field(field, row[field])
                if text != '':
                    parts.append(f'{field}: {text}')
            prefix = ' | '.join(parts)
            if self.lowercase:
                prefix = prefix.lower()
            if prefix != '':
                prefix = prefix + separator
            prefixes.append(prefix)
        return prefixes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        elif isinstance(X, pd.Series):
            frame = X.to_frame(name=self.text_column)
        else:
            frame = pd.DataFrame({self.text_column: list(X)})
        if self.text_column not in frame.columns:
            raise ValueError(f'Missing text column: {self.text_column}')
        if self.output_column == self.text_column:
            raise ValueError('Use a separate output column to preserve the original lyrics.')
        values = frame[self.text_column].fillna('')
        if not values.map(lambda value: isinstance(value, str)).all():
            raise TypeError('Lyrics must contain strings or missing values.')
        cleaned = self.clean_many(values.tolist())
        if INPUT_FIELDS[self.input_fields]:
            cleaned = [prefix + text for prefix, text in zip(self.metadata_prefixes(frame), cleaned)]
        frame[self.output_column] = pd.Series(cleaned, index=frame.index, dtype=object)
        return frame

    def transform_tokens(self, X):
        frame = self.transform(X)
        frame['tokens'] = frame[self.output_column].map(tokenize_text)
        return frame
