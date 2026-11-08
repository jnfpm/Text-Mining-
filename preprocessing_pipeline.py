from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

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

# Which song attributes are fed to the model next to the lyrics.
INPUT_FIELDS = {
    'lyrics': (),
    'lyrics_artist': ('artist',),
    'lyrics_artist_year': ('artist', 'year'),
    'all': ('title', 'artist', 'features', 'year'),
}

# Cleaned lyrics per cleaning configuration, so repeated fits in a search do
# not redo the (slow) cleaning. Keeps the most recent configurations only.
_CLEAN_CACHE = OrderedDict()
_CLEAN_CACHE_SIZE = 8
_PARALLEL_THRESHOLD = 2_000


def flatten_lines(text):
    return ' '.join(text.split())


def select_processed_text(frame, column='lyrics_processed'):
    return frame[column]


def _clean_chunk(cls, params, texts):
    pipeline = cls(**params)
    return [pipeline.clean_text(text) for text in texts]


def _format_value(field, value):
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return ''
    if field == 'year':
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return ''
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

    def _cleaning_params(self):
        excluded = {'input_fields', 'n_jobs', 'text_column', 'output_column'}
        return {key: value for key, value in self.get_params().items()
                if key not in excluded}

    def _workers(self):
        if self.n_jobs in (-1, None):
            return os.cpu_count() or 1
        return max(1, int(self.n_jobs))

    def _clean_many(self, texts):
        params = self._cleaning_params()
        key = (type(self).__module__, type(self).__qualname__,
               tuple(sorted(params.items())))
        cache = _CLEAN_CACHE.setdefault(key, {})
        _CLEAN_CACHE.move_to_end(key)
        while len(_CLEAN_CACHE) > _CLEAN_CACHE_SIZE:
            _CLEAN_CACHE.popitem(last=False)
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        workers = self._workers()
        if workers > 1 and len(missing) >= _PARALLEL_THRESHOLD:
            size = max(1, len(missing) // (workers * 4))
            chunks = [missing[start:start + size] for start in range(0, len(missing), size)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = pool.map(_clean_chunk, [type(self)] * len(chunks),
                                   [params] * len(chunks), chunks)
                for chunk, cleaned in zip(chunks, results):
                    cache.update(zip(chunk, cleaned))
        else:
            for text in missing:
                cache[text] = self.clean_text(text)
        return [cache[text] for text in texts]

    def _metadata_prefix(self, frame):
        if self.input_fields not in INPUT_FIELDS:
            raise ValueError(f'input_fields must be one of {tuple(INPUT_FIELDS)}.')
        fields = INPUT_FIELDS[self.input_fields]
        missing = [field for field in fields if field not in frame.columns]
        if missing:
            raise ValueError(
                f"input_fields={self.input_fields!r} needs columns: {', '.join(missing)}")
        if not fields:
            return None
        separator = '\n' if self.preserve_lines else ' '
        prefixes = []
        for values in zip(*(frame[field] for field in fields)):
            parts = []
            for field, value in zip(fields, values):
                text = _format_value(field, value)
                if text:
                    parts.append(f'{field}: {text}')
            prefix = ' | '.join(parts)
            if self.lowercase:
                prefix = prefix.lower()
            prefixes.append(prefix + separator if prefix else '')
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
        prefixes = self._metadata_prefix(frame)
        cleaned = self._clean_many(values.tolist())
        if prefixes is not None:
            cleaned = [prefix + text for prefix, text in zip(prefixes, cleaned)]
        frame[self.output_column] = pd.Series(cleaned, index=frame.index, dtype=object)
        return frame

    def transform_tokens(self, X):
        frame = self.transform(X)
        frame['tokens'] = frame[self.output_column].map(tokenize_text)
        return frame
