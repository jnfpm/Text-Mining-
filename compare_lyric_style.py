"""Compare a lyrics baseline with a small classifier on its scores and lyric features."""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classification_model import ClassificationModel, compute_metrics
from config import MODEL_PARAMETERS, PREPROCESSING, RUN
from models import get_model
from preprocessing_pipeline import PreprocessingPipeline
from tools.datasets import split_training_data
from tools.lyric_cleaning import tokenize_text


# A small English pilot lexicon. Match whole tokens, including explicit inflections.
PROFANITY = frozenset({
    'fuck', 'fucks', 'fucking', 'fucked', 'shit', 'shitty', 'bullshit',
    'bitch', 'bitches', 'damn', 'motherfucker', 'motherfuckers', 'asshole', 'assholes',
})
# Count these forms separately because their use depends on context.
CONTEXT_DEPENDENT_TERMS = frozenset({'nigga', 'niggas', 'nigguh', 'nigger', 'niggers'})


def lyric_features(texts):
    style_rows, profanity_rows, context_rates = [], [], []
    for text in texts:
        lines = [line.strip() for line in text.casefold().splitlines() if line.strip()]
        tokens_by_line = [tokenize_text(line) for line in lines]
        word_count = sum(len(tokens) for tokens in tokens_by_line)
        flagged_by_line = [sum(token in PROFANITY for token in tokens) for tokens in tokens_by_line]
        flagged = sum(flagged_by_line)
        context_dependent = sum(token in CONTEXT_DEPENDENT_TERMS
                                for tokens in tokens_by_line for token in tokens)
        line_count = max(len(lines), 1)

        style_rows.append([
            np.log1p(word_count),
            word_count / line_count,
            (len(lines) - len(set(lines))) / line_count,
            text.count('!') / max(word_count, 1),
        ])
        profanity_rows.append([
            int(flagged > 0),
            100 * flagged / max(word_count, 1),
            sum(count > 0 for count in flagged_by_line) / line_count,
        ])
        context_rates.append(100 * context_dependent / max(word_count, 1))
    return np.asarray(style_rows), np.asarray(profanity_rows), np.asarray(context_rates)


def compare(baseline='roberta', data_dir=RUN['data_dir'], sample_size=None):
    model_name = 'neural_network' if baseline == 'roberta' else 'tfidf'
    parameters = MODEL_PARAMETERS[model_name].copy()
    if baseline == 'roberta':
        parameters['model_name'] = 'roberta-base'
    model = get_model(model_name, **parameters)

    # Use the repo's outer holdout, then reserve blend rows that the base model never sees.
    experiment = ClassificationModel(model, validation_size=RUN['validation_size'],
                                     random_state=RUN['random_state'])
    _, training, validation, _, _ = experiment.load_data(data_dir, sample_size)
    fit, blend = split_training_data(training, validation_size=RUN['validation_size'],
                                     random_state=RUN['random_state'])
    print(f'Base fit: {len(fit):,}; score combiner fit: {len(blend):,}', flush=True)

    preprocessing = PreprocessingPipeline(**{**PREPROCESSING, 'input_fields': 'lyrics',
                                             'remove_section_headers': True})
    text_column = preprocessing.output_column
    fit_text = preprocessing.fit_transform(fit[['lyrics']])[text_column]
    blend_text = preprocessing.transform(blend[['lyrics']])[text_column]
    valid_text = preprocessing.transform(validation[['lyrics']])[text_column]

    base = make_pipeline(model.build_features(), model.build_classifier())
    base.fit(fit_text, fit['tag'])
    if baseline == 'roberta':
        blend_scores = np.log(np.clip(base.predict_proba(blend_text), 1e-12, 1))
        valid_scores = np.log(np.clip(base.predict_proba(valid_text), 1e-12, 1))
    else:
        blend_scores = base.decision_function(blend_text)
        valid_scores = base.decision_function(valid_text)
        if valid_scores.ndim == 1:
            blend_scores = np.column_stack((-blend_scores, blend_scores))
            valid_scores = np.column_stack((-valid_scores, valid_scores))

    blend_style, blend_profanity, blend_context = lyric_features(blend_text)
    valid_style, valid_profanity, valid_context = lyric_features(valid_text)
    blend_with_style = np.column_stack((blend_scores, blend_style))
    valid_with_style = np.column_stack((valid_scores, valid_style))
    blend_with_profanity = np.column_stack((blend_with_style, blend_profanity))
    valid_with_profanity = np.column_stack((valid_with_style, valid_profanity))
    predictions = {'baseline': base.classes_[valid_scores.argmax(axis=1)]}
    for name, blend_input, valid_input in (
        ('scores_only', blend_scores, valid_scores),
        ('scores_and_style', blend_with_style, valid_with_style),
        ('scores_style_and_profanity', blend_with_profanity, valid_with_profanity),
        ('scores_style_profanity_and_context_terms',
         np.column_stack((blend_with_profanity, blend_context)),
         np.column_stack((valid_with_profanity, valid_context))),
    ):
        combiner = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=500),
        )
        combiner.fit(blend_input, blend['tag'])
        predictions[name] = combiner.predict(valid_input)

    results = {}
    for name, prediction in predictions.items():
        metrics = compute_metrics(validation['tag'], prediction)
        by_genre = f1_score(validation['tag'], prediction, labels=base.classes_,
                            average=None, zero_division=0)
        results[name] = {**metrics, 'genre_f1': dict(zip(base.classes_, by_genre.tolist()))}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', choices=('roberta', 'tfidf'), default='roberta')
    parser.add_argument('--data-dir', default=RUN['data_dir'])
    parser.add_argument('--sample-size', type=int)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.data_dir, args.sample_size), indent=2))
