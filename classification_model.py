import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from models import get_model
from preprocessing_pipeline import PreprocessingPipeline, select_processed_text
from tools.datasets import (
    build_submission,
    duplicate_keys,
    load_datasets,
    split_training_data,
    training_keep_mask,
)

def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    return metrics


class TrainingPipeline(Pipeline):
    """A Pipeline that can drop duplicate songs from the training rows only."""

    # Same signature as sklearn's Pipeline (arguments after * must be named), plus remove_duplicates.
    def __init__(self, steps, *, remove_duplicates=False, memory=None, verbose=False):
        self.remove_duplicates = remove_duplicates
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **params):
        if self.remove_duplicates:
            preprocessing = self.named_steps['preprocessing']
            prepared = preprocessing.fit_transform(X, y)
            prepared['lyrics_processed'] = prepared[preprocessing.output_column]
            keep = training_keep_mask(prepared, y)  # True for the first copy of each song
            X = X[keep]
            y = np.asarray(y)[keep]
        self.training_rows_ = len(X)
        return super().fit(X, y, **params)


class ClassificationModel:
    def __init__(self, model, preprocessing=None, remove_duplicates=False,
                 validation_size=0.2, cv_folds=3, random_state=39,
                 n_jobs=1, parameter_grid=None):
        self.model = model
        self.preprocessing = preprocessing if preprocessing is not None else PreprocessingPipeline()
        self.remove_duplicates = remove_duplicates
        self.validation_size = validation_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parameter_grid = parameter_grid

    def build_pipeline(self):
        return TrainingPipeline([
            ('preprocessing', clone(self.preprocessing)),
            ('text', FunctionTransformer(
                select_processed_text,
                kw_args={'column': self.preprocessing.output_column},
                validate=False,
            )),
            ('features', self.model.build_features()),
            ('classifier', self.model.build_classifier()),
        ], remove_duplicates=self.remove_duplicates)

    def load_data(self, data_dir, sample_size=None):
        train, test, template = load_datasets(data_dir)
        if sample_size and sample_size < len(train):
            train, _ = train_test_split(train, train_size=sample_size, stratify=train['tag'],
                                        random_state=self.random_state)
        train = train.reset_index(drop=True)
        training, validation = split_training_data(
            train, validation_size=self.validation_size, random_state=self.random_state)
        print(f'Training: {len(training):,}; validation: {len(validation):,}; test: {len(test):,}')
        return train, training, validation, test, template

    def run(self, data_dir='data', output_path='data/predictions.csv',
            grid_search=False, sample_size=None, report_dir='reports/runs'):
        train, training, validation, test, template = self.load_data(data_dir, sample_size)
        X_train, y_train = training.drop(columns='tag'), training['tag']
        X_valid, y_valid = validation.drop(columns='tag'), validation['tag']

        pipeline = self.build_pipeline()
        best_parameters = {}
        search = None
        if grid_search:
            parameters = self.model.parameter_grid()
            if self.parameter_grid is not None:
                parameters.update(self.parameter_grid)
            print(f'Grid: {len(ParameterGrid(parameters))} combinations x {self.cv_folds} folds')
            search = GridSearchCV(
                pipeline, parameters, scoring='f1_macro', refit=True, n_jobs=self.n_jobs, error_score='raise',
                cv=StratifiedGroupKFold(n_splits=self.cv_folds, shuffle=True,
                                        random_state=self.random_state),
            )
            search.fit(X_train, y_train, groups=duplicate_keys(training))
            pipeline = search.best_estimator_
            best_parameters = search.best_params_
        else:
            pipeline.fit(X_train, y_train)

        prediction = pipeline.predict(X_valid)
        metrics = compute_metrics(y_valid, prediction)
        print(json.dumps(metrics, indent=2))

        # Retrain the chosen pipeline on all labelled songs before predicting the test set.
        final_model = clone(pipeline)
        final_model.fit(train.drop(columns='tag'), train['tag'])
        submission = save_submission(final_model, test, template, output_path, data_dir)
        report = {
            'model': type(self.model).__name__,
            'model_parameters': vars(self.model),
            'metrics': metrics,
            'classification_report': classification_report(y_valid, prediction, output_dict=True, zero_division=0),
            'best_parameters': best_parameters,
            'preprocessing': pipeline.named_steps['preprocessing'].get_params(),
            'remove_duplicates': pipeline.remove_duplicates,
            'fitted_training_rows': pipeline.training_rows_,
            'validation_rows': len(validation),
            'final_training_rows': final_model.training_rows_,
        }
        stem = Path(output_path).stem
        write_report(report, report_dir, stem)
        if search is not None:
            pd.DataFrame(search.cv_results_).to_csv(Path(report_dir) / f'{stem}_grid.csv', index=False)
        return {'model': final_model, 'validation_model': pipeline,
                'metrics': metrics, 'submission': submission, 'report': report}


def save_submission(model, test, template, output_path, data_dir):
    output_path = Path(output_path)
    for name in ['train.csv', 'test.csv', 'sample_submission.csv']:
        if output_path.resolve() == (Path(data_dir) / name).resolve():
            raise ValueError('Output must not overwrite an input dataset.')
    submission = build_submission(test['id'], model.predict(test), template)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f'Saved {len(submission):,} predictions to {output_path}')
    return submission


def write_report(report, report_dir, stem):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f'{stem}.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


def run_halving_search(experiment, data_dir, output_path, search_space,
                       n_candidates, factor, scoring, report_dir, sample_size=None):
    """Successive-halving search over preprocessing + neural settings.

    Every round uses the same grouped train/validation split (duplicate lyrics
    stay on one side), then the winner is refit on all training rows.
    """
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV

    started = time.time()
    train, training, validation, test, template = experiment.load_data(data_dir, sample_size)
    searched_names = set()
    for space in search_space:
        searched_names.update(space.keys())
    stem = Path(output_path).stem
    log_path = Path(report_dir) / f'{stem}_candidates.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('')

    # The halving search can only rank by one metric, so we give it our own scoring
    # function: it computes all three metrics, saves them to a log file and
    # returns the one used for ranking.
    def scorer(estimator, X, y):
        metrics = compute_metrics(y, estimator.predict(X))
        parameters = {}
        for name, value in estimator.get_params().items():
            if name in searched_names:
                parameters[name] = value
        record = {
            'elapsed_hours': round((time.time() - started) / 3600, 3),
            'fitted_rows': estimator.training_rows_,
            'metrics': metrics,
            'parameters': parameters,
        }
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(json.dumps(record, default=str) + '\n')
        print(f"[search] rows {estimator.training_rows_} | accuracy {metrics['accuracy']:.4f} | "
              f"f1_macro {metrics['f1_macro']:.4f} | f1_weighted {metrics['f1_weighted']:.4f}", flush=True)
        return metrics[scoring]

    search = HalvingRandomSearchCV(
        experiment.build_pipeline(), search_space, n_candidates=n_candidates,
        factor=factor, resource='n_samples', min_resources='exhaust',
        cv=[(training.index.to_numpy(), validation.index.to_numpy())],  # one fixed train/validation split
        scoring=scorer, refit=True, error_score=np.nan, return_train_score=False,
        random_state=experiment.random_state, verbose=2,
    )
    search.fit(train.drop(columns='tag'), train['tag'].to_numpy())

    save_submission(search.best_estimator_, test, template, output_path, data_dir)
    pd.DataFrame(search.cv_results_).to_csv(Path(report_dir) / f'{stem}_halving.csv', index=False)
    report = {
        'scoring': scoring,
        'best_score': search.best_score_,
        'best_parameters': search.best_params_,
        'n_candidates_per_round': [int(n) for n in search.n_candidates_],
        'n_resources_per_round': [int(n) for n in search.n_resources_],
        'final_training_rows': search.best_estimator_.training_rows_,
        'final_history': search.best_estimator_.named_steps['classifier'].history_,
        'elapsed_hours': round((time.time() - started) / 3600, 3),
    }
    write_report(report, report_dir, stem)
    print(json.dumps(report, indent=2, default=str))
    return {'search': search, 'report': report}


def main(argv=None):
    from config import (MODEL, MODEL_PARAMETERS, NEURAL_SEARCH, NEURAL_SEARCH_SPACE,
                        PREPROCESSING, RUN, SEARCH_GRID)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_PARAMETERS), default=MODEL)
    parser.add_argument('--data-dir', default=RUN['data_dir'])
    parser.add_argument('--output')
    parser.add_argument('--sample-size', type=int, default=RUN['sample_size'])
    parser.add_argument('--grid-search', action='store_true', default=RUN['grid_search'])
    parser.add_argument('--halving-search', action='store_true',
                        help='Neural model only: search settings from config.NEURAL_SEARCH_SPACE.')
    parser.add_argument('--remove-duplicates', action=argparse.BooleanOptionalAction,
                        default=RUN['remove_duplicates'])
    args = parser.parse_args(argv)
    if args.halving_search and args.model != 'neural_network':
        parser.error('--halving-search needs --model neural_network.')

    experiment = ClassificationModel(
        model=get_model(args.model, **MODEL_PARAMETERS[args.model]),
        preprocessing=PreprocessingPipeline(**PREPROCESSING),
        remove_duplicates=args.remove_duplicates,
        validation_size=RUN['validation_size'], cv_folds=RUN['cv_folds'],
        random_state=RUN['random_state'], n_jobs=RUN['n_jobs'],
        parameter_grid=SEARCH_GRID,
    )
    if args.halving_search:
        output = args.output if args.output else NEURAL_SEARCH['output_path']
        run_halving_search(
            experiment, args.data_dir, output,
            NEURAL_SEARCH_SPACE, NEURAL_SEARCH['n_candidates'], NEURAL_SEARCH['factor'],
            NEURAL_SEARCH['scoring'], RUN['report_dir'], args.sample_size,
        )
    else:
        output = args.output if args.output else RUN['output_path']
        experiment.run(args.data_dir, output, args.grid_search, args.sample_size, RUN['report_dir'])


if __name__ == '__main__':
    main()
