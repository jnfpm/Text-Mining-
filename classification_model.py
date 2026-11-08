import argparse
import json
import math
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


class TrainingPipeline(Pipeline):
    def __init__(self, steps, *, remove_duplicates=False, memory=None, verbose=False):
        self.remove_duplicates = remove_duplicates
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **params):
        if self.remove_duplicates:
            prepared = self.named_steps['preprocessing'].fit_transform(X, y)
            column = self.named_steps['preprocessing'].output_column
            if column != 'lyrics_processed':
                prepared = prepared.assign(lyrics_processed=prepared[column])
            mask = training_keep_mask(prepared, y, remove_duplicates=True)
            X = X.iloc[np.flatnonzero(mask)].copy()
            y = np.asarray(y)[mask]
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
        preprocessing = clone(self.preprocessing)
        text = FunctionTransformer(
            select_processed_text,
            kw_args={'column': self.preprocessing.output_column},
            validate=False,
        )
        features = self.model.build_features()
        classifier = self.model.build_classifier()
        return TrainingPipeline([
            ('preprocessing', preprocessing),
            ('text', text),
            ('features', features),
            ('classifier', classifier),
        ], remove_duplicates=self.remove_duplicates)

    def run(self, data_dir='data', output_path='data/predictions.csv',
            grid_search=False, sample_size=None, report_dir='reports/runs'):
        output_path = Path(output_path)
        source_paths = [Path(data_dir) / name for name in ('train.csv', 'test.csv', 'sample_submission.csv')]
        if output_path.resolve() in {path.resolve() for path in source_paths}:
            raise ValueError('Output must not overwrite an input dataset.')
        self.preprocessing.clean_text('')
        pipeline = self.build_pipeline()
        train, test, template = load_datasets(data_dir)
        if sample_size is not None:
            if not 0 < sample_size <= len(train):
                raise ValueError('sample_size must be between 1 and the training row count.')
            if sample_size < len(train):
                train, _ = train_test_split(
                    train, train_size=sample_size, stratify=train['tag'],
                    random_state=self.random_state,
                )
        training, validation = split_training_data(
            train, validation_size=self.validation_size,
            random_state=self.random_state,
        )
        X_train, y_train = training.drop(columns='tag'), training['tag']
        X_valid, y_valid = validation.drop(columns='tag'), validation['tag']
        print(f'Training: {len(training):,}; validation: {len(validation):,}; test: {len(test):,}')
        best_parameters = {}
        search_results = None
        if grid_search:
            parameters = self.model.parameter_grid()
            parameters.update(self.parameter_grid or {})
            print(f'Grid: {len(ParameterGrid(parameters))} combinations x {self.cv_folds} folds')
            search = GridSearchCV(
                pipeline, parameters, scoring='f1_macro',
                cv=StratifiedGroupKFold(
                    n_splits=self.cv_folds, shuffle=True,
                    random_state=self.random_state,
                ),
                n_jobs=self.n_jobs, refit=True, error_score='raise',
            )
            search.fit(X_train, y_train, groups=duplicate_keys(training))
            pipeline = search.best_estimator_
            best_parameters = search.best_params_
            search_results = pd.DataFrame(search.cv_results_)
        else:
            pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_valid)
        metrics = {
            'accuracy': accuracy_score(y_valid, prediction),
            'f1_macro': f1_score(y_valid, prediction, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_valid, prediction, average='weighted', zero_division=0),
        }
        print(json.dumps(metrics, indent=2))
        final_model = clone(pipeline)
        if 'classifier__validation_split' in final_model.get_params():
            final_model.set_params(classifier__validation_split=0.0)
        final_model.fit(train.drop(columns='tag'), train['tag'])
        test_prediction = final_model.predict(test)
        submission = build_submission(test['id'], test_prediction, template)
        if not set(submission['tag']).issubset(set(train['tag'])):
            raise ValueError('Predictions contain unknown genre labels.')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        report = {
            'model': type(self.model).__name__,
            'model_parameters': vars(self.model),
            'metrics': metrics,
            'classification_report': classification_report(y_valid, prediction, output_dict=True, zero_division=0),
            'best_parameters': best_parameters,
            'preprocessing': pipeline.named_steps['preprocessing'].get_params(),
            'remove_duplicates': pipeline.remove_duplicates,
            'training_rows': len(training),
            'fitted_training_rows': pipeline.training_rows_,
            'validation_rows': len(validation),
            'final_training_rows': final_model.training_rows_,
            'test_rows': len(test),
            'sample_size': sample_size,
            'grid_search': grid_search,
            'output': str(output_path),
        }
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f'{output_path.stem}.json'
        report_path.write_text(json.dumps(report, indent=2, default=str) + '\n')
        if search_results is not None:
            search_results.to_csv(report_dir / f'{output_path.stem}_grid.csv', index=False)
        print(f'Saved {len(submission):,} predictions to {output_path}')
        return {'model': final_model, 'validation_model': pipeline,
                'metrics': metrics, 'submission': submission, 'report': report}


METRICS = {
    'accuracy': accuracy_score,
    'f1_macro': lambda y, p: f1_score(y, p, average='macro', zero_division=0),
    'f1_weighted': lambda y, p: f1_score(y, p, average='weighted', zero_division=0),
}


class SearchRecorder:
    """Scorer for the halving search.

    Logs accuracy + F1 for every candidate (JSONL), returns the ranking metric,
    and whenever a candidate in the latest (largest) rounds beats the previous
    best it immediately writes a test submission, so a usable file exists even
    if the run is stopped before the final refit.
    """

    def __init__(self, scoring, searched, log_path, test, template, labels,
                 validation_rows, interim_path, plan=None, total_rows=None):
        self.plan = plan or []          # [(candidates, rows) per round]
        self.total_rows = total_rows
        self.total_fits = sum(count for count, _ in self.plan)
        # Work units ~ rows trained on; +1 full-size refit at the end.
        self.total_units = sum(count * rows for count, rows in self.plan) + (total_rows or 0)
        self.done_units = 0
        self.scoring = scoring
        self.searched = searched
        self.log_path = Path(log_path)
        self.test = test
        self.template = template
        self.labels = set(labels)
        self.validation_rows = validation_rows
        self.interim_path = Path(interim_path)
        self.started = time.time()
        self.calls = 0
        self.best = (0, -np.inf)  # (rows evaluated, score)
        self.records = []

    def __call__(self, estimator, X, y):
        self.calls += 1
        prediction = estimator.predict(X)
        metrics = {name: float(metric(y, prediction)) for name, metric in METRICS.items()}
        parameters = {key: value for key, value in estimator.get_params().items()
                      if key in self.searched}
        record = {
            'candidate': self.calls,
            'elapsed_hours': round((time.time() - self.started) / 3600, 3),
            'fitted_rows': getattr(estimator, 'training_rows_', None),
            'validation_rows': len(X),
            **metrics,
            'parameters': parameters,
        }
        score = metrics[self.scoring]
        rows = len(X)
        if rows >= 0.3 * self.validation_rows:
            if rows > self.best[0] * 1.05:
                self.best = (rows, -np.inf)
            if score > self.best[1]:
                self.best = (rows, score)
                submission = build_submission(self.test['id'], estimator.predict(self.test), self.template)
                if set(submission['tag']).issubset(self.labels):
                    self.interim_path.parent.mkdir(parents=True, exist_ok=True)
                    submission.to_csv(self.interim_path, index=False)
                    record['wrote_submission'] = str(self.interim_path)
        self.records.append(record)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, default=str) + '\n')
        print(f"\n[search] candidate {self.calls} | rows {record['fitted_rows']} | "
              f"acc {metrics['accuracy']:.4f} | f1_macro {metrics['f1_macro']:.4f} | "
              f"f1_weighted {metrics['f1_weighted']:.4f}"
              + (' | new best -> submission written' if 'wrote_submission' in record else ''),
              flush=True)
        if self.total_units and self.total_rows:
            self.done_units += len(X) * self.total_rows / self.validation_rows
            print(self.progress_line(), flush=True)
        return score

    def current_round(self):
        fits = 0
        for index, (count, _) in enumerate(self.plan):
            fits += count
            if self.calls <= fits:
                return index + 1
        return len(self.plan)

    def progress_line(self, width=30):
        fraction = min(self.done_units / self.total_units, 1.0)
        filled = int(round(width * fraction))
        elapsed = time.time() - self.started
        eta = elapsed / fraction - elapsed if fraction > 0 else float('nan')
        return (f"[progress] |{'█' * filled}{'░' * (width - filled)}| {fraction:5.1%} | "
                f"fit {self.calls}/{self.total_fits} | round {self.current_round()}/{len(self.plan)} | "
                f"elapsed {elapsed / 3600:.2f} h | ETA ~{eta / 3600:.1f} h (incl. final refit)")


def run_halving_search(experiment, data_dir, output_path, search_space,
                       n_candidates, factor, scoring, report_dir, sample_size=None):
    """HalvingRandomSearchCV over preprocessing + neural hyperparameters.

    One grouped holdout split (duplicate lyrics never straddle train/validation)
    is used for every round; the winner is refit on the full training set and
    used for the final test predictions.
    """
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV

    if scoring not in METRICS:
        raise ValueError(f'scoring must be one of {tuple(METRICS)}.')
    output_path = Path(output_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    experiment.preprocessing.clean_text('')
    train, test, template = load_datasets(data_dir)
    if sample_size is not None and sample_size < len(train):
        train, _ = train_test_split(train, train_size=sample_size, stratify=train['tag'],
                                    random_state=experiment.random_state)
    train = train.reset_index(drop=True)
    training, validation = split_training_data(
        train, validation_size=experiment.validation_size,
        random_state=experiment.random_state)
    split = [(training.index.to_numpy(), validation.index.to_numpy())]
    X, y = train.drop(columns='tag'), train['tag'].to_numpy()
    spaces = search_space if isinstance(search_space, list) else [search_space]
    searched = sorted({key for space in spaces for key in space})
    stem = output_path.stem
    log_path = report_dir / f'{stem}_candidates.jsonl'
    log_path.write_text('')
    # Same schedule HalvingRandomSearchCV uses with min_resources='exhaust'.
    rounds = 1 + int(math.floor(math.log(n_candidates, factor) + 1e-9))
    min_rows = len(train) // factor ** (rounds - 1)
    plan = [(math.ceil(n_candidates / factor ** index), min_rows * factor ** index)
            for index in range(rounds)]
    recorder = SearchRecorder(
        scoring, searched, log_path, test, template, y,
        validation_rows=len(validation),
        interim_path=output_path.with_name(f'{stem}_search_best.csv'),
        plan=plan, total_rows=len(train))
    search = HalvingRandomSearchCV(
        experiment.build_pipeline(), spaces, n_candidates=n_candidates,
        factor=factor, resource='n_samples', min_resources='exhaust',
        cv=split, scoring=recorder, refit=True, return_train_score=False,
        error_score=np.nan, random_state=experiment.random_state,
        n_jobs=1, verbose=2,
    )
    print(f'Training rows: {len(train):,} (search split {len(training):,} / '
          f'{len(validation):,}); test rows: {len(test):,}')
    print(f'Halving search: {n_candidates} candidates, factor {factor}, ranked by {scoring}')
    for index, (count, rows) in enumerate(plan, start=1):
        print(f'  round {index}: {count:>3} candidates x ~{rows:,} songs')
    print(f'  then: refit the winner on all {len(train):,} songs\n', flush=True)
    search.fit(X, y)

    recorder.done_units = recorder.total_units
    print(recorder.progress_line(), flush=True)
    print('\n[search] refit on the full training set finished; predicting test...', flush=True)
    submission = build_submission(test['id'], search.best_estimator_.predict(test), template)
    if not set(submission['tag']).issubset(set(y)):
        raise ValueError('Predictions contain unknown genre labels.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(report_dir / f'{stem}_halving.csv', index=False)
    final_round = [record for record in recorder.records
                   if record['parameters'] == {key: value for key, value in search.best_params_.items()}]
    best_record = max(final_round, key=lambda record: record['validation_rows'], default=None)
    classifier = search.best_estimator_.named_steps['classifier']
    report = {
        'model': type(experiment.model).__name__,
        'scoring': scoring,
        'best_parameters': search.best_params_,
        'best_validation_metrics': best_record,
        'n_candidates': n_candidates,
        'factor': factor,
        'n_resources_per_round': [int(value) for value in search.n_resources_],
        'n_candidates_per_round': [int(value) for value in search.n_candidates_],
        'final_training_rows': int(search.best_estimator_.training_rows_),
        'final_history': getattr(classifier, 'history_', None),
        'test_rows': len(test),
        'elapsed_hours': round((time.time() - started) / 3600, 3),
        'output': str(output_path),
    }
    (report_dir / f'{stem}.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({'best_parameters': search.best_params_,
                      'best_validation_metrics': best_record}, indent=2, default=str))
    print(f'Saved {len(submission):,} predictions to {output_path}')
    print(f"Total time: {report['elapsed_hours']:.2f} h")
    return {'search': search, 'submission': submission, 'report': report}


def check_setup(experiment, data_dir):
    train, test, template = load_datasets(data_dir)
    print(f'Data: {len(train):,} labeled training rows; {len(test):,} unlabeled test rows')
    print(f'Submission columns: {list(template.columns)}')
    for name, check in [
        ('Exercise 1', lambda: experiment.preprocessing.clean_text('[Verse 1: Artist]\nHello!')),
        ('Exercise 2', experiment.build_pipeline),
    ]:
        try:
            check()
            print(f'{name}: implemented')
        except NotImplementedError as error:
            print(f'{name}: unfinished — {error}')
    print('The optional neural model also needs Exercise 3. No models were trained.')


def main(argv=None):
    from config import (MODEL, MODEL_PARAMETERS, NEURAL_SEARCH, NEURAL_SEARCH_SPACE,
                        PREPROCESSING, RUN, SEARCH_GRID)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tfidf', 'complement_nb', 'multinomial_nb', 'neural_network'], default=MODEL)
    parser.add_argument('--data-dir', default=RUN['data_dir'])
    parser.add_argument('--output')
    parser.add_argument('--sample-size', type=int, default=RUN['sample_size'])
    parser.add_argument('--grid-search', action='store_true', default=RUN['grid_search'])
    parser.add_argument('--halving-search', action='store_true',
                        help='HalvingRandomSearchCV over preprocessing + neural settings (config.NEURAL_SEARCH).')
    parser.add_argument('--n-candidates', type=int, default=NEURAL_SEARCH['n_candidates'])
    parser.add_argument('--scoring', choices=['accuracy', 'f1_macro', 'f1_weighted'],
                        default=NEURAL_SEARCH['scoring'])
    parser.add_argument('--remove-duplicates', action=argparse.BooleanOptionalAction, default=RUN['remove_duplicates'])
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--train-neural', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = NEURAL_SEARCH['output_path'] if args.halving_search else RUN['output_path']
    parameters = MODEL_PARAMETERS[args.model].copy()
    if args.device:
        if args.model != 'neural_network':
            parser.error('--device is only used by the neural model.')
        parameters['device'] = args.device
    if args.halving_search and args.model != 'neural_network':
        parser.error('--halving-search is set up for --model neural_network.')
    experiment = ClassificationModel(
        model=get_model(args.model, **parameters),
        preprocessing=PreprocessingPipeline(**PREPROCESSING),
        remove_duplicates=args.remove_duplicates,
        validation_size=RUN['validation_size'], cv_folds=RUN['cv_folds'],
        random_state=RUN['random_state'], n_jobs=RUN['n_jobs'],
        parameter_grid=SEARCH_GRID,
    )
    if args.check:
        check_setup(experiment, args.data_dir)
        return
    if args.model == 'neural_network' and not args.train_neural:
        parser.error('Neural training is opt-in: add --train-neural (or use run_gpu.py).')
    if args.halving_search:
        run_halving_search(
            experiment, data_dir=args.data_dir, output_path=args.output,
            search_space=NEURAL_SEARCH_SPACE, n_candidates=args.n_candidates,
            factor=NEURAL_SEARCH['factor'], scoring=args.scoring,
            report_dir=RUN['report_dir'], sample_size=args.sample_size,
        )
        return
    try:
        experiment.run(
            data_dir=args.data_dir, output_path=args.output,
            grid_search=args.grid_search, sample_size=args.sample_size,
            report_dir=RUN['report_dir'],
        )
    except NotImplementedError as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
