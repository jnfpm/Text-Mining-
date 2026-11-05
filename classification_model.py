import argparse
import json
from pathlib import Path

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
        raise NotImplementedError(
            'Exercise 2: assemble preprocessing, text, features and classifier '
            'into TrainingPipeline, using those exact step names and '
            'self.remove_duplicates.'
        )

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
    from config import MODEL, MODEL_PARAMETERS, PREPROCESSING, RUN, SEARCH_GRID

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tfidf', 'complement_nb', 'multinomial_nb', 'neural_network'], default=MODEL)
    parser.add_argument('--data-dir', default=RUN['data_dir'])
    parser.add_argument('--output', default=RUN['output_path'])
    parser.add_argument('--sample-size', type=int, default=RUN['sample_size'])
    parser.add_argument('--grid-search', action='store_true', default=RUN['grid_search'])
    parser.add_argument('--remove-duplicates', action=argparse.BooleanOptionalAction, default=RUN['remove_duplicates'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'])
    parser.add_argument('--train-neural', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    parameters = MODEL_PARAMETERS[args.model].copy()
    if args.device:
        if args.model != 'neural_network':
            parser.error('--device is only used by the neural model.')
        parameters['device'] = args.device
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
        parser.error('Neural training is opt-in: add --train-neural after completing Exercise 3.')
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
