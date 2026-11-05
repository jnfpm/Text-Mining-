import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer

from classification_model import ClassificationModel, TrainingPipeline
from models import get_model
from preprocessing_pipeline import PreprocessingPipeline, select_processed_text
from tools.datasets import duplicate_keys, split_training_data


class CompletedPreprocessingForTests(PreprocessingPipeline):
    def clean_text(self, text):
        for enabled, operation in self.operations():
            if enabled:
                text = operation(text)
        return text


class CompletedClassificationForTests(ClassificationModel):
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


class RecordingGridSearchForTests(GridSearchCV):
    fits = []

    def fit(self, X, y=None, **params):
        self.fits.append((self, X.copy(), np.asarray(y).copy(), params['groups'].copy()))
        return super().fit(X, y, **params)


def small_frames():
    train = pd.DataFrame([
        {'id': index, 'lyrics': f'{"guitar melody" if index % 2 else "beats rhythm"} uniqueword{index}',
         'tag': 'rock' if index % 2 else 'rap'}
        for index in range(24)
    ])
    repeats = train.iloc[:4].copy()
    repeats['id'] = range(24, 28)
    train = pd.concat([train, repeats], ignore_index=True)
    test = pd.DataFrame({
        'id': [1001, 1002, 1003],
        'lyrics': ['guitar testexclusive', 'beats testexclusive', 'guitar testexclusive'],
    })
    template = pd.DataFrame({'id': [1003, 1001, 1002], 'tag': ['placeholder'] * 3})
    return train, test, template


class PreprocessingInfrastructureTests(unittest.TestCase):
    def test_transform_preserves_raw_source_identifiers_index_and_order(self):
        frame = pd.DataFrame({
            'id': [82, 17, 93],
            'lyrics': ['[Verse 1: Beyoncé]\nCAFÉ!', pd.NA, 'Мир\n漢字'],
            'artist': ['Beyoncé', 'Other', '第三'],
        }, index=pd.Index(['duplicate', 'duplicate', 'last'], name='source_row'))
        original = frame.copy(deep=True)
        preprocessing = CompletedPreprocessingForTests(
            remove_section_headers=True, output_column='prepared',
        )
        result = preprocessing.fit_transform(frame)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(result[original.columns], original)
        self.assertEqual(result['prepared'].tolist(), ['café!', '', 'мир\n漢字'])
        result.iloc[0, result.columns.get_loc('lyrics')] = 'changed result'
        pd.testing.assert_frame_equal(frame, original)

    def test_flags_preserve_raw_mode_and_line_flattening_runs_last(self):
        raw = " [Verse 1: Beyoncé]\r\nCAFÉ\u2005LOVE \r\n"
        preprocessing = CompletedPreprocessingForTests(
            normalize_unicode=False, remove_performer_names=False,
            remove_production_credits=False, remove_unknown_markers=False,
            remove_unbracketed_headers=False, lowercase=False,
        )
        self.assertEqual(preprocessing.transform([raw])['lyrics_processed'].iloc[0], raw)
        processed = CompletedPreprocessingForTests(
            remove_section_headers=True, remove_repeated_lines=True,
            strip_punctuation=True, preserve_lines=False,
        ).transform(['Chorus\nLOVE!\nlove\nLOVE!'])
        self.assertEqual(processed['lyrics_processed'].iloc[0], 'love love')

    def test_unicode_tokens_and_custom_column_are_exposed(self):
        source = pd.Series(["I’m café мир می\u200cروم क्\u200dष"], index=['song'])
        result = CompletedPreprocessingForTests(output_column='prepared').transform_tokens(source)
        self.assertEqual(result['tokens'].iloc[0], ["i'm", 'café', 'мир', 'می\u200cروم', 'क्\u200dष'])
        pd.testing.assert_index_equal(result.index, source.index)
        self.assertEqual(result['lyrics'].iloc[0], source.iloc[0])

    def test_input_errors_and_source_column_collision_are_clear(self):
        with self.assertRaises(ValueError):
            CompletedPreprocessingForTests().transform(pd.DataFrame({'title': ['song']}))
        with self.assertRaises(ValueError):
            CompletedPreprocessingForTests(output_column='lyrics').transform(['song'])
        with self.assertRaises(TypeError):
            CompletedPreprocessingForTests().transform(['song', 12])

    def test_estimator_cloning_keeps_flags_and_nested_grid_names(self):
        preprocessing = CompletedPreprocessingForTests(
            remove_repeated_lines=True, output_column='prepared',
        )
        copied = clone(preprocessing)
        self.assertIsNot(copied, preprocessing)
        self.assertEqual(copied.get_params(), preprocessing.get_params())
        experiment = CompletedClassificationForTests(
            get_model('multinomial_nb', min_df=1), preprocessing,
            remove_duplicates=True,
        )
        pipeline = clone(experiment.build_pipeline())
        self.assertTrue(pipeline.remove_duplicates)
        for name in ('preprocessing__remove_repeated_lines', 'features__ngram_range',
                     'classifier__alpha', 'remove_duplicates'):
            self.assertIn(name, pipeline.get_params())
        pipeline.set_params(preprocessing__lowercase=False, classifier__alpha=0.25)
        self.assertFalse(pipeline.named_steps['preprocessing'].lowercase)
        self.assertEqual(pipeline.named_steps['classifier'].alpha, 0.25)


class TrainingInfrastructureTests(unittest.TestCase):
    def test_fit_filters_same_label_duplicates_and_never_filters_predictions(self):
        features = pd.DataFrame({
            'id': range(8),
            'lyrics': ['same lyric', 'same lyric', 'same lyric', 'guitar rock',
                       'beats rap', None, '', 'same lyric'],
        }, index=[80, 10, 50, 90, 20, 30, 70, 40])
        targets = pd.Series(['rock', 'rock', 'rap', 'rock', 'rap', 'rock', 'rap', 'rock'],
                            index=range(200, 208))
        original, original_targets = features.copy(deep=True), targets.copy(deep=True)
        classifier = DummyClassifier(strategy='most_frequent')
        with mock.patch.object(classifier, 'fit', wraps=classifier.fit) as fit:
            pipeline = TrainingPipeline([
                ('preprocessing', CompletedPreprocessingForTests(output_column='prepared')),
                ('text', FunctionTransformer(select_processed_text, kw_args={'column': 'prepared'})),
                ('features', CountVectorizer()),
                ('classifier', classifier),
            ], remove_duplicates=True)
            pipeline.fit(features, targets)
            fitted_targets = np.asarray(fit.call_args.args[1])
        self.assertEqual(pipeline.training_rows_, 6)
        self.assertEqual(fitted_targets.tolist(), ['rock', 'rap', 'rock', 'rap', 'rock', 'rap'])
        self.assertEqual(len(pipeline.predict(features)), len(features))
        pd.testing.assert_frame_equal(features, original)
        pd.testing.assert_series_equal(targets, original_targets)

    def test_grouped_search_refit_submission_and_training_only_vocabulary(self):
        train, test, template = small_frames()
        training, validation = split_training_data(train, validation_size=0.25, random_state=39)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for frame, name in zip((train, test, template), ('train.csv', 'test.csv', 'sample_submission.csv')):
                frame.to_csv(directory / name, index=False)
            source_bytes = {name: (directory / name).read_bytes()
                            for name in ('train.csv', 'test.csv', 'sample_submission.csv')}
            experiment = CompletedClassificationForTests(
                get_model('multinomial_nb', min_df=1, ngram_range=(1, 1)),
                CompletedPreprocessingForTests(output_column='prepared'),
                validation_size=0.25, cv_folds=2, random_state=39,
                parameter_grid={
                    'features__ngram_range': [(1, 1)],
                    'classifier__alpha': [0.5, 1.0],
                    'preprocessing__remove_section_headers': [False, True],
                    'remove_duplicates': [False, True],
                },
            )
            RecordingGridSearchForTests.fits.clear()
            with mock.patch('classification_model.GridSearchCV', RecordingGridSearchForTests), contextlib.redirect_stdout(io.StringIO()):
                result = experiment.run(directory, directory / 'prediction.csv', grid_search=True,
                                        report_dir=directory / 'reports')
            self.assertEqual(len(RecordingGridSearchForTests.fits), 1)
            search, searched_features, searched_targets, groups = RecordingGridSearchForTests.fits[0]
            self.assertEqual(search.n_splits_, 2)
            self.assertEqual(set(searched_features['id']), set(training['id']))
            self.assertTrue(groups.duplicated().any())
            for fold_train, fold_valid in search.cv.split(searched_features, searched_targets, groups):
                self.assertTrue(set(groups.iloc[fold_train]).isdisjoint(groups.iloc[fold_valid]))
            validation_vocabulary = result['validation_model'].named_steps['features'].vocabulary_
            final_vocabulary = result['model'].named_steps['features'].vocabulary_
            for lyrics in validation['lyrics']:
                unique_word = lyrics.split()[-1]
                self.assertNotIn(unique_word, validation_vocabulary)
                self.assertIn(unique_word, final_vocabulary)
            self.assertNotIn('testexclusive', final_vocabulary)
            validation_rows = training['lyrics'].nunique() if result['validation_model'].remove_duplicates else len(training)
            final_rows = train['lyrics'].nunique() if result['model'].remove_duplicates else len(train)
            self.assertEqual(result['validation_model'].training_rows_, validation_rows)
            self.assertEqual(result['model'].training_rows_, final_rows)
            self.assertEqual(result['submission']['id'].tolist(), template['id'].tolist())
            self.assertEqual(len(result['submission']), len(test))
            expected = pd.Series(result['model'].predict(test), index=test['id'])
            self.assertEqual(result['submission']['tag'].tolist(), template['id'].map(expected).tolist())
            self.assertNotIn('placeholder', result['submission']['tag'].tolist())
            self.assertTrue((directory / 'reports' / 'prediction.json').is_file())
            self.assertTrue((directory / 'reports' / 'prediction_grid.csv').is_file())
            for name, content in source_bytes.items():
                self.assertEqual((directory / name).read_bytes(), content)
            pd.testing.assert_series_equal(groups, duplicate_keys(training))


class StudentExerciseAcceptanceTests(unittest.TestCase):
    def test_exercise_one_when_implemented(self):
        preprocessing = PreprocessingPipeline(remove_section_headers=True, preserve_lines=False)
        try:
            result = preprocessing.transform(['[Verse 1: Beyoncé]\nCAFÉ!\nМир'])
        except NotImplementedError as error:
            self.skipTest(str(error))
        self.assertEqual(result['lyrics_processed'].iloc[0], 'café! мир')

    def test_exercise_two_when_implemented(self):
        experiment = ClassificationModel(
            get_model('multinomial_nb', min_df=1), CompletedPreprocessingForTests(),
            remove_duplicates=True,
        )
        try:
            pipeline = experiment.build_pipeline()
        except NotImplementedError as error:
            self.skipTest(str(error))
        self.assertIsInstance(pipeline, TrainingPipeline)
        self.assertEqual(list(pipeline.named_steps), ['preprocessing', 'text', 'features', 'classifier'])
        self.assertTrue(pipeline.remove_duplicates)
        frame = pd.DataFrame({'lyrics': ['guitar melody', 'guitar live', 'rap beats', 'rap rhythm']})
        pipeline.fit(frame, ['rock', 'rock', 'rap', 'rap'])
        self.assertEqual(len(pipeline.predict(frame)), len(frame))


if __name__ == '__main__':
    unittest.main()
