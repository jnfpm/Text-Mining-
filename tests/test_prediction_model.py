import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from model import build_model, build_word2vec_model, evaluate_model, split_lyrics_data
from preprocessing import MeanWord2Vec, tokenize_lyrics
from utils import load_prediction_data


class PredictionModelTests(unittest.TestCase):
    def test_tfidf_vocabulary_and_idf_stay_training_only(self):
        lyrics = ['guitar drums loud', 'piano melody quiet'] * 6
        model = build_model()
        model.fit(lyrics, ['rock', 'pop'] * 6)
        vectorizer = model.named_steps['tfidf']
        vocabulary = vectorizer.vocabulary_.copy()
        idf = vectorizer.idf_.copy()

        scores = model.predict_proba(['validationonlyword guitar', ''])

        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertIn('guitar drums', vocabulary)
        self.assertNotIn('validationonlyword', vectorizer.vocabulary_)
        self.assertEqual(vocabulary, vectorizer.vocabulary_)
        np.testing.assert_array_equal(idf, vectorizer.idf_)
        self.assertEqual(scores.shape, (2, 2))
        self.assertTrue(np.isfinite(scores).all())

    def test_embedding_vocabulary_stays_training_only(self):
        lyrics = ['guitar drums loud', 'piano melody quiet'] * 6
        model = build_word2vec_model(vector_size=8, epochs=2, min_count=1)
        model.fit(lyrics, ['rock', 'pop'] * 6)
        embedding = model.named_steps['embeddings']
        vocabulary = set(embedding.model_.wv.key_to_index)
        weights = embedding.model_.wv.vectors.copy()

        scores = model.predict_proba(['validationonlyword guitar'])

        self.assertNotIn('validationonlyword', embedding.model_.wv)
        self.assertEqual(vocabulary, set(embedding.model_.wv.key_to_index))
        np.testing.assert_array_equal(weights, embedding.model_.wv.vectors)
        self.assertTrue(np.isfinite(scores).all())

    def test_empty_and_unknown_lyrics_have_zero_vectors(self):
        embedding = MeanWord2Vec(vector_size=8, epochs=2, min_count=1)
        embedding.fit([['guitar', 'drums'], ['piano', 'melody']])

        vectors = embedding.transform([[], ['unknownword'], ['guitar']])

        self.assertEqual(vectors.shape, (3, 8))
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_array_equal(vectors[:2], np.zeros((2, 8)))
        np.testing.assert_allclose(vectors[2], embedding.model_.wv['guitar'])

    def test_duplicate_final_tokens_do_not_cross_split(self):
        rows = []
        for group in range(20):
            tag = 'rock' if group % 2 else 'pop'
            rows.extend([
                {'id': 2 * group, 'lyrics': f'[Chorus] SONG {group}!', 'tag': tag},
                {'id': 2 * group + 1, 'lyrics': f'song {group}', 'tag': tag},
            ])
        frame = pd.DataFrame(rows)

        training, validation = split_lyrics_data(frame)
        train_tokens = {tuple(tokenize_lyrics(text)) for text in training['lyrics']}
        valid_tokens = {tuple(tokenize_lyrics(text)) for text in validation['lyrics']}

        self.assertTrue(training.shape[0] and validation.shape[0])
        self.assertFalse(train_tokens & valid_tokens)
        self.assertFalse(set(training['id']) & set(validation['id']))
        self.assertEqual(set(training['id']) | set(validation['id']), set(frame['id']))

    def test_genuine_multilabel_predictions_and_label_order(self):
        lyrics = ['guitar drums loud', 'piano melody quiet',
                  'guitar drums piano melody'] * 20
        targets = np.array([[1, 0], [0, 1], [1, 1]] * 20)
        model = build_model()
        model.fit(lyrics, targets)

        result = evaluate_model(model, lyrics, targets,
                                label_names=['rock', 'pop'])

        self.assertTrue(model.named_steps['classifier'].multilabel_)
        self.assertEqual(result['predictions'].shape, targets.shape)
        self.assertTrue((result['predictions'].sum(axis=1) == 2).any())
        self.assertEqual(result['roc_auc_by_class'].index.tolist(), ['rock', 'pop'])
        self.assertEqual(result['confusion_matrices'].shape, (2, 2, 2))

    def test_undefined_auc_is_excluded_from_weighted_score(self):
        scores = np.array([[0.4, 0.1, 0.9], [0.6, 0.8, 0.7],
                           [0.8, 0.2, 0.6]])
        targets = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        model = SimpleNamespace(
            classes_=np.arange(3),
            named_steps={'classifier': SimpleNamespace(multilabel_=True)},
            predict_proba=lambda texts: scores[:len(texts)],
        )

        result = evaluate_model(model, ['a', 'b', 'c'], targets,
                                label_names=['rock', 'pop', 'absent'])

        self.assertEqual(result['auc_undefined_labels'], ['absent'])
        self.assertTrue(np.isnan(result['roc_auc_by_class'].loc['absent', 'roc_auc']))
        self.assertAlmostEqual(result['metrics']['roc_auc_weighted'], 2 / 3)

        undefined = evaluate_model(model, ['a'], targets[:1],
                                   label_names=['rock', 'pop', 'absent'])
        self.assertTrue(np.isnan(undefined['metrics']['roc_auc_weighted']))

    def test_binary_single_label_predictions_keep_one_class_per_row(self):
        lyrics = ['guitar drums loud', 'piano melody quiet'] * 10
        tags = np.array(['rock', 'pop'] * 10)
        model = build_model()
        model.fit(lyrics, tags)

        result = evaluate_model(model, lyrics, tags)

        self.assertFalse(model.named_steps['classifier'].multilabel_)
        self.assertEqual(model.predict_proba(lyrics).shape, (len(lyrics), 2))
        self.assertEqual(result['predictions'].shape, tags.shape)
        self.assertEqual(result['roc_auc_by_class'].index.tolist(), ['pop', 'rock'])
        self.assertEqual(result['confusion_matrices'].shape, (2, 2, 2))
        self.assertTrue(np.isfinite(result['metrics'].to_numpy()).all())

    def test_processed_labels_are_restored_by_id_without_changing_raw_files(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            processed_dir = data_dir / 'processed'
            processed_dir.mkdir()
            pd.DataFrame({'id': [1, 2, 3], 'lyrics': ['raw a', 'raw b', 'raw c'],
                          'tag': ['rock', 'pop', 'rap']}).to_csv(
                              data_dir / 'train.csv', index=False)
            pd.DataFrame({'id': [4], 'lyrics': ['raw test']}).to_csv(
                data_dir / 'test.csv', index=False)
            pd.DataFrame({'id': [3, 1, 2],
                          'lyrics': ['saved c', 'saved a', 'saved b']}).to_csv(
                              processed_dir / 'train.csv', index=False)
            pd.DataFrame({'id': [4], 'lyrics': ['saved test']}).to_csv(
                processed_dir / 'test.csv', index=False)
            original = {name: (data_dir / name).read_bytes()
                        for name in ['train.csv', 'test.csv']}

            with redirect_stdout(io.StringIO()):
                train, test = load_prediction_data(data_dir)

            self.assertEqual(train['id'].tolist(), [3, 1, 2])
            self.assertEqual(train['tag'].tolist(), ['rap', 'rock', 'pop'])
            self.assertEqual(train['lyrics'].tolist(), ['saved c', 'saved a', 'saved b'])
            self.assertEqual(test['lyrics'].tolist(), ['saved test'])
            for name, content in original.items():
                self.assertEqual((data_dir / name).read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
