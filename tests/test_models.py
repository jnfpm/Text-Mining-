from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from models import get_model
from models.neural_network import NeuralNetworkClassifier, class_weights, truncate_ids


class ClassicalModelTests(unittest.TestCase):
    def test_each_classical_model_fits_count_and_tfidf_features(self):
        texts = ['guitar melody café', 'guitar strings melody', 'guitar rock live',
                 'rap beats мир', 'rap beats microphone', 'rap rhythm street']
        labels = ['rock'] * 3 + ['rap'] * 3
        for name in ('tfidf', 'complement_nb', 'multinomial_nb'):
            for representation in ('tfidf', 'count'):
                with self.subTest(name=name, representation=representation):
                    model = get_model(name, representation=representation, min_df=1, ngram_range=(1, 1))
                    pipeline = Pipeline([
                        ('features', model.build_features()),
                        ('classifier', model.build_classifier()),
                    ])
                    pipeline.fit(texts, labels)
                    self.assertEqual(pipeline.predict(texts).tolist(), labels)
                    self.assertIn('café', pipeline.named_steps['features'].vocabulary_)
                    self.assertIn('мир', pipeline.named_steps['features'].vocabulary_)
                    self.assertEqual(pipeline.named_steps['features'].transform(['unseenword']).nnz, 0)
                    copied = clone(pipeline)
                    for parameter in model.parameter_grid():
                        self.assertIn(parameter, copied.get_params())

    def test_naive_bayes_rejects_signed_word2vec_features(self):
        for name in ('complement_nb', 'multinomial_nb'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    get_model(name, representation='word2vec').build_features()

    def test_invalid_model_and_feature_names_fail_clearly(self):
        with self.assertRaises(ValueError):
            get_model('unknown')
        with self.assertRaises(ValueError):
            get_model('tfidf', representation='unknown').build_features()

    def test_base_imports_and_model_construction_do_not_import_optional_engines(self):
        script = """
import builtins
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'tensorflow', 'keras', 'gensim', 'torch', 'transformers', 'skorch'}:
        raise AssertionError('Optional engine imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import config
import preprocessing_pipeline
import classification_model
from models import get_model

for name in ('tfidf', 'complement_nb', 'multinomial_nb', 'neural_network'):
    model = get_model(name)
    model.build_features()
    model.build_classifier()
get_model('tfidf', representation='word2vec').build_features()
assert not any(name.split('.')[0] in {'tensorflow', 'gensim', 'torch', 'transformers', 'skorch'} for name in sys.modules)
print('optional imports stayed lazy')
"""
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('optional imports stayed lazy', result.stdout)


class NeuralConfigurationTests(unittest.TestCase):
    def test_neural_configuration_is_cloneable_without_building_a_network(self):
        model = get_model('neural_network', device='cpu', max_length=128, max_epochs=2)
        classifier = clone(model.build_classifier())
        self.assertIsInstance(classifier, NeuralNetworkClassifier)
        self.assertEqual(classifier.device, 'cpu')
        self.assertEqual(classifier.max_length, 128)
        self.assertEqual(classifier.max_epochs, 2)
        self.assertFalse(hasattr(classifier, 'net_'))
        features = model.build_features()
        np.testing.assert_array_equal(features.transform(['café', 'мир']), ['café', 'мир'])
        for parameter in model.parameter_grid():
            self.assertIn(parameter, Pipeline([('classifier', classifier)]).get_params())

    def test_unknown_neural_parameter_fails_fast(self):
        with self.assertRaises(TypeError):
            get_model('neural_network', embedding_dim=32)

    def test_head_tail_truncation_keeps_start_and_end(self):
        ids = list(range(20))
        self.assertEqual(truncate_ids(ids, 8, 'head'), list(range(8)))
        self.assertEqual(truncate_ids(ids, 8, 'head_tail'), [0, 1, 14, 15, 16, 17, 18, 19])
        self.assertEqual(truncate_ids(ids[:5], 8, 'head_tail'), ids[:5])
        with self.assertRaises(ValueError):
            truncate_ids(ids, 8, 'middle')

    def test_class_weights_upweight_rare_classes(self):
        encoded = np.array([0] * 90 + [1] * 10)
        self.assertIsNone(class_weights(encoded, 2, None))
        balanced = class_weights(encoded, 2, 'balanced')
        sqrt = class_weights(encoded, 2, 'sqrt')
        self.assertAlmostEqual(float(balanced.mean()), 1.0, places=5)
        self.assertGreater(balanced[1], sqrt[1])
        self.assertGreater(sqrt[1], sqrt[0])
        with self.assertRaises(ValueError):
            class_weights(encoded, 2, 'unknown')


if __name__ == '__main__':
    unittest.main()
