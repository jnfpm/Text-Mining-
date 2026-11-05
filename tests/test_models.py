from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from models import get_model
from models.neural_network import NeuralNetworkClassifier


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
    if name.split('.')[0] in {'tensorflow', 'keras', 'gensim'}:
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
assert not any(name.split('.')[0] in {'tensorflow', 'gensim'} for name in sys.modules)
print('optional imports stayed lazy')
"""
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('optional imports stayed lazy', result.stdout)


class NeuralConfigurationTests(unittest.TestCase):
    def fake_tensorflow(self, devices):
        return SimpleNamespace(config=SimpleNamespace(
            list_physical_devices=mock.Mock(return_value=devices),
            experimental=SimpleNamespace(set_memory_growth=mock.Mock()),
        ))

    def test_neural_configuration_is_cloneable_without_building_a_network(self):
        model = get_model('neural_network', device='cpu', embedding_dim=32, epochs=2)
        classifier = clone(model.build_classifier())
        self.assertIsInstance(classifier, NeuralNetworkClassifier)
        self.assertEqual(classifier.device, 'cpu')
        self.assertEqual(classifier.embedding_dim, 32)
        self.assertEqual(classifier.epochs, 2)
        self.assertFalse(hasattr(classifier, 'network_'))
        features = model.build_features()
        np.testing.assert_array_equal(features.transform(['café', 'мир']), ['café', 'мир'])

    def test_cpu_mode_does_not_initialize_gpu_memory_growth(self):
        tensorflow = self.fake_tensorflow(['fake-gpu'])
        self.assertEqual(NeuralNetworkClassifier(device='cpu')._select_device(tensorflow), '/CPU:0')
        tensorflow.config.experimental.set_memory_growth.assert_not_called()

    def test_auto_selects_available_gpu_and_enables_memory_growth(self):
        tensorflow = self.fake_tensorflow(['gpu-one', 'gpu-two'])
        self.assertEqual(NeuralNetworkClassifier(device='auto')._select_device(tensorflow), '/GPU:0')
        self.assertEqual(tensorflow.config.experimental.set_memory_growth.call_args_list,
                         [mock.call('gpu-one', True), mock.call('gpu-two', True)])

    def test_gpu_request_requires_a_visible_device_and_auto_can_use_cpu(self):
        tensorflow = self.fake_tensorflow([])
        with self.assertRaises(RuntimeError):
            NeuralNetworkClassifier(device='gpu')._select_device(tensorflow)
        self.assertEqual(NeuralNetworkClassifier(device='auto')._select_device(tensorflow), '/CPU:0')

    def test_existing_gpu_context_is_accepted_and_invalid_mode_is_rejected(self):
        tensorflow = self.fake_tensorflow(['fake-gpu'])
        tensorflow.config.experimental.set_memory_growth.side_effect = RuntimeError('already initialized')
        self.assertEqual(NeuralNetworkClassifier(device='gpu')._select_device(tensorflow), '/GPU:0')
        with self.assertRaises(ValueError):
            NeuralNetworkClassifier(device='invalid')._select_device(tensorflow)


if __name__ == '__main__':
    unittest.main()
