"""Fine-tuned pretrained transformer (PyTorch + skorch) for genre classification.

torch / transformers / skorch are imported lazily inside fit/predict, so the
classical models keep working without the GPU stack installed.
"""
import gc
import math
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.utils.validation import check_is_fitted

TRUNCATION_MODES = ('head', 'head_tail')
CLASS_WEIGHT_MODES = (None, 'sqrt', 'balanced')


def class_weights(encoded, class_count, mode):
    """Per-class loss weights: None, inverse frequency ('balanced') or its sqrt."""
    if mode is None:
        return None
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f'class_weight must be one of {CLASS_WEIGHT_MODES}.')
    counts = np.bincount(encoded, minlength=class_count).astype(np.float64)
    weights = counts.sum() / (class_count * np.maximum(counts, 1.0))
    if mode == 'sqrt':
        weights = np.sqrt(weights)
    return (weights / weights.mean()).astype(np.float32)


def truncate_ids(ids, budget, mode):
    """Keep `budget` tokens: the start ('head') or start + end ('head_tail').

    head_tail keeps 1/4 of the budget from the start (where any metadata prefix
    sits) and 3/4 from the end, following Sun et al. (2019).
    """
    if len(ids) <= budget:
        return ids
    if mode == 'head':
        return ids[:budget]
    if mode == 'head_tail':
        head = budget // 4
        return ids[:head] + ids[len(ids) - (budget - head):]
    raise ValueError(f'truncation must be one of {TRUNCATION_MODES}.')


class NeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-compatible wrapper: tokenizer + skorch NeuralNetClassifier on CUDA."""

    def __init__(self, model_name='distilroberta-base', max_length=256,
                 truncation='head_tail', pooling='cls', dropout=0.1,
                 learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.06,
                 batch_size=32, max_epochs=4, patience=1, valid_fraction=0.1,
                 class_weight='sqrt', label_smoothing=0.0, grad_clip=1.0,
                 bf16=True, device='cuda', random_state=39, verbose=1):
        self.model_name = model_name
        self.max_length = max_length
        self.truncation = truncation
        self.pooling = pooling
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.valid_fraction = valid_fraction
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip
        self.bf16 = bf16
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ----------------------------------------------------------------- setup
    def _check_device(self, torch):
        if self.device not in {'cuda', 'cpu'}:
            raise ValueError("device must be 'cuda' or 'cpu'.")
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA is not available to PyTorch. Use the project venv '
                '(.venv\\Scripts\\python.exe) where torch was installed from the '
                'CUDA wheel index (see requirements-gpu.txt).'
            )

    def _encode(self, texts):
        tokenizer = self.tokenizer_
        texts = [str(text) for text in texts]
        ids = tokenizer(texts, add_special_tokens=False, truncation=False,
                        verbose=False)['input_ids']
        # Special tokens around a sequence, e.g. [CLS] ... [SEP] or <s> ... </s>.
        special = list(tokenizer('', add_special_tokens=True)['input_ids'])
        prefix, suffix = special[:1], special[1:]
        budget = self.max_length - len(special)
        input_ids = np.full((len(ids), self.max_length), tokenizer.pad_token_id,
                            dtype=np.int64)
        attention_mask = np.zeros((len(ids), self.max_length), dtype=np.int64)
        for row, sequence in enumerate(ids):
            sequence = prefix + truncate_ids(list(sequence), budget, self.truncation) + suffix
            input_ids[row, :len(sequence)] = sequence
            attention_mask[row, :len(sequence)] = 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    # ------------------------------------------------------------------- fit
    def fit(self, X, y):
        try:
            import torch
            from skorch import NeuralNetClassifier
            from skorch.callbacks import (EarlyStopping, GradientNormClipping,
                                          LRScheduler)
            from skorch.dataset import ValidSplit
            from torch.optim.lr_scheduler import LambdaLR
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging
        except ImportError as error:
            raise ImportError(
                'The neural model needs requirements-gpu.txt (torch with CUDA, '
                'transformers, skorch).'
            ) from error
        from .transformer_module import EpochProgressBar, LinearWarmupDecay, TransformerModule

        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        hf_logging.set_verbosity_error()
        self._check_device(torch)
        if self.truncation not in TRUNCATION_MODES:
            raise ValueError(f'truncation must be one of {TRUNCATION_MODES}.')
        if not 0 < self.valid_fraction < 1:
            raise ValueError('valid_fraction must be between 0 and 1.')
        texts = np.asarray(X, dtype=object)
        labels = np.asarray(y)
        if texts.ndim != 1 or labels.ndim != 1 or len(texts) != len(labels):
            raise ValueError('Provide one text string and one class label per row.')

        # Free GPU memory left by the previous search candidate.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.label_encoder_ = LabelEncoder().fit(labels)
        encoded = self.label_encoder_.transform(labels).astype(np.int64)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) < 2:
            raise ValueError('At least two classes are required.')

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        features = self._encode(texts)

        steps_per_epoch = math.ceil(len(texts) * (1 - self.valid_fraction) / self.batch_size)
        total_steps = max(1, steps_per_epoch * self.max_epochs)
        weights = class_weights(encoded, len(self.classes_), self.class_weight)
        criterion_options = {'criterion__label_smoothing': self.label_smoothing}
        if weights is not None:
            criterion_options['criterion__weight'] = torch.tensor(weights)

        self.net_ = NeuralNetClassifier(
            TransformerModule,
            module__model_name=self.model_name,
            module__num_classes=len(self.classes_),
            module__dropout=self.dropout,
            module__pooling=self.pooling,
            module__bf16=self.bf16,
            criterion=torch.nn.CrossEntropyLoss,
            **criterion_options,
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=self.weight_decay,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            train_split=ValidSplit(self.valid_fraction, stratified=True,
                                   random_state=self.random_state),
            iterator_train__shuffle=True,
            iterator_valid__batch_size=self.batch_size * 2,
            callbacks=([('progress', EpochProgressBar(self.max_epochs))] if self.verbose else []) + [
                ('lr_schedule', LRScheduler(
                    policy=LambdaLR, step_every='batch',
                    lr_lambda=LinearWarmupDecay(
                        int(self.warmup_ratio * total_steps), total_steps))),
                ('clip', GradientNormClipping(self.grad_clip)),
                ('early_stopping', EarlyStopping(
                    monitor='valid_loss', patience=self.patience,
                    load_best=True)),
            ],
            device=self.device,
            classes=np.arange(len(self.classes_)),
            verbose=self.verbose,
        )
        if self.verbose:
            print(f'[neural] {self.model_name} | {len(texts):,} rows | '
                  f'max_length={self.max_length} | up to {self.max_epochs} epochs '
                  f'(early stopping, patience {self.patience}) | device={self.device}', flush=True)
        try:
            self.net_.fit(features, encoded)
        except torch.cuda.OutOfMemoryError:
            self.net_ = None
            gc.collect()
            torch.cuda.empty_cache()
            raise
        self.history_ = [
            {key: epoch.get(key) for key in ('epoch', 'train_loss', 'valid_loss', 'valid_acc', 'dur')}
            for epoch in self.net_.history
        ]
        return self

    # --------------------------------------------------------------- predict
    def predict_proba(self, X):
        check_is_fitted(self, 'net_')
        texts = np.asarray(X, dtype=object)
        if texts.ndim != 1:
            raise ValueError('Provide a one-dimensional collection of text strings.')
        return np.asarray(self.net_.predict_proba(self._encode(texts)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class NeuralNetworkModel:
    def __init__(self, **parameters):
        self.parameters = parameters
        NeuralNetworkClassifier(**parameters)  # fail fast on unknown names

    def build_features(self):
        # The classifier tokenizes raw text itself.
        return FunctionTransformer(validate=False)

    def build_classifier(self):
        return NeuralNetworkClassifier(**self.parameters)

    def parameter_grid(self):
        return {
            'classifier__learning_rate': [2e-5, 4e-5],
            'classifier__max_length': [256, 512],
        }
