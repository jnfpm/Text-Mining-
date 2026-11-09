"""Fine-tuned pretrained transformer (PyTorch + skorch) for genre classification.

torch / transformers / skorch are imported inside fit, so the classical models
work without the GPU stack installed.
"""
import gc
import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.utils.validation import check_is_fitted


def class_weights(encoded, class_count, mode):
    """Loss weight per class: None, inverse frequency ('balanced') or its square root ('sqrt')."""
    if mode is None:
        return None
    if mode not in ('sqrt', 'balanced'):
        raise ValueError("class_weight must be None, 'sqrt' or 'balanced'.")
    counts = np.bincount(encoded, minlength=class_count)
    weights = counts.sum() / (class_count * np.maximum(counts, 1.0))
    if mode == 'sqrt':
        weights = np.sqrt(weights)
    return (weights / weights.mean()).astype(np.float32)


def truncate_ids(ids, budget, mode):
    """Keep `budget` tokens: the start ('head'), or 1/4 start + 3/4 end ('head_tail', Sun et al. 2019)."""
    if mode not in ('head', 'head_tail'):
        raise ValueError("truncation must be 'head' or 'head_tail'.")
    if len(ids) <= budget or mode == 'head':
        return ids[:budget]
    head = budget // 4
    return ids[:head] + ids[len(ids) - (budget - head):]


class NeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-compatible wrapper: tokenizer + skorch NeuralNetClassifier."""

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

    def _encode(self, texts):
        """Token ids padded to max_length, with the model's start/end tokens around them."""
        ids = self.tokenizer_([str(text) for text in texts], add_special_tokens=False,
                              verbose=False)['input_ids']  # no too-long warnings: we truncate below
        special = self.tokenizer_('')['input_ids']  # e.g. [<s>, </s>] or [CLS, SEP]
        input_ids = np.full((len(ids), self.max_length), self.tokenizer_.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(ids), self.max_length), dtype=np.int64)
        for row, sequence in enumerate(ids):
            sequence = special[:1] + truncate_ids(sequence, self.max_length - len(special),
                                                  self.truncation) + special[1:]
            input_ids[row, :len(sequence)] = sequence
            attention_mask[row, :len(sequence)] = 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def fit(self, X, y):
        import torch
        from skorch import NeuralNetClassifier
        from skorch.callbacks import EarlyStopping, GradientNormClipping, LRScheduler
        from skorch.dataset import ValidSplit
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup

        from .transformer_module import EpochProgressBar, TransformerModule

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA not available: run with .venv\\Scripts\\python.exe '
                               '(torch CUDA build, see requirements-gpu.txt).')
        # Free GPU memory left by the previous search candidate.
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(self.random_state)

        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        encoded = self.label_encoder_.transform(y).astype(np.int64)
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)

        total_steps = self.max_epochs * math.ceil(len(encoded) * (1 - self.valid_fraction) / self.batch_size)
        weights = class_weights(encoded, len(self.classes_), self.class_weight)
        callbacks = [
            ('lr_schedule', LRScheduler(
                get_linear_schedule_with_warmup, step_every='batch',
                num_warmup_steps=int(self.warmup_ratio * total_steps),
                num_training_steps=total_steps)),
            ('clip', GradientNormClipping(self.grad_clip)),
            ('early_stopping', EarlyStopping(patience=self.patience, load_best=True)),
        ]
        if self.verbose:
            callbacks.insert(0, ('progress', EpochProgressBar(self.max_epochs)))
        self.net_ = NeuralNetClassifier(
            TransformerModule,
            module__model_name=self.model_name,
            module__num_classes=len(self.classes_),
            module__dropout=self.dropout,
            module__pooling=self.pooling,
            module__bf16=self.bf16,
            criterion=torch.nn.CrossEntropyLoss,
            criterion__label_smoothing=self.label_smoothing,
            criterion__weight=None if weights is None else torch.tensor(weights),
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=self.weight_decay,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            iterator_valid__batch_size=self.batch_size * 2,
            max_epochs=self.max_epochs,
            train_split=ValidSplit(self.valid_fraction, stratified=True,
                                   random_state=self.random_state),
            callbacks=callbacks,
            device=self.device,
            classes=np.arange(len(self.classes_)),
            verbose=self.verbose,
        )
        if self.verbose:
            print(f'[neural] {self.model_name} | {len(encoded):,} rows | max_length={self.max_length} '
                  f'| up to {self.max_epochs} epochs | device={self.device}', flush=True)
        self.net_.fit(self._encode(X), encoded)
        self.history_ = [{key: epoch.get(key) for key in ('epoch', 'train_loss', 'valid_loss', 'valid_acc', 'dur')}
                         for epoch in self.net_.history]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'net_')
        return self.net_.predict_proba(self._encode(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class NeuralNetworkModel:
    def __init__(self, **parameters):
        self.parameters = parameters
        NeuralNetworkClassifier(**parameters)  # fail fast on unknown names

    def build_features(self):
        return FunctionTransformer(validate=False)  # the classifier tokenizes raw text itself

    def build_classifier(self):
        return NeuralNetworkClassifier(**self.parameters)

    def parameter_grid(self):
        return {
            'classifier__learning_rate': [2e-5, 4e-5],
            'classifier__max_length': [256, 512],
        }
