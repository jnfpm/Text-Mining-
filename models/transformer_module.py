"""PyTorch network used by models.neural_network (imported lazily: needs torch)."""
import math
import sys

import torch
from skorch.callbacks import Callback
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoModel


class EpochProgressBar(Callback):
    """One terminal progress bar per epoch: 'epoch 2/3 |████░░| 410/1250 loss=0.81'."""

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.bar_ = None

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        batches = math.ceil(len(dataset_train) / net.batch_size)
        if dataset_valid is not None:
            valid_batch = getattr(net, 'iterator_valid__batch_size', net.batch_size)
            batches += math.ceil(len(dataset_valid) / valid_batch)
        epoch = len(net.history)
        self.bar_ = tqdm(total=batches, desc=f'  epoch {epoch}/{self.max_epochs}',
                         unit='batch', leave=False, dynamic_ncols=True,
                         file=sys.stdout, mininterval=1.0)

    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        if self.bar_ is None:
            return
        key = 'train_loss' if training else 'valid_loss'
        loss = net.history[-1, 'batches', -1].get(key)
        self.bar_.set_postfix_str(f"{'train' if training else 'valid'} loss={loss:.3f}"
                                  if loss is not None else '', refresh=False)
        self.bar_.update(1)

    def on_epoch_end(self, net, **kwargs):
        if self.bar_ is not None:
            self.bar_.close()
            self.bar_ = None

    on_train_end = on_epoch_end

    def __getstate__(self):
        state = self.__dict__.copy()
        state['bar_'] = None
        return state


class TransformerModule(nn.Module):
    """Pretrained transformer encoder followed by a dropout + linear genre head."""

    def __init__(self, model_name, num_classes, dropout=0.1, pooling='cls', bf16=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.pooling = pooling
        self.bf16 = bf16

    def forward(self, input_ids, attention_mask):
        # Batches arrive padded to max_length; trim to the longest real sequence
        # in this batch so short lyrics do not pay for padding.
        length = max(int(attention_mask.sum(dim=1).max()), 1)
        input_ids = input_ids[:, :length]
        attention_mask = attention_mask[:, :length]
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=self.bf16 and input_ids.is_cuda):
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float()
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden[:, 0]
        return self.head(self.dropout(pooled))

