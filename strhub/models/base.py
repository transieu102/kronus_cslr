import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.optim import create_optimizer_v2

from strhub.data.utils import BaseGlossTokenizer, CTCGlossTokenizer, GlossTokenizer
from utils.metrics import wer_single_list as wer_single

@dataclass
class BatchResult:
    num_samples: int
    correct: int
    wer: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int


EPOCH_OUTPUT = list[dict[str, BatchResult]]


class BaseSystem(pl.LightningModule, ABC):

    def __init__(
        self,
        tokenizer: BaseGlossTokenizer,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay
        self.outputs: EPOCH_OUTPUT = []

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """Inference

        Args:
            images: Batch of skeleton. Shape: B, T, J, D
            max_length: Max sequence length of the output. If None, will use default.

        Returns:
            logits: B, L, C (L = sequence length, C = number of classes, typically len(gloss_tokenizer)
        """
        raise NotImplementedError

    @abstractmethod
    def forward_logits_loss(self, images: Tensor, labels: list[list[str]]) -> tuple[Tensor, Tensor, int]:
        """Like forward(), but also computes the loss (calls forward() internally).

        Args:
            images: Batch of skeleton. Shape: B, T, J, D
            labels: List of list of gloss tokens. Shape: B, L (L = length of the gloss sequence)

        Returns:
            logits: B, L, C (L = sequence length, C = number of classes, typically len(gloss_tokenizer))
            loss: mean loss for the batch
            loss_numel: number of elements the loss was calculated from
        """
        raise NotImplementedError

    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.0
        lr = lr_scale * self.lr
        optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(
            optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct, cycle_momentum=False
        )
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _eval_step(self, batch, validation: bool) -> Optional[STEP_OUTPUT]:
        images, labels = batch

        correct = 0
        total = 0
        wer = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        else:
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            logits = self.forward(images)
            loss = loss_numel = None  # Only used for validation; not needed at test-time.

        probs = logits.softmax(-1)
        # print("logit:", logits.shape)
        preds, probs = self.tokenizer.decode(probs)
        # print("preds:", preds)
        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item()
            # Follow ICDAR 2019 definition of N.E.D.
            # print(gt, '||', pred)
            wer += wer_single(gt.split(), pred)['wer']
            # print(gt, pred, wer)
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, wer, confidence, label_length, loss, loss_numel))

    @staticmethod
    def _aggregate_results(outputs: EPOCH_OUTPUT) -> tuple[float, float, float]:
        if not outputs:
            return 0.0, 0.0, 0.0
        total_loss = 0
        total_loss_numel = 0
        total_n_correct = 0
        total_wer = 0
        total_size = 0
        for result in outputs:
            result = result['output']
            total_loss += result.loss_numel * result.loss
            total_loss_numel += result.loss_numel
            total_n_correct += result.correct
            total_wer += result.wer
            total_size += result.num_samples
        acc = total_n_correct / total_size
        wer = total_wer / total_size
        loss = total_loss / total_loss_numel
        return acc, wer, loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        result = self._eval_step(batch, True)
        self.outputs.append(result)
        return result

    def on_validation_epoch_end(self) -> None:
        acc, wer, loss = self._aggregate_results(self.outputs)
        self.outputs.clear()
        # self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_wer', wer, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        # self.log('hp_metric', wer, sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, False)


class CrossEntropySystem(BaseSystem):

    def __init__(
        self, tokenizer: GlossTokenizer, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
    ) -> None:
        super().__init__(tokenizer, batch_size, lr, warmup_pct, weight_decay)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits = self.forward(images, max_length=max_len)
        # print(logits.shape)
        # print(targets.shape)
        # input()
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel


class CTCSystem(BaseSystem):

    def __init__(
        self, tokenizer: CTCGlossTokenizer, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
    ) -> None:
        super().__init__(tokenizer, batch_size, lr, warmup_pct, weight_decay)
        self.blank_id = tokenizer.blank_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets, target_lengths = self.tokenizer.encode(labels, self.device)
        # print("Targets shape:", targets.shape)
        # print("Targets values:", targets)
        # print("Labels:", labels)
        
        logits = self.forward(images)
        # print("Logits shape:", logits.shape)
        
        log_probs = logits.log_softmax(-1).transpose(0, 1)  # swap batch and seq. dims
        # print("Log probs shape:", log_probs.shape)
        
        T, N, _ = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        # target_lengths = torch.as_tensor(list(map(len, [label.split() for label in labels])), dtype=torch.long, device=self.device)
        
        # print("Input lengths:", input_lengths)
        # print("Target lengths:", target_lengths)
        
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        return logits, loss, N

class NARCTCSystem(BaseSystem):

    def __init__(
        self, tokenizer_ctc: CTCGlossTokenizer, tokenizer_entropy: GlossTokenizer, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
    ) -> None:
        super().__init__(tokenizer_ctc, batch_size, lr, warmup_pct, weight_decay)
        self.blank_id = tokenizer_ctc.blank_id
        self.tokenizer_entropy = tokenizer_entropy
        self.bos_id = tokenizer_entropy.bos_id
        self.eos_id = tokenizer_entropy.eos_id
        self.pad_id = tokenizer_entropy.pad_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        # print("Targets shape:", targets.shape)
        # print("Targets values:", targets)
        # print("Labels:", labels)
        
        logits = self.forward(images)
        # print("Logits shape:", logits.shape)
        
        log_probs = logits.log_softmax(-1).transpose(0, 1)  # swap batch and seq. dims
        # print("Log probs shape:", log_probs.shape)
        
        T, N, _ = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        target_lengths = torch.as_tensor(list(map(len, [label.split() for label in labels])), dtype=torch.long, device=self.device)
        
        # print("Input lengths:", input_lengths)
        # print("Target lengths:", target_lengths)
        
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        return logits, loss, N




# class CrossEntropySystem_MultiSupervision(BaseSystem):

#     def __init__(
#         self, tokenizer: GlossTokenizer, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
#     ) -> None:
#         super().__init__(tokenizer, batch_size, lr, warmup_pct, weight_decay)
#         self.bos_id = tokenizer.bos_id
#         self.eos_id = tokenizer.eos_id
#         self.pad_id = tokenizer.pad_id

#     def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
#         targets = self.tokenizer.encode(labels, self.device)
#         targets = targets[:, 1:]  # Discard <bos>
#         max_len = targets.shape[1] - 1  # exclude <eos> from count
#         logits_list = self.forward(images, max_length=max_len)
#         # print(logits.shape)
#         # print(targets.shape)
#         # input()
#         #logit 0 là final of model, các logit sau là sub path way 
#         total_loss = 0
#         total_loss_numel = 0
#         for logits in logits_list:
#             loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
#             loss_numel = (targets != self.pad_id).sum()
#             total_loss += loss
#             total_loss_numel += loss_numel
#         return logits_list[0], total_loss, total_loss_numel