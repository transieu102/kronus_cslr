import torch
import math
import torch.nn.functional as F
from strhub.models.base import CrossEntropySystem
from .model import GNN_AR_Model
import numpy as np
from torch import Tensor
from itertools import permutations
from typing import Any, Optional, Sequence
from pytorch_lightning.utilities.types import STEP_OUTPUT

class GNN_AR_System(CrossEntropySystem):
    def __init__(self, tokenizer, config):
        batch_size = config["trainer"]["batch_size"]
        lr = config["trainer"]["lr"]
        warmup_pct = config["trainer"]["warmup_pct"]
        weight_decay = config["trainer"]["weight_decay"]
        input_dim = config["model"]["input_dim"]
        hidden_dim = config["model"]["hidden_dim"]
        num_layers = config["model"]["num_layers"]
        encoder_num_heads = config["model"]["encoder_num_heads"]
        decoder_num_heads = config["model"]["decoder_num_heads"]
        conv_channels = config["model"]["conv_channels"]
        mlp_hidden = config["model"]["mlp_hidden"]
        num_classes = len(tokenizer)
        dropout = config["model"]["dropout"]
        max_input_len = config["model"]["max_input_len"]
        max_output_len = config["model"]["max_output_len"]
        num_decoder_layers = config["model"]["num_decoder_layers"]
        dec_mlp_ratio = config["model"]["dec_mlp_ratio"]
        refine_iters = config["model"]["refine_iters"]
        epochs = config["trainer"]["epochs"]
        # train_refine_epoch = config["trainer"].get("train_refine_epoch", 0)
        
        # GNN specific parameters
        gnn_type = config["model"].get("gnn_type", "gcn")  # 'gcn', 'gat', 'sage', 'gin'
        num_gnn_layers = config["model"].get("num_gnn_layers", 2)
        gnn_hidden_dim = config["model"].get("gnn_hidden_dim", 256)
        gnn_feature_dim = config["model"].get("gnn_feature_dim", 64)
        # gnn_pool_type = config["model"].get("gnn_pool_type", "mean")  # 'mean', 'max', 'sum'
        gnn_dropout = config["model"].get("gnn_dropout", 0.1)
        gnn_kwargs = config["model"].get("gnn_kwargs", {})  # Additional GNN parameters
        
        super().__init__(tokenizer, batch_size, lr, warmup_pct, weight_decay)
        self.model = GNN_AR_Model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
            conv_channels=conv_channels,
            mlp_hidden=mlp_hidden,
            num_classes=num_classes,
            dropout=dropout,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            num_decoder_layers=num_decoder_layers,
            dec_mlp_ratio=dec_mlp_ratio,
            refine_iters=refine_iters,
            # GNN specific parameters
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_feature_dim=gnn_feature_dim,
            gnn_dropout=gnn_dropout,
            **gnn_kwargs
        )
        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        perm_num = 6
        perm_forward = True
        perm_mirrored = True
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        # self.train_refine_epoch = train_refine_epoch
        # self.epochs = epochs
        # self.current_epoch_index = -1
    # def forward(self, poses, max_length=None):
    #     return self.model(self.tokenizer, poses, max_length)

    # def forward_logits_loss(self, poses, labels):
    #     targets = self.tokenizer.encode(labels, self.device)
    #     targets = targets[:, 1:]  # Remove <bos>
    #     max_len = targets.shape[1] - 1  # exclude <eos>
    #     logits = self(poses, max_length=max_len)
    #     loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.tokenizer.pad_id)
    #     loss_numel = (targets != self.tokenizer.pad_id).sum()
    #     return logits, loss, loss_numel

    # def training_step(self, batch, batch_idx):
    #     poses, labels = batch
    #     logits, loss, loss_numel = self.forward_logits_loss(poses, labels)
    #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     return loss 

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
        This works because the same attention mask can be used for the shorter sequences
        because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=self._device,
            )[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))]
            )
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1 :]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = True  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(self.tokenizer, images, max_length)
    
    # def training_full_step(self, batch, batch_idx) -> STEP_OUTPUT:
    #     images, labels = batch
    #     targets = self.tokenizer.encode(labels, self.device)
    #     targets = targets[:, 1:]  # Discard <bos>
    #     max_len = targets.shape[1] - 1  # exclude <eos> from count
    #     logits = self.forward(images, max_length=max_len)
    #     # print(logits.shape)
    #     # print(targets.shape)
    #     # input(
    #     loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
    #     self.log('loss_refine', loss)
    #     return loss
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # if self.epochs - self.current_epoch_index <= self.train_refine_epoch:
        #     return self.training_full_step(batch, batch_idx)
        # if batch_idx == 0:
        #     self.current_epoch_index += 1
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.model.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.model.out_proj(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        self.log('loss', loss)
        return loss