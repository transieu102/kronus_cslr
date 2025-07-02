import torch
import math
import torch.nn.functional as F
from strhub.models.base import CrossEntropySystem, BatchResult, EPOCH_OUTPUT
from .model import MultiRegionCSLRModel
import numpy as np
from torch import Tensor
from itertools import permutations
from typing import Any, Optional, Sequence, Dict, List, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
from utils.metrics import wer_single_list as wer_single
class MultiRegionCSLRSystemRGB(CrossEntropySystem):
    """System for multi-region CSLR with individual region training"""
    def __init__(
        self,
        tokenizer,
        config: Dict[str, Any]
    ):
        # Extract training parameters
        batch_size = config["trainer"]["batch_size"]
        lr = config["trainer"]["lr"]
        warmup_pct = config["trainer"]["warmup_pct"]
        weight_decay = config["trainer"]["weight_decay"]
        
        # Extract model parameters
        model_config = config["model"]
        input_dim = model_config["input_dim"]
        d_model = model_config["d_model"]
        num_encoder_layers = model_config["num_encoder_layers"]
        num_decoder_layers = model_config["num_decoder_layers"]
        encoder_num_heads = model_config["encoder_num_heads"]
        decoder_num_heads = model_config["decoder_num_heads"]
        dec_mlp_ratio = model_config["dec_mlp_ratio"]
        dim_feedforward = model_config["dim_feedforward"]
        num_classes = len(tokenizer)
        dropout = model_config["dropout"]
        max_input_len = model_config["max_input_len"]
        max_output_len = model_config["max_output_len"]
        refine_iters = model_config["refine_iters"]
        # GNN specific parameters
        gnn_type = model_config.get("gnn_type", "gcn")
        num_gnn_layers = model_config.get("num_gnn_layers", 2)
        gnn_hidden_dim = model_config.get("gnn_hidden_dim", 256)
        gnn_feature_dim = model_config.get("gnn_feature_dim", 64)
        region_feature_dim = model_config.get("region_feature_dim", 256)
        # Initialize base class
        super().__init__(tokenizer, batch_size, lr, warmup_pct, weight_decay)
        
        # Create model
        self.model = MultiRegionCSLRModel(
            input_dim=input_dim,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            encoder_num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
            dec_mlp_ratio=dec_mlp_ratio,
            refine_iters=refine_iters,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            dropout=dropout,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_feature_dim=gnn_feature_dim,
            region_feature_dim=region_feature_dim
        )
        
        # Permutation settings
        self.rng = np.random.default_rng()
        perm_num = 6
        perm_forward = True
        perm_mirrored = True
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
    def forward_logits_loss(self, images: Tensor, rgb_frames: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits = self.forward(images, rgb_frames, max_length=max_len)
        # print(logits.shape)
        # print(targets.shape)
        # input()
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel
    def _eval_step(self, batch, validation: bool) -> Optional[STEP_OUTPUT]:
        images, rgb_frames, labels = batch

        correct = 0
        total = 0
        wer = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, rgb_frames, labels)
        else:
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            logits = self.forward(images, rgb_frames)
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
        
    def forward(self, images: Tensor, rgb_frames: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(self.tokenizer, images, rgb_frames, max_length)
   
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, rgb_frames, labels = batch  # Unpack batch to include RGB frames
        bs = images.shape[0]
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode all streams
        pose_memory, rgb_memory, fusion_memory, region_features = self.model.encode(images, rgb_frames)

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
            
            # 1. Pose Stream
            # Decode pose stream
            # out = self.model.decode_stream(tgt_in, pose_memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            # out = self.model.decode_stream(tgt_in, pose_memory,self.model.pose_pos_queries,self.model.pose_decoder,self.model.pose_out_proj,tgt_mask,tgt_padding_mask, tgt_query_mask=query_mask)
            out = self.model.decode_stream(
                tgt_in,
                pose_memory,
                self.model.pose_pos_queries,
                self.model.pose_decoder,
                self.model.pose_out_proj,
                tgt_mask,
                tgt_padding_mask,
                tgt_query=self.model.pose_pos_queries[:, :tgt_in.shape[1]].expand(bs, -1, -1),
                tgt_query_mask=query_mask,
                token_embedding=self.model.token_embedding_pose
            )
            logits = out.flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n

            # Region-specific losses
            for i, (region_name, joint_indices) in enumerate(self.model.regions.items()):
                region_feat = region_features[i]
                region_output = self.model.region_models[region_name].decode(
                    tgt_in, region_feat, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask
                )
                region_logits = self.model.region_models[region_name].out_proj(region_output).flatten(end_dim=1)
                loss += n * F.cross_entropy(region_logits, tgt_out.flatten(), ignore_index=self.pad_id) * 0.5
                loss_numel += n

            # 2. RGB Stream
            rgb_out = self.model.decode_stream(
                tgt_in,
                rgb_memory,
                self.model.rgb_pos_queries,
                self.model.rgb_decoder,
                self.model.rgb_out_proj,
                tgt_mask,
                tgt_padding_mask,
                tgt_query=self.model.rgb_pos_queries[:, :tgt_in.shape[1]].expand(bs, -1, -1),
                tgt_query_mask=query_mask,
                token_embedding=self.model.token_embedding_rgb
            )
            rgb_logits = rgb_out.flatten(end_dim=1)
            loss += n * F.cross_entropy(rgb_logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n

            # 3. Fusion Stream
            fusion_out = self.model.decode_stream(
                tgt_in,
                fusion_memory,
                self.model.fusion_pos_queries,
                self.model.fusion_decoder,
                self.model.fusion_out_proj,
                tgt_mask,
                tgt_padding_mask,
                tgt_query=self.model.fusion_pos_queries[:, :tgt_in.shape[1]].expand(bs, -1, -1),
                tgt_query_mask=query_mask,
                token_embedding=self.model.token_embedding_fusion
            )
            fusion_logits = fusion_out.flatten(end_dim=1)
            loss += n * F.cross_entropy(fusion_logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n

            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()

        loss /= loss_numel

        # Log individual losses for monitoring
        self.log('pose_loss', loss)
        self.log('rgb_loss', loss)
        self.log('fusion_loss', loss)
        self.log('total_loss', loss)

        return loss