import torch
import math
import torch.nn.functional as F
from strhub.models.base import NARCTCSystem
from .model import RegionGNNCSLRModel
import numpy as np
from torch import Tensor
from itertools import permutations
from typing import Any, Optional, Sequence, Dict, List, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn

class CTC_NAR_RegionGNNSystem(NARCTCSystem):
    """System for region-focused GNN-based CSLR"""
    def __init__(
        self,
        tokenizer_ctc,
        tokenizer_entropy,
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
        num_classes_ctc = len(tokenizer_ctc)
        num_classes_entropy = len(tokenizer_entropy)
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
        super().__init__(tokenizer_ctc, tokenizer_entropy, batch_size, lr, warmup_pct, weight_decay)
        
        # Create model
        self.model = RegionGNNCSLRModel(
            input_dim=input_dim,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            encoder_num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
            dec_mlp_ratio=dec_mlp_ratio,
            refine_iters=refine_iters,
            dim_feedforward=dim_feedforward,
            num_classes_ctc=num_classes_ctc,
            num_classes_entropy=num_classes_entropy,
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
        # self.rng = np.random.default_rng()
        # perm_num = 6
        # perm_forward = True
        # perm_mirrored = True
        # self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        # self.perm_forward = perm_forward
        # self.perm_mirrored = perm_mirrored
        
    
        
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(self.tokenizer, images)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        targets_ctc = self.tokenizer.encode(labels, self.device)
        targets_entropy = self.tokenizer_entropy.encode(labels, self.device)
        targets_entropy = targets_entropy[:, 1:]  # Discard <bos>
        max_len = targets_entropy.shape[1] - 1   # exclude <eos> from count
        logits_ctc, logits_entropy = self.model.forward(self.tokenizer, images, mode='both', max_len=max_len)
        
        #ctc loss
        log_probs_ctc = logits_ctc.log_softmax(-1).transpose(0, 1)
        T, N, _ = log_probs_ctc.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        target_lengths = torch.as_tensor(list(map(len, [label.split() for label in labels])), dtype=torch.long, device=self.device)
        loss_ctc = F.ctc_loss(log_probs_ctc, targets_ctc, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        
        #entropy loss
        # loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_entropy = F.cross_entropy(logits_entropy.flatten(end_dim=1), targets_entropy.flatten(), ignore_index=self.pad_id)
        
        total_loss = loss_ctc + loss_entropy
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("ctc_loss", loss_ctc, prog_bar=True)
        self.log("entropy_loss", loss_entropy, prog_bar=True)
        return total_loss
    
    # def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
    #     images, labels = batch
    #     #training CTC loss
    #     targets_ctc = self.tokenizer.encode(labels, self.device)
    #     # Encode images
    #     time_step_features, frame_features = self.model.encode(images)  # [B, T/4, d_model]
    #     logits = self.model.main_proj(time_step_features)  # [B, T/4, num_classes]
    #     log_probs = logits.log_softmax(-1).transpose(0, 1)  # swap batch and seq. dims
    #     T, N, _ = log_probs.shape
    #     input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
    #     target_lengths = torch.as_tensor(list(map(len, [label.split() for label in labels])), dtype=torch.long, device=self.device)
    #     loss = F.ctc_loss(log_probs, targets_ctc, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)

    #     #training entropy loss
    #     frame_memory = self.model.pos_encoder(frame_features.detach())
    #     frame_memory = self.model.transformer(frame_memory)
    #     frame_memory = self.model.mlp_frame(frame_memory)
    #     bs = images.shape[0]
    #     targets_entropy = self.tokenizer_entropy.encode(labels, self.device)
    #     targets_entropy = targets_entropy[:, 1:]  # Discard <bos>
    #     max_len = targets_entropy.shape[1]   # exclude <eos> from count
    #     pos_queries = self.model.pos_queries[:, :max_len].expand(bs, -1, -1)
    #      # Initialize target sequence with BOS token
    #     tgt_in = torch.full((bs, max_len), self.tokenizer_entropy.pad_id, dtype=torch.long, device=self._device)
    #     tgt_in[:, 0] = self.tokenizer_entropy.bos_id
    #     logits_entropy, _, ca_weights = self.model.decode(
    #         tgt_in, frame_memory, 
    #         tgt_query=pos_queries,
    #         tgt_mask=None,  # No causal mask for non-autoregressive
    #         tgt_query_mask=None,  # No query mask for non-autoregressive
    #     )
    #     logits_entropy = self.model.out_proj(logits_entropy)
        
    #     # Cross entropy loss
    #     loss_entropy = F.cross_entropy(logits_entropy.flatten(end_dim=1), targets_entropy.flatten(), ignore_index=self.pad_id)

    #     # Frame level loss with pseudo labels
    #     frame_logits = self.model.frame_proj(frame_features)  # [B, T, num_classes]
    #     frame_loss = 0.0
    #     with torch.no_grad():
    #         preds = logits_entropy.argmax(dim=-1)  # [B, L]
    #         match_mask = (preds == targets_entropy) | (targets_entropy == self.pad_id)
    #         correct_seq_mask = match_mask.all(dim=1)  # [B]
    #     if correct_seq_mask.any():
    #         pseudo_targets = preds[correct_seq_mask]            # [B', L]
    #         pseudo_ca_weights = ca_weights[correct_seq_mask]    # [B', L, T]
    #         frame_feats = frame_features[correct_seq_mask]      # [B', T, D]

    #         Bp, L, T = pseudo_ca_weights.shape
    #         C = len(self.tokenizer_entropy)

    #         # One-hot encode predicted gloss tokens
    #         mask = (pseudo_targets != self.pad_id).float()       # [B', L]
    #         one_hot = F.one_hot(pseudo_targets, num_classes=C).float() * mask.unsqueeze(-1)  # [B', L, C]

    #         # Transpose attention weights for bmm
    #         ca_weights_T = pseudo_ca_weights.transpose(1, 2)     # [B', T, L]

    #         # Compute soft labels for each frame
    #         soft_targets = torch.bmm(ca_weights_T, one_hot)      # [B', T, C]
    #         soft_targets = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-8)

    #         frame_logits = self.model.frame_proj(frame_feats)        # [B', T, C]
    #         log_probs = F.log_softmax(frame_logits, dim=-1)          # [B', T, C]

    #         # Soft-label cross entropy
    #         loss_ce_soft = -torch.sum(soft_targets * log_probs, dim=-1)  # [B', T]
    #         loss_ce_soft = loss_ce_soft.mean()  # hoặc .sum() / valid_frame nếu cần
    #         frame_loss += loss_ce_soft

        
    #     total_loss = loss + loss_entropy + frame_loss
    #     self.log("train_loss", total_loss, prog_bar=True)
    #     self.log("ctc_loss", loss, prog_bar=True)
    #     self.log("entropy_loss", loss_entropy, prog_bar=True)
    #     self.log("frame_loss", frame_loss, prog_bar=True)
    #     # self.log("correct_samples_ratio", correct_mask.float().mean(), prog_bar=True)
    #     return total_loss