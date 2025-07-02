import torch
import math
import torch.nn.functional as F
from strhub.models.base import CrossEntropySystem
from .model import RegionGNNCSLRModel
import numpy as np
from torch import Tensor
from itertools import permutations
from typing import Any, Optional, Sequence, Dict, List, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn

class NARRegionGNNCSLRSystem(CrossEntropySystem):
    """System for region-focused GNN-based CSLR"""
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
        # self.rng = np.random.default_rng()
        # perm_num = 6
        # perm_forward = True
        # perm_mirrored = True
        # self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        # self.perm_forward = perm_forward
        # self.perm_mirrored = perm_mirrored
        
    
        
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(self.tokenizer, images, max_length)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
    # def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        images, labels = batch
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits, sa_weights, ca_weights = self.forward(images, max_length=max_len)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        
        # ca_weights: [B, Q, T] = [batch_size, query_len, memory_len]

        # (1) Attention Entropy Loss (light regularization)
        # eps = 1e-8
        # entropy = -torch.sum(ca_weights * torch.log(ca_weights + eps), dim=-1)  # [B, Q]
        # attention_entropy_loss = torch.mean(entropy)

        # (2) Attention Diversity Loss
        # Ép các query trong cùng batch có attention khác nhau
        div_loss = 0.0
        Q = ca_weights.size(1)
        for i in range(Q):
            for j in range(i + 1, Q):
                sim = F.cosine_similarity(ca_weights[:, i, :], ca_weights[:, j, :], dim=-1)  # [B]
                div_loss += torch.mean(sim)  # càng giống nhau thì càng bị phạt
        div_loss /= (Q * (Q - 1) / 2)  # normalize

        # (3) Local Smoothness Loss
        local_diff = ca_weights[:, :, 1:] - ca_weights[:, :, :-1]  # [B, T, S-1]
        local_smoothness_loss = torch.mean(local_diff.abs())
        # (4) Center of Mass Loss
        S = ca_weights.size(-1)
        positions = torch.arange(S, device=ca_weights.device).float()  # [S]
        center = torch.sum(ca_weights * positions[None, None, :], dim=-1, keepdim=True)
        variance = torch.sum(ca_weights * (positions[None, None, :] - center) ** 2, dim=-1)
        center_of_mass_loss = torch.mean(variance)
        # (3) Tổng loss
        total_loss = loss \
                + 0.5 * div_loss \
                + 0.5 * local_smoothness_loss \
                + 0.001 * center_of_mass_loss
                # + 0.5 * attention_entropy_loss 
        # print(loss, div_loss, local_smoothness_loss, center_of_mass_loss)
        # input()
        # Combine losses with a weight factor
        # entropy_weight = 0.2  # You can adjust this weight
        # total_loss = loss + entropy_weight * attention_entropy_loss
        
        loss_numel = (targets != self.pad_id).sum()
        self.log("train_loss", total_loss, prog_bar=True)
        # self.log("attention_entropy_loss", attention_entropy_loss, prog_bar=True)
        return total_loss