import torch
import math
import torch.nn.functional as F
from strhub.models.base import CrossEntropySystem
from .model import MultiRegionCSLRModel
import numpy as np
from torch import Tensor
from itertools import permutations
from typing import Any, Optional, Sequence, Dict, List, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn

class MultiRegionCSLRSystem(CrossEntropySystem):
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
             
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(self.tokenizer, images, max_length)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # if self.epochs - self.current_epoch_index <= self.train_refine_epoch:
        #     return self.training_full_step(batch, batch_idx)
        # if batch_idx == 0:
        #     self.current_epoch_index += 1
        images, labels = batch
        targets = self.tokenizer.encode(labels, self._device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        # Process each region through its SingleRegionModel
        region_features = []
        single_region_features = []
        for i, (region_name, joint_indices) in enumerate(self.model.regions.items()):
            # Extract joints for this region
            region_poses = images[:, :, joint_indices, :]  # [B, T, num_joints, D]
            
            # Get encoded features from SingleRegionModel
            region_feat = self.model.region_models[region_name].encode(region_poses)  # [B, T/9, region_feature_dim]
            single_region_features.append(region_feat)
            # Add region position embedding
            region_feat = region_feat + self.model.region_pos_emb[i]  # [B, T/9, region_feature_dim]
            
            # Apply region-specific transformer
            region_feat = self.model.region_transformer(region_feat)  # [B, T/9, region_feature_dim]
            
            region_features.append(region_feat)
        
        # Concatenate all region features
        frame_features = torch.cat(region_features, dim=-1)  # [B, T/9, num_regions*region_feature_dim]
        
        # Apply final transformer encoding
        x = self.model.pos_encoder(frame_features)
        x = self.model.transformer(x)  # [B, T/9, d_model]
        
        # Final MLP
        memory = self.model.mlp(x)  # [B, T/9, d_model]
        # return memory
        loss = 0
        loss_numel = 0

        logits = self.model.decode(memory, self.tokenizer, max_length=max_len)
        # logits = self.model.out_proj(out).flatten(end_dim=1)
        loss += F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel += (targets != self.pad_id).sum()
        for i, (region_name, joint_indices) in enumerate(self.model.regions.items()):
            region_feat = single_region_features[i]
            bs = region_feat.shape[0]
            num_steps = max_len + 1
            pos_queries = self.model.region_models[region_name].pos_queries[:, :num_steps].expand(bs, -1, -1)
            tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

            # Khởi tạo sequence đầu vào cho từng batch
            region_tokens = [torch.full((bs, 1), self.tokenizer.bos_id, dtype=torch.long, device=self._device)]
            region_logits = []

            for k in range(num_steps):
                # Ghép các token đã sinh thành input cho bước này
                region_tgt_in = torch.cat(region_tokens, dim=1)
                tgt_out = self.model.region_models[region_name].decode(
                    region_tgt_in,
                    region_feat,
                    tgt_mask[:k+1, :k+1],
                    tgt_query=pos_queries[:, k:k+1],
                    tgt_query_mask=query_mask[k:k+1, :k+1],
                )
                p_i = self.model.region_models[region_name].out_proj(tgt_out)
                region_logits.append(p_i)
                if k < num_steps - 1:
                    next_token = p_i.squeeze().argmax(-1, keepdim=True)
                    region_tokens.append(next_token)

            region_logits = torch.cat(region_logits, dim=1)
            region_loss = F.cross_entropy(region_logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
            loss += region_loss
            loss_numel += (targets != self.pad_id).sum()
        loss /= loss_numel

        self.log('loss', loss)
        return loss