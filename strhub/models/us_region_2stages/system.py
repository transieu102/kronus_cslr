import torch
import math
import torch.nn.functional as F
from strhub.models.base import CTCSystem
from .model import RegionGNNCSLRModel
import numpy as np
from torch import Tensor
from typing import Any, Optional, Sequence, Dict, List, Union, Tuple
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn

class RegionGNNSystem2Stage(CTCSystem):
    """System for region-focused GNN-based CSLR with CTC loss"""
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
        dec_hidden_dim = model_config.get("dec_hidden_dim", 512)  # Hidden dimension for BiLSTM
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
            dec_hidden_dim=dec_hidden_dim,
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
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(
            blank=self.tokenizer.blank_id,
            zero_infinity=True,
            reduction='mean'
        )
        
    def forward(self, images: Tensor) -> Tensor:
        """Forward pass for inference"""
        return self.model.forward(self.tokenizer, images)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        targets, target_lengths = self.tokenizer.encode(labels, self.device)

        # Encode images
        first_memory, second_memory, region_features = self.model.encode(images,return_region_features= True)
        first_logits = self.model.out_proj(first_memory)  # [B, T/4, num_classes]
        second_logits = self.model.out_proj(second_memory)  # [B, T/4, num_classes]
        # second_logits, first_logits = self.model(self.tokenizer, images, return_first_logits=True)  # [B, T/4, vocab]
        log_probs = second_logits.log_softmax(-1).transpose(0, 1)  # [T, B, V]

        # CTC Loss
        T, N, V = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        # target_lengths = torch.tensor([len(label.split()) for label in labels], dtype=torch.long, device=self.device)

        ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        first_ctc_loss = F.ctc_loss(first_logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        
        # ctc_va_loss = compute_va_loss(second_logits, first_logits)
        total_loss = ctc_loss + 0.1*first_ctc_loss 
        region_loss_weight = 0.05
        for region_id, region_info in enumerate(self.model.regions.items()):
            region_name, _ = region_info
            region_feature = region_features[region_name].permute(0, 2, 1)  # [B, region_feature_dim, T/4]
            region_feature = self.model.region_decoder[region_name](region_feature).permute(0, 2, 1)  # [B, T/4, region_feature_dim]
            # region_logits = self.model.region_decoder[region_name](region_feature).permute(0, 2, 1)  # [B, T/4, num_classes]
            region_logits = self.model.region_out_proj[region_name](region_feature)
            region_logits = region_logits.log_softmax(-1).transpose(0, 1)
            region_loss = F.ctc_loss(region_logits, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
            total_loss += region_loss * region_loss_weight 

        self.log_dict({
            'loss': total_loss,
            'second_loss': ctc_loss,
            'first_loss': first_ctc_loss,
            # 'va_loss': ctc_va_loss
            # 'gfe_loss': gfe_loss,
            # 'br': br,
        }, prog_bar=True)
        return total_loss

        