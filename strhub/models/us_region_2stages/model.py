import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .modules import Decoder, DecoderLayer, TokenEmbedding
from .gnn_modules import RegionGNN
from functools import partial
from typing import Optional, Sequence, Dict, Any, List, Union, Tuple
from torch import Tensor
from strhub.data.utils import CTCGlossTokenizer as Tokenizer
import numpy as np
# import torchaudio

class PosEnc(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, C]
        Returns:
            Positional encoded tensor [B, T, C]
        """
        return x + self.pe[:, :x.size(1)]

class RegionGNNCSLRModel(nn.Module):
    """CSLR model with region-focused GNN encoding and CTC decoding"""
    def __init__(
        self,
        input_dim: int = 86,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 2,  # Number of BiLSTM layers
        encoder_num_heads: int = 8,
        dec_hidden_dim: int = 512,    # Hidden dimension for BiLSTM
        refine_iters: int = 1,
        dim_feedforward: int = 2048,
        num_classes: int = 100,
        dropout: float = 0.1,
        max_input_len: int = 1000,
        max_output_len: int = 100,
        # GNN specific parameters
        gnn_type: str = 'gcn',  # 'gcn', 'gat', 'sage', 'gin'
        num_gnn_layers: int = 2,
        gnn_hidden_dim: int = 256,
        gnn_feature_dim: int = 64,  # Feature dimension per joint
        region_feature_dim: int = 256,  # Fixed dimension for each region
        **kwargs
    ):
        super().__init__()
        
        # Define regions and their joint indices based on dataset structure
        self.regions = {
            'right_hand': list(range(0, 21)),      # 21 joints
            'left_hand': list(range(21, 42)),      # 21 joints
            'lips': list(range(42, 61)),           # 19 joints
            'body': list(range(61, 86))            # 25 joints
        }
        d_model = region_feature_dim * len(self.regions)
        
        # Create GNN encoders for each region
        self.region_encoders = nn.ModuleDict()
        for region_name, joint_indices in self.regions.items():
            self.region_encoders[region_name] = RegionGNN(
                in_channels=2,  # x,y coordinates
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_feature_dim,
                transformer_dim=gnn_feature_dim,  # Project to joint feature dimension
                gnn_type=gnn_type,
                num_layers=num_gnn_layers,
                dropout=dropout,
                **kwargs
            )
            
        # Projection layers for each region to fixed dimension
        self.region_projections = nn.ModuleDict()
        for region_name, joint_indices in self.regions.items():
            num_joints = len(joint_indices)
            self.region_projections[region_name] = nn.Sequential(
                nn.Linear(num_joints * gnn_feature_dim, region_feature_dim),
                nn.LayerNorm(region_feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        self.region_decoder = nn.ModuleDict()
        for region_name, joint_indices in self.regions.items():
            self.region_decoder[region_name] = nn.Sequential(
                nn.Conv1d(region_feature_dim, region_feature_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(region_feature_dim, region_feature_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AvgPool1d(kernel_size=2, stride=2)
            )
        self.region_out_proj = nn.ModuleDict()
        for region_name, joint_indices in self.regions.items():
            self.region_out_proj[region_name] = nn.Linear(region_feature_dim, num_classes)
        
        self.temporal_pooling = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        
        self.second_temporal_pooling = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Output projection
        self.out_proj = nn.Linear(d_model, num_classes)
        self.num_classes = num_classes
        self.max_output_len = max_output_len
        self.dropout = nn.Dropout(p=dropout)
        self.refine_iters = refine_iters
    @property
    def _device(self) -> torch.device:
        return next(self.out_proj.parameters(recurse=False)).device
    def _create_edge_index(self, num_nodes: int) -> torch.Tensor:
        """Create a fully connected edge index for skeleton graph"""
        rows = torch.arange(num_nodes, device=self._device)
        cols = torch.arange(num_nodes, device=self._device)
        rows = rows.view(-1, 1).repeat(1, num_nodes).view(-1)
        cols = cols.repeat(num_nodes)
        edge_index = torch.stack([rows, cols], dim=0)
        return edge_index
        
    def encode(self, poses, return_region_features=False):
        """
        Args:
            poses: [B, T, J, D] where D=2 (x,y coordinates)
        Returns:
            memory: [B, T/4, d_model]
        """
        B, T, J, D = poses.shape
        
        # Process each region separately
        region_features = []
        region_features_to_return = {}
        for region_index, (region_name, joint_indices) in enumerate(self.regions.items()):
            # Extract joints for this region
            region_poses = poses[:, :, joint_indices, :]  # [B, T, num_joints, D]
            
            # Reshape for GNN processing
            x = region_poses.reshape(-1, D)  # [B*T*num_joints, D]
            
            # Create edge index for this region
            edge_index = self._create_edge_index(len(joint_indices))
            
            # Create batch assignment
            batch = torch.arange(B*T, device=self._device).repeat_interleave(len(joint_indices))
            
            # Process through region-specific GNN
            region_feat = self.region_encoders[region_name](x, edge_index, batch)  # [B*T*num_joints, gnn_feature_dim]
            
            # Reshape back to [B, T, num_joints, gnn_feature_dim]
            region_feat = region_feat.reshape(B, T, len(joint_indices), -1)
            
            # Concatenate joint features for this region
            region_feat = region_feat.reshape(B, T, -1)  # [B, T, num_joints*gnn_feature_dim]
            
            # Project to fixed dimension
            region_feat = self.region_projections[region_name](region_feat)  # [B, T, region_feature_dim]
            
            region_features.append(region_feat)
            region_features_to_return[region_name] = region_feat
        
        # Concatenate all region features
        frame_features = torch.cat(region_features, dim=-1)  # [B, T, num_regions*region_feature_dim]
        
        # Perform temporal pooling first
        x = frame_features.permute(0, 2, 1)  # [B, d_model, T]
        init_memory = self.temporal_pooling(x)  # [B, d_model, T/4]
        x = init_memory.permute(0, 2, 1)  # [B, T/4, d_model]
        first_memory = self.mlp(x)  # [B, T/4, d_model]
        second_memory = self.second_temporal_pooling(init_memory)
        second_memory = second_memory.permute(0, 2, 1)
        # Final MLP
        if return_region_features:
            return first_memory, second_memory, region_features_to_return
        return first_memory, second_memory
    
    def forward(self, tokenizer: Tokenizer, images: Tensor, return_first_logits: bool = False) -> Tensor:
        """
        Forward pass for inference
        Args:
            tokenizer: CTC tokenizer
            images: Input skeleton features [B, T, J, 2]
            max_length: Maximum output length (optional)
        Returns:
            Logits [B, T/4, num_classes]
        """
        # Encode images
        first_memory, second_memory = self.encode(images)  # [B, T/4, d_model]
        first_logits = self.out_proj(first_memory)  # [B, T/4, num_classes]
        second_logits = self.out_proj(second_memory)  # [B, T/4, num_classes]
        if return_first_logits:
            return second_logits, first_logits
        else:
            return second_logits

    


    

  