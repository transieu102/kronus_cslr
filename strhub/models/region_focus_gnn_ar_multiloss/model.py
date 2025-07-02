import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .modules import Decoder, DecoderLayer, TokenEmbedding
from .single_region_model import SingleRegionModel
from functools import partial
from typing import Optional, Sequence, Dict, Any, List, Union
from torch import Tensor
from strhub.data.utils import GlossTokenizer as Tokenizer

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

class MultiRegionCSLRModel(nn.Module):
    """CSLR model with multiple SingleRegionModel for each region"""
    def __init__(
        self,
        input_dim: int = 86,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        encoder_num_heads: int = 8,
        decoder_num_heads: int = 12,
        dec_mlp_ratio: int = 4,
        refine_iters: int = 1,
        dim_feedforward: int = 2048,
        num_classes: int = 100,
        dropout: float = 0.1,
        max_input_len: int = 1000,
        max_output_len: int = 100,
        # Region specific parameters
        region_feature_dim: int = 256,  # Feature dimension for each region
        # SingleRegionModel parameters
        gnn_type: str = 'gcn',
        num_gnn_layers: int = 2,
        gnn_hidden_dim: int = 256,
        gnn_feature_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        
        # Define regions and their joint indices
        self.regions = {
            'right_hand': list(range(0, 21)),      # 21 joints
            'left_hand': list(range(21, 42)),      # 21 joints
            'lips': list(range(42, 61)),           # 19 joints
            'body': list(range(61, 86))            # 25 joints
        }
        
        # Create SingleRegionModel for each region
        self.region_models = nn.ModuleDict()
        for region_name, joint_indices in self.regions.items():
            self.region_models[region_name] = SingleRegionModel(
                input_dim=len(joint_indices) * 2,  # x,y coordinates
                hidden_dim=region_feature_dim,
                num_layers=2,
                encoder_num_heads=4,
                decoder_num_heads=4,
                conv_channels=region_feature_dim,
                mlp_hidden=region_feature_dim,
                num_classes=num_classes,
                dropout=dropout,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                num_decoder_layers=1,
                dec_mlp_ratio=dec_mlp_ratio,
                refine_iters=0,  # No refinement in individual models
                gnn_type=gnn_type,
                num_gnn_layers=num_gnn_layers,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_feature_dim=gnn_feature_dim,
                **kwargs
            )
        
        # Region feature fusion transformer
        # Create learnable region position embeddings
        self.region_pos_emb = nn.Parameter(torch.randn(len(self.regions), region_feature_dim))
        
        region_encoder_layer = nn.TransformerEncoderLayer(
            d_model=region_feature_dim,
            nhead=4,  # Smaller number of heads for region features
            dim_feedforward=region_feature_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.region_transformer = nn.TransformerEncoder(region_encoder_layer, num_layers=2)
        
        # Final fusion transformer
        d_model = region_feature_dim * len(self.regions)
        self.pos_encoder = PosEnc(d_model, max_len=max_input_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Decoder components
        self.token_embedding = TokenEmbedding(num_classes, d_model)
        decoder_layer = DecoderLayer(d_model, decoder_num_heads, d_model * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model))
        self.out_proj = nn.Linear(d_model, num_classes)
        
        self.num_classes = num_classes
        self.max_output_len = max_output_len
        self.pos_queries = nn.Parameter(torch.randn(1, self.max_output_len + 1, d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.refine_iters = refine_iters
        
    @property
    def _device(self) -> torch.device:
        return next(self.out_proj.parameters(recurse=False)).device
        
    def encode(self, poses):
        """
        Args:
            poses: [B, T, J, D] where D=2 (x,y coordinates)
        Returns:
            memory: [B, T/9, d_model]
        """
        B, T, J, D = poses.shape
        
        # Process each region through its SingleRegionModel
        region_features = []
        for i, (region_name, joint_indices) in enumerate(self.regions.items()):
            # Extract joints for this region
            region_poses = poses[:, :, joint_indices, :]  # [B, T, num_joints, D]
            
            # Get encoded features from SingleRegionModel
            region_feat = self.region_models[region_name].encode(region_poses)  # [B, T/9, region_feature_dim]
            
            # Add region position embedding
            region_feat = region_feat + self.region_pos_emb[i]  # [B, T/9, region_feature_dim]
            
            # Apply region-specific transformer
            region_feat = self.region_transformer(region_feat)  # [B, T/9, region_feature_dim]
            
            region_features.append(region_feat)
        
        # Concatenate all region features
        frame_features = torch.cat(region_features, dim=-1)  # [B, T/9, num_regions*region_feature_dim]
        
        # Apply final transformer encoding
        x = self.pos_encoder(frame_features)
        x = self.transformer(x)  # [B, T/9, d_model]
        
        # Final MLP
        memory = self.mlp(x)  # [B, T/9, d_model]
        return memory
    
    def encode_with_region_logits(self, tokenizer: Tokenizer, images: Tensor, max_length: Optional[int] = None):
        """
        Args:
            poses: [B, T, J, D] where D=2 (x,y coordinates)
        Returns:
            memory: [B, T/9, d_model]
        """
        B, T, J, D = images.shape
        
        # Process each region through its SingleRegionModel
        region_features = []
        region_logits = []
        for i, (region_name, joint_indices) in enumerate(self.regions.items()):
            # Extract joints for this region
            region_poses = images[:, :, joint_indices, :]  # [B, T, num_joints, D]
            
            # Get encoded features from SingleRegionModel
            region_logit, region_feat = self.region_models[region_name].forward_with_memory(tokenizer, region_poses, max_length)  # [B, T/9, region_feature_dim]
            region_logits.append(region_logit)

            # Add region position embedding
            region_feat = region_feat + self.region_pos_emb[i]  # [B, T/9, region_feature_dim]
            
            # Apply region-specific transformer
            region_feat = self.region_transformer(region_feat)  # [B, T/9, region_feature_dim]
            
            region_features.append(region_feat)
        
        # Concatenate all region features
        frame_features = torch.cat(region_features, dim=-1)  # [B, T/9, num_regions*region_feature_dim]
        
        # Apply final transformer encoding
        x = self.pos_encoder(frame_features)
        x = self.transformer(x)  # [B, T/9, d_model]
        
        # Final MLP
        memory = self.mlp(x)  # [B, T/9, d_model]
        return memory, region_logits
        
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.token_embedding(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.token_embedding(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)
    
    def forward(self, tokenizer: Tokenizer, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_output_len if max_length is None else min(max_length, self.max_output_len)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = tokenizer.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            tgt_out = self.decode(
                tgt_in[:, :j],
                memory,
                tgt_mask[:j, :j],
                tgt_query=pos_queries[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
            )
            p_i = self.out_proj(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)
        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.out_proj(tgt_out)

        return logits
    def fusion_forward(self, tokenizer: Tokenizer, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_output_len if max_length is None else min(max_length, self.max_output_len)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory, region_logits = self.encode_with_region_logits(tokenizer, images, max_length)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = tokenizer.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            tgt_out = self.decode(
                tgt_in[:, :j],
                memory,
                tgt_mask[:j, :j],
                tgt_query=pos_queries[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
            )
            p_i = self.out_proj(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1) 
        # Voting: Average logits from main model and region models
        # Stack all logits: [num_models, B, T, num_classes]
        all_logits = [logits] + region_logits
        
        # Find the maximum sequence length among all logits
        max_len = max(logit.size(1) for logit in all_logits)
        
        # Pad all logits to the same length
        padded_logits = []
        for logit in all_logits:
            if logit.size(1) < max_len:
                # Pad with zeros to match the maximum length
                padding = torch.zeros(logit.size(0), max_len - logit.size(1), logit.size(2), device=logit.device)
                padded_logit = torch.cat([logit, padding], dim=1)
                padded_logits.append(padded_logit)
            else:
                padded_logits.append(logit)
        
        # Stack the padded logits
        stacked_logits = torch.stack(padded_logits, dim=0)
        
        # Average logits across all models
        logits = torch.mean(stacked_logits, dim=0)  # [B, T, num_classes]
        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.out_proj(tgt_out)

        
        
        return logits
    
    
    
    
    