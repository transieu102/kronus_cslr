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
    def __init__(self, d_model: int, max_len: int = 200):
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

class RGBStream(nn.Module):
    """RGB video stream processing"""
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        # CNN for frame feature extraction
        self.cnn = nn.Sequential(
            # Input: 112x112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to 1x1
        )

        
        # Position encoding
        self.pos_encoder = PosEnc(256, max_len=200)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=256*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # self.mlp = nn.Sequential(
        #     nn.Linear(256, d_model),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        # )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, C, H, W] where H=W=112
        Returns:
            Output tensor [B, T/9, d_model]
        """
        B, T, C, H, W = x.shape
        
        # CNN feature extraction
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)  # [B*T, 256, 1, 1]
        x = x.view(B, T, 256, 1, 1).squeeze(-1).squeeze(-1)  # [B, T, 256]
        
        # # Temporal pooling
        # x = x.transpose(1, 2)  # [B, 256, T]
        # x = self.temporal_pool(x)  # [B, d_model, T/9]
        # x = x.transpose(1, 2)  # [B, T/9, d_model]
        # print(x.shape)
        # input()
        # Position encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # x = self.mlp(x)
        
        return x

class FusionStream(nn.Module):
    """Fusion stream for combining pose and RGB features"""
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        # Position encoding
        self.pos_encoder = PosEnc(d_model)  # Double size for concatenated features
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, pose_features: torch.Tensor, rgb_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_features: [B, T/9, d_model]
            rgb_features: [B, T/9, d_model]
        Returns:
            Output tensor [B, T/9, d_model*2]
        """
        # Concatenate features
        x = torch.cat([pose_features, rgb_features], dim=-1)  # [B, T/9, d_model*2]
        
        # Position encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        return x

        
class MultiRegionCSLRModel(nn.Module):
    """CSLR model with multiple SingleRegionModel for each region and RGB stream"""
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
    
        # RGB Stream
        d_model_rgb = 256
        self.rgb_stream = RGBStream(d_model=d_model_rgb, dropout=dropout)
        
        # Fusion Stream
        self.fusion_stream = FusionStream(d_model=d_model+d_model_rgb, dropout=dropout)
        
        # Position queries for each stream
        self.pose_pos_queries = nn.Parameter(torch.randn(1, max_output_len + 1, d_model))
        self.rgb_pos_queries = nn.Parameter(torch.randn(1, max_output_len + 1, d_model_rgb))
        self.fusion_pos_queries = nn.Parameter(torch.randn(1, max_output_len + 1, d_model+d_model_rgb))
        
        # Decoders for each stream
        self.pose_decoder = Decoder(
            DecoderLayer(d_model, decoder_num_heads, d_model * dec_mlp_ratio, dropout),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.rgb_decoder = Decoder(
            DecoderLayer(d_model_rgb, decoder_num_heads, d_model_rgb * dec_mlp_ratio, dropout),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model_rgb)
        )
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(2*d_model, d_model),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        # )
        self.fusion_decoder = Decoder(
            DecoderLayer(d_model+d_model_rgb, decoder_num_heads, (d_model+d_model_rgb) * dec_mlp_ratio, dropout),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model+d_model_rgb)
        )
        
        # print(num_decoder_layers)
        # input()
        # Output projections for each stream
        self.pose_out_proj = nn.Linear(d_model, num_classes)
        self.rgb_out_proj = nn.Linear(d_model_rgb, num_classes)
        self.fusion_out_proj = nn.Linear(d_model+d_model_rgb, num_classes)
        
        # Token embedding
        self.token_embedding_pose = TokenEmbedding(num_classes, d_model)
        self.token_embedding_rgb = TokenEmbedding(num_classes, d_model_rgb)
        self.token_embedding_fusion = TokenEmbedding(num_classes, d_model+d_model_rgb)
        
        self.num_classes = num_classes
        self.max_output_len = max_output_len
        self.dropout = nn.Dropout(p=dropout)
        self.refine_iters = refine_iters
        
    @property
    def _device(self) -> torch.device:
        return next(self.fusion_out_proj.parameters(recurse=False)).device
        
    def encode(self, poses: torch.Tensor, rgb_frames: torch.Tensor) -> tuple:
        """
        Args:
            poses: [B, T, J, D] where D=2 (x,y coordinates)
            rgb_frames: [B, T, C, H, W] RGB video frames
        Returns:
            pose_memory: [B, T/9, d_model]
            rgb_memory: [B, T/9, d_model]
            fusion_memory: [B, T/9, d_model*2]
            region_features: List of [B, T/9, region_feature_dim] for each region
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
        
        # Concatenate all region features for pose stream
        pose_features = torch.cat(region_features, dim=-1)  # [B, T/9, num_regions*region_feature_dim]
        
        # Process RGB stream
        rgb_features = self.rgb_stream(rgb_frames)  # [B, T/9, d_model]
        
        # Process fusion stream
        fusion_features = self.fusion_stream(pose_features, rgb_features)  # [B, T/9, d_model*2]

        # Process fusion stream
        # fusion_features = self.mlp(fusion_features)  # [B, T/9, d_model]
        
        return pose_features, rgb_features, fusion_features, region_features
    
    def decode_stream(self, tgt: torch.Tensor, memory: torch.Tensor, pos_queries: torch.Tensor,
                     decoder: Decoder, out_proj: nn.Linear, tgt_mask: Optional[torch.Tensor] = None,
                     tgt_padding_mask: Optional[torch.Tensor] = None, tgt_query: Optional[torch.Tensor] = None,
                     tgt_query_mask: Optional[torch.Tensor] = None, token_embedding: TokenEmbedding = None) -> torch.Tensor:
        """
        Decode for a single stream
        """
        N, L = tgt.shape
        
        # Token embedding
        # null_ctx = self.token_embedding(tgt[:, :1])
        # tgt_emb = pos_queries[:, :L-1] + self.token_embedding(tgt[:, 1:])
        # tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        
        null_ctx = token_embedding(tgt[:, :1])
        tgt_emb = pos_queries[:, :L-1] + token_embedding(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        
        if tgt_query is None:
            tgt_query = pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        
        # Decode
        out = decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)
        
        # Project to logits
        logits = out_proj(out)
        
        return logits
    
    def forward(self, tokenizer: Tokenizer, poses: torch.Tensor, rgb_frames: torch.Tensor,
                max_length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            poses: [B, T, J, D]
            rgb_frames: [B, T, C, H, W]
        Returns:
            fusion_logits: [B, L, num_classes]
        """
        testing = max_length is None
        max_length = self.max_output_len if max_length is None else min(max_length, self.max_output_len)
        bs = poses.shape[0]
        num_steps = max_length + 1
        
        # Encode all streams
        _, _, fusion_memory, _ = self.encode(poses, rgb_frames)
        
        # Prepare masks
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)
        
        # Initialize target sequence
        tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = tokenizer.bos_id
        
        # Decode for each stream
        # pose_logits = []
        # rgb_logits = []
        fusion_logits = []
        
        for i in range(num_steps):
            j = i + 1
            
            # Decode each stream
            # pose_out = self.decode_stream(
            #     tgt_in[:, :j],
            #     pose_memory,
            #     self.pose_pos_queries,
            #     self.pose_decoder,
            #     self.pose_out_proj,
            #     tgt_mask[:j, :j],
            #     tgt_query=self.pose_pos_queries[:, i:j],
            #     tgt_query_mask=query_mask[i:j, :j]
            # )
            
            # rgb_out = self.decode_stream(
            #     tgt_in[:, :j],
            #     rgb_memory,
            #     self.rgb_pos_queries,
            #     self.rgb_decoder,
            #     self.rgb_out_proj,
            #     tgt_mask[:j, :j],
            #     tgt_query=self.rgb_pos_queries[:, i:j],
            #     tgt_query_mask=query_mask[i:j, :j]
            # )
            
            fusion_out = self.decode_stream(
                tgt_in[:, :j],
                fusion_memory,
                self.fusion_pos_queries,
                self.fusion_decoder,
                self.fusion_out_proj,
                tgt_mask[:j, :j],
                tgt_query=self.fusion_pos_queries[:, :num_steps].expand(bs, -1, -1)[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
                token_embedding=self.token_embedding_fusion
            )
            
            # pose_logits.append(pose_out)
            # rgb_logits.append(rgb_out)
            fusion_logits.append(fusion_out)
            
            if j < num_steps:
                tgt_in[:, j] = fusion_out.squeeze().argmax(-1)
                if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                    break
        
        # Concatenate logits
        # pose_logits = torch.cat(pose_logits, dim=1)
        # rgb_logits = torch.cat(rgb_logits, dim=1)
        fusion_logits = torch.cat(fusion_logits, dim=1)
        
        # Refinement if needed
        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            
            for i in range(self.refine_iters):
                tgt_in = torch.cat([bos, fusion_logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                
                # Refine each stream
                # pose_out = self.decode_stream(
                #     tgt_in,
                #     pose_memory,
                #     self.pose_pos_queries,
                #     self.pose_decoder,
                #     self.pose_out_proj,
                #     tgt_mask,
                #     tgt_padding_mask,
                #     self.pose_pos_queries,
                #     query_mask[:, :tgt_in.shape[1]]
                # )
                
                # rgb_out = self.decode_stream(
                #     tgt_in,
                #     rgb_memory,
                #     self.rgb_pos_queries,
                #     self.rgb_decoder,
                #     self.rgb_out_proj,
                #     tgt_mask,
                #     tgt_padding_mask,
                #     self.rgb_pos_queries,
                #     query_mask[:, :tgt_in.shape[1]]
                # )
                
                fusion_out = self.decode_stream(
                    tgt_in,
                    fusion_memory,
                    self.fusion_pos_queries,
                    self.fusion_decoder,
                    self.fusion_out_proj,
                    tgt_mask,
                    tgt_padding_mask,
                    self.fusion_pos_queries[:, :num_steps].expand(bs, -1, -1),
                    query_mask[:, :tgt_in.shape[1]],
                    self.token_embedding_fusion
                )
                
                # pose_logits = pose_out
                # rgb_logits = rgb_out
                fusion_logits = fusion_out
        
        return fusion_logits  # Only return fusion logits for inference
   