import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .modules import Decoder, DecoderLayer, TokenEmbedding
from .gnn_modules import RegionGNN
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

class RegionGNNCSLRModel(nn.Module):
    """CSLR model with region-focused GNN encoding"""
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
        
        self.pos_encoder = PosEnc(d_model, max_len=max_input_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.temporal_pooling = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.token_embedding = TokenEmbedding(num_classes, d_model)
        decoder_layer = DecoderLayer(d_model, decoder_num_heads, d_model * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model))
        self.out_proj = nn.Linear(d_model, num_classes)
        
        self.num_classes = num_classes
        self.max_output_len = max_output_len
        self.pos_queries = nn.Parameter(torch.randn(1, self.max_output_len + 1, region_feature_dim * len(self.regions)))
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
        
    def encode(self, poses):
        """
        Args:
            poses: [B, T, J, D] where D=2 (x,y coordinates)
        Returns:
            memory: [B, T/4, d_model]
        """
        B, T, J, D = poses.shape
        
        # Process each region separately
        region_features = []
        for region_name, joint_indices in self.regions.items():
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
        
        # Concatenate all region features
        frame_features = torch.cat(region_features, dim=-1)  # [B, T, num_regions*region_feature_dim]
        
        # Continue with transformer encoding
        x = self.pos_encoder(frame_features)
        x = self.transformer(x)
        
        # Temporal pooling
        x = x.permute(0, 2, 1)  # [B, d_model, T]
        x = self.temporal_pooling(x)  # [B, d_model, T/4]
        x = x.permute(0, 2, 1)  # [B, T/4, d_model]
        
        # Final MLP
        memory = self.mlp(x)  # [B, T/4, d_model]
        return memory
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
            # Efficient decoding:
            # Input the context up to the ith token. We use only one query (at position = i) at a time.
            # This works because of the lookahead masking effect of the canonical (forward) AR context.
            # Past tokens have no access to future tokens, hence are fixed once computed.
            tgt_out = self.decode(
                tgt_in[:, :j],
                memory,
                tgt_mask[:j, :j],
                tgt_query=pos_queries[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
            )
            # the next token probability is in the output's ith token position
            p_i = self.out_proj(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)
        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # Mask tokens beyond the first EOS token.
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.out_proj(tgt_out)

        return logits
    # def decode(
    #     self,
    #     tgt: torch.Tensor,
    #     memory: torch.Tensor,
    #     tgt_mask: Optional[torch.Tensor] = None,
    #     memory_mask: Optional[torch.Tensor] = None,
    #     tgt_key_padding_mask: Optional[torch.Tensor] = None,
    #     memory_key_padding_mask: Optional[torch.Tensor] = None
    # ) -> torch.Tensor:
    #     """
    #     Args:
    #         tgt: Target sequence [B, L]
    #         memory: Encoded features [B, T, C]
    #         tgt_mask: Target mask [L, L]
    #         memory_mask: Memory mask [L, T]
    #         tgt_key_padding_mask: Target padding mask [B, L]
    #         memory_key_padding_mask: Memory padding mask [B, T]
    #     Returns:
    #         Decoded features [B, L, C]
    #     """
    #     # Add positional encoding
    #     tgt = self.pos_encoder(tgt)
        
    #     # Transformer decoding
    #     x = self.decoder(
    #         tgt,
    #         memory,
    #         tgt_mask=tgt_mask,
    #         memory_mask=memory_mask,
    #         tgt_key_padding_mask=tgt_key_padding_mask,
    #         memory_key_padding_mask=memory_key_padding_mask
    #     )
        
    #     return x
    
    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     tgt: Optional[torch.Tensor] = None,
    #     mask: Optional[torch.Tensor] = None,
    #     tgt_mask: Optional[torch.Tensor] = None,
    #     memory_mask: Optional[torch.Tensor] = None,
    #     tgt_key_padding_mask: Optional[torch.Tensor] = None,
    #     memory_key_padding_mask: Optional[torch.Tensor] = None
    # ) -> torch.Tensor:
    #     """
    #     Args:
    #         x: Input skeleton features [B, T, J, 2]
    #         tgt: Target sequence [B, L]
    #         mask: Padding mask [B, T]
    #         tgt_mask: Target mask [L, L]
    #         memory_mask: Memory mask [L, T]
    #         tgt_key_padding_mask: Target padding mask [B, L]
    #         memory_key_padding_mask: Memory padding mask [B, T]
    #     Returns:
    #         Output logits [B, L, C]
    #     """
    #     # Encode
    #     memory = self.encode(x)
        
    #     # Decode if target is provided
    #     if tgt is not None:
    #         x = self.decode(
    #             tgt,
    #             memory,
    #             tgt_mask=tgt_mask,
    #             memory_mask=memory_mask,
    #             tgt_key_padding_mask=tgt_key_padding_mask,
    #             memory_key_padding_mask=memory_key_padding_mask
    #         )
    #         return self.out_proj(x)
        
    #     # Generate sequence if no target
    #     batch_size = x.size(0)
    #     device = x.device
        
    #     # Initialize with BOS token
    #     tgt = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
    #     # Generate sequence
    #     for _ in range(self.max_output_len):
    #         # Get current output
    #         out = self.decode(
    #             tgt,
    #             memory,
    #             tgt_mask=tgt_mask,
    #             memory_mask=memory_mask,
    #             tgt_key_padding_mask=tgt_key_padding_mask,
    #             memory_key_padding_mask=memory_key_padding_mask
    #         )
    #         out = self.out_proj(out[:, -1:])
            
    #         # Sample next token
    #         probs = F.softmax(out, dim=-1)
    #         next_token = torch.multinomial(probs.squeeze(1), 1)
            
    #         # Append to sequence
    #         tgt = torch.cat([tgt, next_token], dim=1)
            
    #         # Stop if EOS token is generated
    #         if (next_token == self.tokenizer.eos_id).any():
    #             break
        
    #     return tgt
    
    