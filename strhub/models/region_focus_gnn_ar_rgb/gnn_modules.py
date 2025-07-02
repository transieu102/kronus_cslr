import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from typing import Optional, Dict, Any, List, Union

class GNNFactory:
    """Factory class to create different types of GNN layers"""
    @staticmethod
    def create_gnn_layer(gnn_type: str, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        """
        Create a GNN layer based on the specified type
        
        Args:
            gnn_type: Type of GNN layer ('gcn', 'gat', 'sage', 'gin')
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            **kwargs: Additional arguments for the GNN layer
        """
        gnn_type = gnn_type.lower()
        if gnn_type == 'gcn':
            return GCNConv(in_channels, out_channels, **kwargs)
        elif gnn_type == 'gat':
            return GATConv(in_channels, out_channels, **kwargs)
        elif gnn_type == 'sage':
            return SAGEConv(in_channels, out_channels, **kwargs)
        elif gnn_type == 'gin':
            return GINConv(nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            ), **kwargs)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

class RegionGNN(nn.Module):
    """GNN module for region-specific skeleton processing"""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        transformer_dim: int,
        gnn_type: str = 'gcn',
        num_layers: int = 2,
        dropout: float = 0.1,
        **gnn_kwargs
    ):
        """
        Args:
            in_channels: Input feature dimension (usually 2 for x,y coordinates)
            hidden_channels: Hidden feature dimension for GNN
            out_channels: Output feature dimension per joint from GNN
            transformer_dim: Dimension for transformer input
            gnn_type: Type of GNN layer ('gcn', 'gat', 'sage', 'gin')
            num_layers: Number of GNN layers
            dropout: Dropout rate
            **gnn_kwargs: Additional arguments for the GNN layer
        """
        super().__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # Create GNN layers
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.gnn_layers.append(
            GNNFactory.create_gnn_layer(
                gnn_type, in_channels, hidden_channels, **gnn_kwargs
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gnn_layers.append(
                GNNFactory.create_gnn_layer(
                    gnn_type, hidden_channels, hidden_channels, **gnn_kwargs
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.gnn_layers.append(
                GNNFactory.create_gnn_layer(
                    gnn_type, hidden_channels, out_channels, **gnn_kwargs
                )
            )
            self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer for each joint to transformer dimension
        self.joint_proj = nn.Linear(out_channels, transformer_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment [N]
        Returns:
            Projected node features [N, transformer_dim]
        """
        # Process through GNN layers
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            x = gnn(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            if i < len(self.gnn_layers) - 1:  # No dropout after last layer
                x = self.dropout(x)
                
        # Project each joint to transformer dimension
        x = self.joint_proj(x)  # [N, transformer_dim]
        
        return x 