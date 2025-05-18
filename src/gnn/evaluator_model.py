import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TransformerBlock(nn.Module):
    """Transformer block implementation."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attended, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection
        x = self.norm2(x + self.ff(x))
        
        return x


class PolicyHead(nn.Module):
    """Policy head to predict move probabilities."""
    
    def __init__(self, node_dim, hidden_dim, num_transformer_layers=4, num_heads=8, dropout=0.1):
        super(PolicyHead, self).__init__()
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=node_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(num_transformer_layers)
        ])
        
        # Move prediction layers
        self.move_predictor = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_embeddings, batch_size=None):
        """
        Forward pass of policy head.
        
        Args:
            node_embeddings: Node embeddings from GNN encoder (batch × num_nodes × node_dim)
            batch_size: Batch size for reshaping
            
        Returns:
            Move logits (batch × num_legal_moves)
        """
        # Apply transformer layers
        x = node_embeddings
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Generate all potential move pairs (from_square, to_square)
        # In practice, we would filter to only legal moves during training/inference
        batch_size = batch_size or node_embeddings.size(0)
        num_squares = 64
        
        # Extract just the square node embeddings (first 64 nodes)
        square_embeddings = x[:, :num_squares, :]
        
        # Create all possible from-to combinations
        from_embeddings = square_embeddings.unsqueeze(2).expand(-1, -1, num_squares, -1)
        to_embeddings = square_embeddings.unsqueeze(1).expand(-1, num_squares, -1, -1)
        
        # Concatenate from-to embeddings
        move_embeddings = torch.cat([from_embeddings, to_embeddings], dim=-1)
        
        # Reshape for the move predictor
        move_embeddings = move_embeddings.view(batch_size, num_squares * num_squares, -1)
        
        # Predict move logits
        move_logits = self.move_predictor(move_embeddings).squeeze(-1)
        
        return move_logits


class ValueHead(nn.Module):
    """Value head to predict position evaluation."""
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(ValueHead, self).__init__()
        
        self.transformer_layer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=8,
            ff_dim=hidden_dim,
            dropout=dropout
        )
        
        # Value prediction layers with Monte-Carlo dropout
        self.value_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Uncertainty prediction
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, global_embedding, mc_samples=1):
        """
        Forward pass of value head with Monte-Carlo dropout for uncertainty estimation.
        
        Args:
            global_embedding: Global graph embedding (batch × embed_dim)
            mc_samples: Number of Monte-Carlo samples for uncertainty estimation
            
        Returns:
            Tuple of (value, uncertainty)
        """
        # Process with transformer
        x = global_embedding.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_layer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        
        if mc_samples > 1 and self.training:
            # Monte-Carlo sampling for uncertainty estimation
            values = []
            for _ in range(mc_samples):
                # Each forward pass will have different dropout patterns
                value = self.value_predictor(x)
                values.append(value)
            
            # Stack the samples
            values = torch.stack(values, dim=1)  # (batch, samples, 1)
            
            # Compute mean and variance
            mean_value = values.mean(dim=1)
            uncertainty = values.var(dim=1).sqrt()  # Standard deviation as uncertainty
            
            return mean_value, uncertainty
        else:
            # Standard forward pass
            value = self.value_predictor(x)
            uncertainty = self.uncertainty_predictor(x)
            
            return value, uncertainty


class DualHeadEvaluator(nn.Module):
    """Dual-head transformer-GNN evaluator."""
    
    def __init__(self, node_dim=128, hidden_dim=256, policy_layers=4, value_layers=2):
        super(DualHeadEvaluator, self).__init__()
        
        # Policy head
        self.policy_head = PolicyHead(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_transformer_layers=policy_layers
        )
        
        # Value head
        self.value_head = ValueHead(
            embed_dim=node_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(self, node_embeddings, global_embedding, batch_size=None, mc_samples=1):
        """
        Forward pass through both heads.
        
        Args:
            node_embeddings: Node embeddings from GNN encoder
            global_embedding: Global graph embedding
            batch_size: Batch size for reshaping
            mc_samples: Monte-Carlo samples for value head
            
        Returns:
            Tuple of (move_logits, value, uncertainty)
        """
        # Policy prediction
        move_logits = self.policy_head(node_embeddings, batch_size)
        
        # Value prediction
        value, uncertainty = self.value_head(global_embedding, mc_samples)
        
        # Scale value from [-1, 1] to centipawn range [-2000, 2000]
        value_cp = value * 2000
        uncertainty_cp = uncertainty * 2000
        
        return move_logits, value_cp, uncertainty_cp


class ChessModel(nn.Module):
    """Complete chess model with GNN encoder and dual-head evaluator."""
    
    def __init__(self, gnn_encoder, node_dim=128, hidden_dim=256):
        super(ChessModel, self).__init__()
        
        self.gnn_encoder = gnn_encoder
        self.evaluator = DualHeadEvaluator(
            node_dim=node_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(self, graph_batch, mc_samples=1):
        """
        Forward pass through the complete model.
        
        Args:
            graph_batch: Batch of graph data
            mc_samples: Monte-Carlo samples for value head
            
        Returns:
            Tuple of (move_logits, value, uncertainty)
        """
        # Encode the board into node and global embeddings
        node_embeddings, global_embedding = self.gnn_encoder(graph_batch)
        
        # Get batch size
        batch_size = graph_batch.num_graphs
        
        # Feed through evaluator
        move_logits, value, uncertainty = self.evaluator(
            node_embeddings, 
            global_embedding, 
            batch_size=batch_size,
            mc_samples=mc_samples
        )
        
        return move_logits, value, uncertainty
    
    def get_move_probabilities(self, graph_batch, legal_moves_mask):
        """
        Convert move logits to probabilities considering only legal moves.
        
        Args:
            graph_batch: Batch of graph data
            legal_moves_mask: Boolean mask of legal moves (batch × 4096)
            
        Returns:
            Move probabilities (batch × 4096)
        """
        move_logits, _, _ = self.forward(graph_batch)
        
        # Apply mask to set illegal moves to -inf
        masked_logits = move_logits.clone()
        masked_logits[~legal_moves_mask] = float('-inf')
        
        # Convert to probabilities with softmax
        move_probs = F.softmax(masked_logits, dim=-1)
        
        return move_probs 