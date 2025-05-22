import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, GraphConv, GINConv
from torch_geometric.data import Data, Batch

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

class ResidualGNNLayer(nn.Module):
    """GNN layer with residual connections for better gradient flow"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResidualGNNLayer, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=4, concat=False)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer if dimensions don't match
        self.project = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, edge_index, edge_attr=None):
        # Save input for residual connection
        identity = self.project(x)
        
        # Apply convolution
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Add residual connection
        return out + identity

class PositionSpecificFeatureExtractor(nn.Module):
    """Extracts position-specific features for different board patterns"""
    def __init__(self, in_channels, out_channels):
        super(PositionSpecificFeatureExtractor, self).__init__()
        self.pawn_structure = nn.Linear(in_channels, out_channels // 4)
        self.king_safety = nn.Linear(in_channels, out_channels // 4)
        self.piece_mobility = nn.Linear(in_channels, out_channels // 4)
        self.piece_coordination = nn.Linear(in_channels, out_channels // 4)
        
    def forward(self, x, piece_types):
        # Extract features based on piece types
        pawns_mask = (piece_types == 1)
        kings_mask = (piece_types == 6)
        
        # Pawn structure features (for pawns)
        pawn_features = torch.zeros_like(x)
        if pawns_mask.any():
            pawn_features[pawns_mask] = self.pawn_structure(x[pawns_mask])
        
        # King safety features (for kings and adjacent pieces)
        king_features = torch.zeros_like(x)
        if kings_mask.any():
            king_features[kings_mask] = self.king_safety(x[kings_mask])
        
        # Mobility features (for all pieces)
        mobility_features = self.piece_mobility(x)
        
        # Piece coordination features
        coordination_features = self.piece_coordination(x)
        
        # Combine all features
        return torch.cat([pawn_features, king_features, mobility_features, coordination_features], dim=1)

class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block for piece interactions"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = x + ff_output
        
        return x

class EnhancedDualHeadTransformerGNN(nn.Module):
    """Enhanced Dual-Head Transformer-GNN for chess position evaluation"""
    def __init__(self, node_features=128, hidden_dim=256, num_gnn_layers=8, 
                 num_transformer_layers=6, num_heads=8, dropout=0.1):
        super(EnhancedDualHeadTransformerGNN, self).__init__()
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # GNN layers with residual connections
        self.gnn_layers = nn.ModuleList([
            ResidualGNNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # Position-specific feature extractor
        self.feature_extractor = PositionSpecificFeatureExtractor(hidden_dim, hidden_dim * 4)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            MultiHeadAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Global pooling
        self.global_attention_pool = nn.Linear(hidden_dim, 1)
        
        # Value head (regression)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Policy head (move probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)  # Will be projected to legal moves later
        )
        
        # Uncertainty head for search guidance
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, data):
        # Extract graph components
        x, edge_index, batch = data.x, data.edge_index, data.batch
        piece_types = data.piece_types
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
        
        # Extract position-specific features
        pos_features = self.feature_extractor(x, piece_types)
        
        # Reshape for transformer (batch_size, num_nodes, hidden_dim)
        batch_size = batch.max().item() + 1
        node_count = x.size(0) // batch_size
        x = x.reshape(batch_size, node_count, -1)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Flatten back to (total_nodes, hidden_dim)
        x = x.reshape(-1, x.size(-1))
        
        # Global pooling with attention
        attention_weights = self.global_attention_pool(x).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Apply attention weights and pool by batch
        x_weighted = x * attention_weights.unsqueeze(-1)
        global_representation = torch.zeros(batch_size, x.size(-1), device=x.device)
        global_representation.index_add_(0, batch, x_weighted)
        
        # Value prediction
        value = self.value_head(global_representation)
        
        # Policy prediction
        policy = self.policy_head(x)
        
        # Uncertainty prediction
        uncertainty = self.uncertainty_head(global_representation)
        
        return {
            'value': value.squeeze(-1),
            'policy': policy,
            'uncertainty': uncertainty.squeeze(-1)
        }

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for more robust evaluation"""
    def __init__(self, model_configs, node_features=128):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList([
            EnhancedDualHeadTransformerGNN(
                node_features=node_features,
                hidden_dim=config.get('hidden_dim', 256),
                num_gnn_layers=config.get('num_gnn_layers', 8),
                num_transformer_layers=config.get('num_transformer_layers', 6),
                num_heads=config.get('num_heads', 8),
                dropout=config.get('dropout', 0.1)
            )
            for config in model_configs
        ])
        
        # Learnable weights for ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(len(model_configs)))
        
    def forward(self, data):
        # Get predictions from all models
        all_outputs = [model(data) for model in self.models]
        
        # Normalize ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average of values
        values = torch.stack([output['value'] for output in all_outputs], dim=0)
        ensemble_value = torch.sum(values * weights.unsqueeze(1), dim=0)
        
        # Weighted average of policies
        policies = torch.stack([output['policy'] for output in all_outputs], dim=0)
        ensemble_policy = torch.sum(policies * weights.unsqueeze(1).unsqueeze(2), dim=0)
        
        # Weighted average of uncertainties
        uncertainties = torch.stack([output['uncertainty'] for output in all_outputs], dim=0)
        ensemble_uncertainty = torch.sum(uncertainties * weights.unsqueeze(1), dim=0)
        
        return {
            'value': ensemble_value,
            'policy': ensemble_policy,
            'uncertainty': ensemble_uncertainty
        }

# Knowledge distillation model
class DistillationModel(nn.Module):
    """Knowledge distillation from multiple teacher models"""
    def __init__(self, teacher_models, node_features=128, hidden_dim=192):
        super(DistillationModel, self).__init__()
        
        # Student model (smaller than teachers)
        self.student = EnhancedDualHeadTransformerGNN(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_gnn_layers=6,
            num_transformer_layers=4,
            num_heads=6,
            dropout=0.1
        )
        
        # Freeze teacher models
        self.teacher_models = teacher_models
        for teacher in self.teacher_models:
            for param in teacher.parameters():
                param.requires_grad = False
        
    def forward(self, data):
        # Get student predictions
        student_output = self.student(data)
        
        # Get teacher predictions (if in training mode)
        if self.training:
            teacher_outputs = [teacher(data) for teacher in self.teacher_models]
            
            # Average teacher predictions
            teacher_values = torch.stack([output['value'] for output in teacher_outputs], dim=0).mean(0)
            teacher_policies = torch.stack([output['policy'] for output in teacher_outputs], dim=0).mean(0)
            
            return {
                'student_output': student_output,
                'teacher_value': teacher_values,
                'teacher_policy': teacher_policies
            }
        
        return student_output 