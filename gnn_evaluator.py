import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch_geometric is required.") from e

from chess_graph import ChessGNN, ChessGraph


class PolicyTransformerHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj_from = nn.Linear(hidden_dim, hidden_dim)
        self.proj_to = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        # node_emb: [N, H]; assume first 64 are square nodes
        N, H = node_emb.shape
        x = node_emb.unsqueeze(0)  # [1, N, H]
        x = self.enc(x)  # [1, N, H]
        x = x.squeeze(0)
        squares = x[:64, :]  # [64, H]
        from_proj = self.proj_from(squares)  # [64, H]
        to_proj = self.proj_to(squares)      # [64, H]
        logits = from_proj @ to_proj.t()     # [64, 64]
        return logits.view(-1)               # [4096]


class ValueTransformerHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, nhead: int = 8, dropout: float = 0.2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_emb: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        x = node_emb.unsqueeze(0)  # [1, N, H]
        # To apply MC dropout, we keep dropout active by calling self.train() at call-site
        x = self.enc(x)
        x = x.mean(dim=1)  # global mean over nodes: [1, H]
        x = self.dropout(x)
        v = self.mlp(x)  # [1, 1]
        return torch.tanh(v.squeeze(-1))  # [-1,1]


class GNNEvaluator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, gnn_layers: int = 6, policy_layers: int = 4, value_layers: int = 2, device: torch.device | None = None):
        super().__init__()
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.gnn = ChessGNN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=gnn_layers).to(self.device)
        self.policy = PolicyTransformerHead(hidden_dim=hidden_dim, num_layers=policy_layers).to(self.device)
        self.value = ValueTransformerHead(hidden_dim=hidden_dim, num_layers=value_layers).to(self.device)
        self.to(self.device)

    @torch.no_grad()
    def evaluate(self, graph: Data, mc_samples: int = 5) -> Tuple[torch.Tensor, float, float]:
        # Ensure tensors are on correct device
        graph = graph.to(self.device)
        self.eval()
        # Backbone embeddings
        global_emb, node_emb = self.gnn(graph)
        # Policy logits and probabilities (over 64x64 moves)
        policy_logits = self.policy(node_emb)
        policy_probs = F.softmax(policy_logits, dim=0)

        # MC Dropout for value: enable dropout only in value head
        vals = []
        # Put value head in train mode to activate dropout; keep grads disabled
        self.value.train()
        for _ in range(max(1, mc_samples)):
            v = self.value(node_emb, mc_dropout=True)
            vals.append(v.item())
        self.value.eval()
        vals_t = torch.tensor(vals, device=self.device)
        value_est = vals_t.mean().item()
        uncertainty = vals_t.std(unbiased=False).item()
        return policy_probs.detach().cpu(), float(value_est), float(uncertainty)


def warm_start_and_run_example(fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    builder = ChessGraph(device=device)
    data = builder.fen_to_graph(fen)
    in_dim = data.x.size(1)
    evaluator = GNNEvaluator(in_dim=in_dim, hidden_dim=64, gnn_layers=6, policy_layers=4, value_layers=2, device=device)
    policy_probs, value_est, uncertainty = evaluator.evaluate(data, mc_samples=5)
    # Return compact summary
    topk = torch.topk(policy_probs, 5)
    return {
        'top_policy_indices': topk.indices.tolist(),
        'top_policy_probs': [float(p) for p in topk.values.tolist()],
        'value': value_est,
        'uncertainty': uncertainty,
    }


if __name__ == '__main__':
    out = warm_start_and_run_example()
    print(out)


