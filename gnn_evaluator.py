"""GNN position evaluator: GCN backbone + transformer policy and value heads.

* Policy: logits over 64x64 from-to pairs (index = from * 64 + to, squares
  a1 = 0 ... h8 = 63). Promotions share the index of their from/to pair.
* Value: tanh output in [-1, 1] from the side to move's point of view
  (+1 = side to move wins, 0 = draw, -1 = side to move loses).
* Uncertainty: standard deviation of the value over Monte-Carlo dropout samples.

Checkpoints are dicts {"format": ..., "config": {...}, "state_dict": ...} so
the model can be rebuilt with the right dimensions (see save_checkpoint /
load_checkpoint).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import to_dense_batch
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch_geometric is required.") from e

from chess_graph import FEATURE_DIM, ChessGNN, ChessGraph

POLICY_DIM = 64 * 64
CHECKPOINT_FORMAT = "nags-gnn-v2"


def move_index(uci: str) -> int:
    """Policy index of a UCI move string (e.g. 'e2e4', 'e7e8q')."""
    if len(uci) < 4:
        raise ValueError(f"invalid UCI move {uci!r}")
    f0, r0, f1, r1 = ord(uci[0]) - 97, ord(uci[1]) - 49, ord(uci[2]) - 97, ord(uci[3]) - 49
    if not all(0 <= v < 8 for v in (f0, r0, f1, r1)):
        raise ValueError(f"invalid UCI move {uci!r}")
    return (r0 * 8 + f0) * 64 + (r1 * 8 + f1)


def _nhead_for(hidden_dim: int, preferred: int = 8) -> int:
    for n in (preferred, 4, 2, 1):
        if hidden_dim % n == 0:
            return n
    return 1


class PolicyTransformerHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=_nhead_for(hidden_dim),
                                           dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.proj_from = nn.Linear(hidden_dim, hidden_dim)
        self.proj_to = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # dense: [B, N, H], mask: [B, N] (True = real node). Square nodes come first in every graph.
        x = self.enc(dense, src_key_padding_mask=~mask)
        squares = x[:, :64, :]
        logits = torch.bmm(self.proj_from(squares), self.proj_to(squares).transpose(1, 2)) * self.scale
        return logits.reshape(x.size(0), POLICY_DIM)


class ValueTransformerHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=_nhead_for(hidden_dim),
                                           dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.enc(dense, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return torch.tanh(self.mlp(self.dropout(pooled)).squeeze(-1))  # [B]


class GNNEvaluator(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 128, gnn_layers: int = 6,
                 policy_layers: int = 4, value_layers: int = 2, device: torch.device | None = None):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden_dim=hidden_dim, gnn_layers=gnn_layers,
                           policy_layers=policy_layers, value_layers=value_layers)
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.gnn = ChessGNN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=gnn_layers)
        self.policy = PolicyTransformerHead(hidden_dim=hidden_dim, num_layers=policy_layers)
        self.value = ValueTransformerHead(hidden_dim=hidden_dim, num_layers=value_layers)
        self.to(self.device)

    def _dense(self, graph: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        _, node_emb = self.gnn(graph)
        batch = getattr(graph, 'batch', None)
        if batch is None:
            batch = node_emb.new_zeros(node_emb.size(0), dtype=torch.long)
        return to_dense_batch(node_emb, batch)

    def forward(self, graph: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy logits [B, 4096], values [B]) for a Data or Batch."""
        graph = graph.to(self.device)
        dense, mask = self._dense(graph)
        return self.policy(dense, mask), self.value(dense, mask)

    @torch.no_grad()
    def evaluate(self, graph: Data, mc_samples: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference with Monte-Carlo dropout on the value head.

        Returns (policy probabilities [B, 4096], value mean [B], value std [B])
        on the CPU. Restores the module's previous train/eval mode.
        """
        was_training = self.training
        try:
            self.eval()
            graph = graph.to(self.device)
            dense, mask = self._dense(graph)
            probs = F.softmax(self.policy(dense, mask), dim=-1)
            samples = max(1, mc_samples)
            if samples > 1:
                self.value.train()  # dropout on, gradients still disabled
            values = torch.stack([self.value(dense, mask) for _ in range(samples)])  # [S, B]
            mean = values.mean(dim=0)
            std = values.std(dim=0, unbiased=False) if samples > 1 else torch.zeros_like(mean)
            return probs.cpu(), mean.cpu(), std.cpu()
        finally:
            self.train(was_training)


def legal_move_priors(policy_row: torch.Tensor, moves: Sequence[str]) -> List[float]:
    """Renormalises a 4096-entry probability vector over the given legal moves."""
    if not moves:
        return []
    idx = torch.tensor([move_index(m) for m in moves], dtype=torch.long)
    p = policy_row[idx].clamp(min=0)
    total = float(p.sum())
    if total <= 0:
        return [1.0 / len(moves)] * len(moves)
    return (p / total).tolist()


def save_checkpoint(model: GNNEvaluator, path: str, extra: Optional[Dict] = None) -> None:
    payload = {"format": CHECKPOINT_FORMAT, "config": dict(model.config),
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device | None = None) -> GNNEvaluator:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint (re-train with training_pipeline.py)")
    model = GNNEvaluator(**payload["config"], device=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def warm_start_and_run_example(fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    """Runs an untrained evaluator on one position (shape/smoke check)."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    builder = ChessGraph(device=device)
    data = builder.fen_to_graph(fen)
    evaluator = GNNEvaluator(in_dim=data.x.size(1), hidden_dim=64, gnn_layers=6, policy_layers=4,
                             value_layers=2, device=device)
    probs, value, uncertainty = evaluator.evaluate(data, mc_samples=5)
    topk = torch.topk(probs[0], 5)
    return {
        'top_policy_indices': topk.indices.tolist(),
        'top_policy_probs': [float(p) for p in topk.values.tolist()],
        'value': float(value[0]),
        'uncertainty': float(uncertainty[0]),
    }


if __name__ == '__main__':
    print(warm_start_and_run_example())
