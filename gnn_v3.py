"""v3 GNN evaluator ('nags-gnn-v3'): relational message passing over the
typed graph of relational_graph.py, a promotion-aware policy
(policy_encoding.py) and a win/draw/loss value head.

This module is additive. The legacy evaluator (gnn_evaluator.GNNEvaluator,
'nags-gnn-v2', 4096-entry policy, chess_graph.ChessGraph) is unchanged, and
load_any_checkpoint() serves either format with its own graph builder and
move encoding, so one format's outputs are never read with the other's
encoding.

Model
  * Trunk: input projection, then `layers` residual blocks
        h = LayerNorm(h + Dropout(ReLU(RelationalConv(h))))
        h = LayerNorm(h + Dropout(FFN(h)))            (if ffn_mult > 0)
    RelationalConv is an R-GCN layer (Schlichtkrull et al., 2018) with basis
    decomposition and a mean per relation:
        out_i = W_0 h_i + sum_r W_r mean_{j -[r]-> i} h_j,   W_r = sum_b a_rb V_b
    computed with one gather and `num_bases` index_add calls (no per-relation
    Python loop, no torch-scatter needed).
  * Policy (POLICY_V3_DIM = 4228 logits): bilinear from-square x to-square
    scores of the square embeddings (plus the global embedding) for the
    4096 from/to entries; each underpromotion logit is the logit of its
    from/to pair (= the queen promotion) plus a learned offset per piece
    computed from the two square embeddings.
  * WDL: 3 logits (win, draw, loss for the side to move) from the global
    node and the mean of all nodes; value = P(win) - P(loss) in [-1, 1].
  * Optional auxiliary search-score head: tanh output trained towards the
    search's expected score (see training_pipeline.py, step 'distill').

evaluate() returns (policy probabilities [B, 4228], value [B], uncertainty
[B]) with the meaning of the legacy evaluate(): the policy is a softmax over
all 4228 entries, the value is from the side to move's point of view, and
the uncertainty is the standard deviation of the value over `mc_samples`
Monte-Carlo dropout samples of the WDL head (0 when mc_samples <= 1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import global_mean_pool
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch_geometric is required.") from e

from chess_graph import ChessGraph
from gnn_evaluator import CHECKPOINT_FORMAT as CHECKPOINT_FORMAT_V2
from gnn_evaluator import POLICY_DIM as POLICY_DIM_V2
from gnn_evaluator import GNNEvaluator, legal_move_priors, load_checkpoint
from policy_encoding import (FROM_TO_DIM, NUM_PROMOTION_PAIRS, POLICY_V3_DIM, PROMOTION_FROM, PROMOTION_TO,
                             move_index_v3)
from relational_graph import (FEATURE_DIM_V3, GRAPH_VERSION, KIND_GLOBAL, KIND_SQUARE, NUM_RELATIONS,
                              RelationalChessGraph)

CHECKPOINT_FORMAT_V3 = "nags-gnn-v3"
POLICY_ENCODING_V3 = "v3"


class RelationalConv(nn.Module):
    """R-GCN layer with basis decomposition; messages are averaged per relation."""

    def __init__(self, dim: int, num_relations: int, num_bases: int):
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.root = nn.Linear(dim, dim)
        self.basis = nn.Linear(num_bases * dim, dim, bias=False)
        self.comp = nn.Parameter(torch.empty(num_relations, num_bases))
        nn.init.xavier_uniform_(self.comp)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_type: torch.Tensor,
                inv_deg: torch.Tensor) -> torch.Tensor:
        hs = h.index_select(0, src)
        coef = self.comp.index_select(0, edge_type) * inv_deg.unsqueeze(1)  # [E, B]
        aggs = [torch.zeros_like(h).index_add_(0, dst, hs * coef[:, b:b + 1]) for b in range(self.num_bases)]
        return self.root(h) + self.basis(torch.cat(aggs, dim=1))


def relation_inv_degree(dst: torch.Tensor, edge_type: torch.Tensor, num_nodes: int, num_relations: int) -> torch.Tensor:
    """1 / (number of edges of the same relation into the same node), per edge."""
    key = dst * num_relations + edge_type
    counts = torch.bincount(key, minlength=num_nodes * num_relations)
    return 1.0 / counts.index_select(0, key).to(torch.float)


class RelationalGNNEvaluator(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM_V3, hidden_dim: int = 128, layers: int = 6, num_bases: int = 4,
                 ffn_mult: int = 2, dropout: float = 0.0, value_dropout: float = 0.2, aux_score: bool = True,
                 num_relations: int = NUM_RELATIONS, policy_encoding: str = POLICY_ENCODING_V3,
                 graph: str = GRAPH_VERSION, policy_dim: int = POLICY_V3_DIM, device: torch.device | None = None):
        super().__init__()
        if policy_encoding != POLICY_ENCODING_V3 or policy_dim != POLICY_V3_DIM:
            raise ValueError(f"unsupported policy encoding {policy_encoding!r} ({policy_dim} entries); "
                             f"this code knows {POLICY_ENCODING_V3!r} ({POLICY_V3_DIM})")
        if graph != GRAPH_VERSION or in_dim != FEATURE_DIM_V3 or num_relations != NUM_RELATIONS:
            raise ValueError(f"unsupported graph {graph!r} ({in_dim} features, {num_relations} relations); "
                             f"this code builds {GRAPH_VERSION!r} ({FEATURE_DIM_V3}, {NUM_RELATIONS})")
        self.config = dict(in_dim=in_dim, hidden_dim=hidden_dim, layers=layers, num_bases=num_bases,
                           ffn_mult=ffn_mult, dropout=dropout, value_dropout=value_dropout, aux_score=aux_score,
                           num_relations=num_relations, policy_encoding=policy_encoding, graph=graph,
                           policy_dim=policy_dim)
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        H = hidden_dim
        self.proj_in = nn.Linear(in_dim, H)
        self.convs = nn.ModuleList([RelationalConv(H, num_relations, num_bases) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(layers)])
        self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(H, H * ffn_mult), nn.ReLU(), nn.Linear(H * ffn_mult, H))
                                   for _ in range(layers)]) if ffn_mult > 0 else None
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(layers)]) if ffn_mult > 0 else None
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # Policy head.
        self.pol_from = nn.Linear(H, H)
        self.pol_to = nn.Linear(H, H)
        self.scale = H ** -0.5
        self.under = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, 3))
        nn.init.constant_(self.under[-1].bias, -2.0)  # underpromotions start e^-2 times as likely as the queen
        self.register_buffer("promo_from", PROMOTION_FROM.clone(), persistent=False)
        self.register_buffer("promo_to", PROMOTION_TO.clone(), persistent=False)

        # Value heads.
        self.wdl_head = nn.Sequential(nn.Dropout(value_dropout), nn.Linear(2 * H, H), nn.ReLU(),
                                      nn.Dropout(value_dropout), nn.Linear(H, 3))
        self.score_head = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, 1)) if aux_score else None
        self.to(self.device)

    # -- trunk ----------------------------------------------------------------
    def _embed(self, graph: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (square embeddings [B, 64, H], global embeddings [B, H], pooled [B, 2H])."""
        x, edge_index, edge_type, kind = graph.x, graph.edge_index, graph.edge_type, graph.node_kind
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        inv_deg = relation_inv_degree(dst, edge_type, n, self.config["num_relations"])
        h = self.proj_in(x)
        for i, conv in enumerate(self.convs):
            h = self.norms[i](h + self.drop(self.act(conv(h, src, dst, edge_type, inv_deg))))
            if self.ffns is not None:
                h = self.ffn_norms[i](h + self.drop(self.ffns[i](h)))
        batch = getattr(graph, 'batch', None)
        if batch is None:
            batch = kind.new_zeros(n)
        glob = h[kind == KIND_GLOBAL]
        num_graphs = glob.size(0)
        squares = h[kind == KIND_SQUARE].view(num_graphs, 64, -1)
        pooled = torch.cat([glob, global_mean_pool(h, batch, size=num_graphs)], dim=1)
        return squares, glob, pooled

    def _policy(self, squares: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        s = squares + glob.unsqueeze(1)
        f, t = self.pol_from(s), self.pol_to(s)
        b = s.size(0)
        from_to = torch.bmm(f, t.transpose(1, 2)) * self.scale                  # [B, 64, 64]
        base = from_to[:, self.promo_from, self.promo_to]                      # [B, 44]
        offset = self.under(torch.cat([f[:, self.promo_from], t[:, self.promo_to]], dim=-1))  # [B, 44, 3]
        under = (base.unsqueeze(-1) + offset).reshape(b, NUM_PROMOTION_PAIRS * 3)
        return torch.cat([from_to.reshape(b, FROM_TO_DIM), under], dim=1)

    def forward(self, graph: Data) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (policy logits [B, 4228], WDL logits [B, 3], auxiliary score in [-1, 1] [B] or None)."""
        graph = graph.to(self.device)
        squares, glob, pooled = self._embed(graph)
        score = torch.tanh(self.score_head(pooled).squeeze(-1)) if self.score_head is not None else None
        return self._policy(squares, glob), self.wdl_head(pooled), score

    @staticmethod
    def wdl_value(wdl_logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(wdl_logits, dim=-1)
        return p[..., 0] - p[..., 2]

    @torch.no_grad()
    def evaluate(self, graph: Data, mc_samples: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(policy probabilities [B, 4228], value mean [B], value std over MC-dropout samples [B]) on the CPU.

        Restores the module's previous train/eval mode.
        """
        was_training = self.training
        try:
            self.eval()
            graph = graph.to(self.device)
            squares, glob, pooled = self._embed(graph)
            probs = F.softmax(self._policy(squares, glob), dim=-1)
            samples = max(1, mc_samples)
            if samples > 1:
                self.wdl_head.train()  # dropout on, gradients still disabled
            values = torch.stack([self.wdl_value(self.wdl_head(pooled)) for _ in range(samples)])
            mean = values.mean(dim=0)
            std = values.std(dim=0, unbiased=False) if samples > 1 else torch.zeros_like(mean)
            return probs.cpu(), mean.cpu(), std.cpu()
        finally:
            self.train(was_training)

    @torch.no_grad()
    def evaluate_wdl(self, graph: Data) -> torch.Tensor:
        """Win/draw/loss probabilities [B, 3] for the side to move (dropout off)."""
        was_training = self.training
        try:
            self.eval()
            _, _, pooled = self._embed(graph.to(self.device))
            return F.softmax(self.wdl_head(pooled), dim=-1).cpu()
        finally:
            self.train(was_training)


def legal_move_priors_v3(policy_row: torch.Tensor, moves: Sequence[str]) -> List[float]:
    """Renormalises a 4228-entry v3 probability vector over the given legal moves (same order)."""
    if not moves:
        return []
    idx = torch.tensor([move_index_v3(m) for m in moves], dtype=torch.long)
    p = policy_row[idx].clamp(min=0)
    total = float(p.sum())
    if total <= 0:
        return [1.0 / len(moves)] * len(moves)
    return (p / total).tolist()


def save_checkpoint_v3(model: RelationalGNNEvaluator, path: str, extra: Optional[Dict] = None) -> None:
    """Writes {'format': 'nags-gnn-v3', 'config': {...}, 'state_dict': ...} (plus 'extra').

    Keep `extra` to plain Python data (dict/list/str/number/bool/None): v3
    checkpoints are read with torch.load(weights_only=True).
    """
    payload: Dict[str, Any] = {"format": CHECKPOINT_FORMAT_V3, "config": dict(model.config),
                               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def _read_plain(path: str) -> Any:
    return torch.load(path, map_location='cpu', weights_only=True)


def load_checkpoint_v3(path: str, device: torch.device | None = None) -> RelationalGNNEvaluator:
    """Loads a 'nags-gnn-v3' checkpoint; ValueError for any other format or an unknown encoding/graph."""
    try:
        payload = _read_plain(path)
    except Exception as e:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT_V3} checkpoint ({type(e).__name__})") from e
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT_V3:
        found = payload.get("format") if isinstance(payload, dict) else None
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT_V3} checkpoint (format: {found!r})")
    try:
        model = RelationalGNNEvaluator(**payload["config"], device=device)
    except TypeError as e:  # a config key this code does not know
        raise ValueError(f"{path}: unsupported {CHECKPOINT_FORMAT_V3} config ({e})") from e
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def checkpoint_format(path: str) -> Optional[str]:
    """The 'format' field of a checkpoint, or None if it has none or is not plain data."""
    try:
        payload = _read_plain(path)
    except Exception:
        return None
    return payload.get("format") if isinstance(payload, dict) else None


# ----------------------------------------------------------------------------
# Adapters: a loaded network together with the graph builder and the move
# encoding of its format.

class GNNAdapter:
    format: str = ""
    policy_dim: int = 0

    def __init__(self, model: nn.Module, builder: Any):
        self.model = model
        self.builder = builder

    def build_batch(self, positions: Sequence[Any]) -> Batch:
        """Graphs for FENs (ValueError on a malformed FEN)."""
        return self.builder.fens_to_batch(positions)

    def evaluate(self, batch: Data, mc_samples: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.evaluate(batch, mc_samples=mc_samples)

    def move_priors(self, policy_row: torch.Tensor, moves: Sequence[str]) -> List[float]:
        raise NotImplementedError

    def evaluate_fens(self, fens: Sequence[str], moves: Optional[Sequence[Sequence[str]]] = None,
                      mc_samples: int = 1) -> List[Dict[str, Any]]:
        """Convenience: [{'value', 'uncertainty', 'move_priors' (if moves given)}] per FEN."""
        probs, values, unc = self.evaluate(self.build_batch(list(fens)), mc_samples=mc_samples)
        out = []
        for i in range(len(fens)):
            item: Dict[str, Any] = {"value": float(values[i]), "uncertainty": float(unc[i])}
            if moves is not None:
                item["move_priors"] = self.move_priors(probs[i], list(moves[i]))
            out.append(item)
        return out


class GNNAdapterV2(GNNAdapter):
    """Legacy 'nags-gnn-v2': GNNEvaluator + ChessGraph + 4096-entry from/to priors (unchanged behaviour)."""
    format = CHECKPOINT_FORMAT_V2
    policy_dim = POLICY_DIM_V2

    def move_priors(self, policy_row: torch.Tensor, moves: Sequence[str]) -> List[float]:
        return legal_move_priors(policy_row, moves)


class GNNAdapterV3(GNNAdapter):
    """'nags-gnn-v3': RelationalGNNEvaluator + RelationalChessGraph + 4228-entry promotion-aware priors."""
    format = CHECKPOINT_FORMAT_V3
    policy_dim = POLICY_V3_DIM

    def move_priors(self, policy_row: torch.Tensor, moves: Sequence[str]) -> List[float]:
        return legal_move_priors_v3(policy_row, moves)


def as_adapter(model: Union[GNNAdapter, nn.Module], builder: Any = None) -> GNNAdapter:
    """Wraps a model (and optionally its builder) in the adapter of its format."""
    if isinstance(model, GNNAdapter):
        return model
    if isinstance(model, RelationalGNNEvaluator):
        return GNNAdapterV3(model, builder if builder is not None else RelationalChessGraph(device=model.device))
    if isinstance(model, GNNEvaluator):
        return GNNAdapterV2(model, builder if builder is not None else ChessGraph(device=model.device))
    raise TypeError(f"no adapter for {type(model).__name__}")


def load_any_checkpoint(path: str, device: torch.device | None = None) -> GNNAdapter:
    """Loads a 'nags-gnn-v2' or 'nags-gnn-v3' checkpoint into the adapter of its format.

    v2 files go through gnn_evaluator.load_checkpoint exactly as before, so a
    v2 network gives the same outputs as when it was trained.
    """
    fmt = checkpoint_format(path)
    if fmt == CHECKPOINT_FORMAT_V3:
        model = load_checkpoint_v3(path, device=device)
        return GNNAdapterV3(model, RelationalChessGraph(device=model.device))
    if fmt not in (None, CHECKPOINT_FORMAT_V2):
        raise ValueError(f"{path}: unknown checkpoint format {fmt!r} "
                         f"(expected {CHECKPOINT_FORMAT_V2!r} or {CHECKPOINT_FORMAT_V3!r})")
    try:
        legacy = load_checkpoint(path, device=device)
    except ValueError as e:
        raise ValueError(f"{path} is neither a {CHECKPOINT_FORMAT_V2} nor a {CHECKPOINT_FORMAT_V3} checkpoint") from e
    return GNNAdapterV2(legacy, ChessGraph(device=legacy.device))
