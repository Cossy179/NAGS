"""Search-policy distillation for the v3 GNN (training_pipeline.py steps
'label' and 'distill').

Labelling: the alpha-beta engine analyses each position with MultiPV k at a
fixed node budget. A label row (one JSON object per line) holds
    fen               the position
    legal             every legal move (UCI)
    scored            the top-k moves, best first, each {"move": uci} plus
                      exactly one of {"cp": centipawns} or {"mate": n}, from
                      the side to move's point of view (mate n > 0: the side
                      to move mates in n moves; n < 0: it is mated in -n).
                      Each MultiPV line is a full-window search, so these
                      scores are exact for the search that produced them
                      ("bound": "lower"/"upper" marks a line the engine
                      reported as a bound).
    multipv           k as requested
    rest_upper_bound  true when some legal moves are not in "scored": those
                      are only known to score at most the k-th (last) score
    depth, seldepth, nodes   of the search (depth: the shallowest line)
    result            the source game's result for the side to move
                      (1 / 0 / -1) when the source row has one, else null
    source, game_id, ply     where the position came from, when known

Targets (all from the side to move's point of view):
  * Scores become centipawns with effective_cp(): cp clipped to
    [-cp_clip, cp_clip]; a mate in n becomes +(mate_cp - mate_step * (n - 1)),
    being mated in n becomes -(mate_cp - mate_step * (n - 1)), each kept
    beyond cp_clip, so mates are large but finite, outrank every cp score,
    and faster mates (slower losses) are preferred.
  * Policy: softmax(effective_cp / temperature_cp) over the scored moves,
    plus one pooled "rest" bucket for the legal moves that were not scored.
    Each of those is only known to score at most the k-th (lowest scored)
    score s_k, so together they hold at most n_rest * exp(s_k / T) of the
    softmax weight; giving them none (the earlier target) made the target
    too sharp whenever s_k is within a few T of the best score. The rest
    bucket gets rest_mass_scale (default 0.02) times that upper bound: an
    estimate between the lower bound 0 and the upper bound (most unscored
    moves score far below s_k). Half the bound (0.5, the first default)
    gave the bucket more target mass on average than the engine's best
    move and the worst held-out policy of 0.5 / 0.1 / 0.02 / 0, while 0
    lost value accuracy (tools/gnn_compare.py, docs/ARCHITECTURE.md). The network is trained on
    the summed probability of the unscored moves for that bucket, not on
    any of them individually (they are censored, not known). Illegal moves
    are masked out of the network's log-softmax. Rows with a line the
    engine reported as a bound are not used.
  * WDL from the search score s (the best line's effective cp) with a
    logistic model:
        P(win)  = sigmoid((s - wdl_a) / wdl_b)
        P(loss) = sigmoid((-s - wdl_a) / wdl_b)
        P(draw) = 1 - P(win) - P(loss)          (>= 0 for wdl_a >= 0)
    The defaults wdl_a = 400, wdl_b = 250 make the expected score
    P(win) + P(draw) / 2 close to sigmoid(s / 400), the scale the NNUE
    trainer uses (tools/nnue/train.py, SCALE = 400), and give a 66% draw
    probability at s = 0. With result_weight w > 0 and a known game
    result, target = (1 - w) * WDL(s) + w * one_hot(result).
  * Value target for the MSE metric: P(win) - P(loss) of that target; the
    auxiliary score head is trained towards the search-only P(win) - P(loss).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine
import torch
from torch.utils.data import Dataset

from policy_encoding import POLICY_V3_DIM, legal_mask_v3, move_index_v3
from relational_graph import build_graph


# ----------------------------------------------------------------------------
# Scores and targets

def score_record(score: chess.engine.Score) -> Dict[str, int]:
    """{'mate': n} or {'cp': x} for a score from the side to move's point of view."""
    if score.is_mate():
        return {"mate": int(score.mate())}
    return {"cp": int(score.score())}


def effective_cp(rec: Dict[str, Any], cp_clip: float = 2000.0, mate_cp: float = 3000.0,
                 mate_step: float = 10.0) -> float:
    """Centipawns of a {'cp': x} / {'mate': n} record, with mates large but finite."""
    mate = rec.get("mate")
    if mate is not None:
        n = int(mate)
        magnitude = max(mate_cp - mate_step * max(abs(n) - 1, 0), cp_clip + 1.0)
        return magnitude if n > 0 else -magnitude
    return float(min(max(float(rec["cp"]), -cp_clip), cp_clip))


def wdl_from_cp(cp: float, a: float = 400.0, b: float = 250.0) -> Tuple[float, float, float]:
    """(P(win), P(draw), P(loss)) for the side to move from a centipawn score (logistic model)."""
    if a < 0 or b <= 0:
        raise ValueError("wdl_a must be >= 0 and wdl_b > 0")

    def sig(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    w = sig((cp - a) / b)
    l = sig((-cp - a) / b)
    return w, max(0.0, 1.0 - w - l), l


def policy_target(scored: Sequence[Dict[str, Any]], temperature: float, **cp_kwargs) -> List[Tuple[str, float]]:
    """[(uci, probability)] = softmax(effective_cp / temperature) over the scored moves."""
    if not scored:
        return []
    if temperature <= 0:
        raise ValueError("temperature_cp must be > 0")
    cps = [effective_cp(r, **cp_kwargs) for r in scored]
    top = max(cps)
    weights = [math.exp((c - top) / temperature) for c in cps]
    total = sum(weights)
    return [(r["move"], w / total) for r, w in zip(scored, weights)]


def _cp_kwargs(params: Dict[str, Any]) -> Dict[str, float]:
    return {"cp_clip": float(params["cp_clip"]), "mate_cp": float(params["mate_cp"]),
            "mate_step": float(params["mate_step"])}


def policy_with_rest(scored: Sequence[Dict[str, Any]], n_rest: int, temperature: float, rest_scale: float,
                     **cp_kwargs) -> Tuple[List[Tuple[str, float]], float]:
    """([(uci, probability)] over the scored moves, probability of the pooled
    unscored moves): see the module docstring."""
    if not scored:
        return [], 0.0
    if temperature <= 0:
        raise ValueError("temperature_cp must be > 0")
    cps = [effective_cp(r, **cp_kwargs) for r in scored]
    top = max(cps)
    weights = [math.exp((c - top) / temperature) for c in cps]
    rest = n_rest * rest_scale * math.exp((min(cps) - top) / temperature) if n_rest > 0 and rest_scale > 0 else 0.0
    total = sum(weights) + rest
    return [(r["move"], w / total) for r, w in zip(scored, weights)], rest / total


def row_targets(row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Policy / WDL / value targets of one label row (see the module docstring)."""
    kw = _cp_kwargs(params)
    scored = row["scored"]
    legal = row.get("legal")
    if legal is not None:
        n_legal = len(legal)
    else:  # label rows carry "legal"; without it or a FEN no unscored move is known
        n_legal = chess.Board(row["fen"]).legal_moves.count() if "fen" in row else len(scored)
    policy, rest = policy_with_rest(scored, n_legal - len(scored), float(params["temperature_cp"]),
                                    float(params.get("rest_mass_scale", 0.02)), **kw)
    best = max(scored, key=lambda r: effective_cp(r, **kw))
    search_cp = effective_cp(best, **kw)
    a, b = float(params["wdl_a"]), float(params["wdl_b"])
    search_wdl = wdl_from_cp(search_cp, a, b)
    wdl = list(search_wdl)
    weight = float(params.get("result_weight", 0.0))
    result = row.get("result")
    if weight > 0 and result in (1, 0, -1, 1.0, 0.0, -1.0):
        one_hot = {1: (1.0, 0.0, 0.0), 0: (0.0, 1.0, 0.0), -1: (0.0, 0.0, 1.0)}[int(result)]
        wdl = [(1 - weight) * s + weight * o for s, o in zip(search_wdl, one_hot)]
    return {"policy": policy, "rest": rest, "best": best["move"], "search_cp": search_cp, "wdl": wdl,
            "value": wdl[0] - wdl[2], "search_value": search_wdl[0] - search_wdl[2]}


def fast_graph(board: chess.Board):
    """build_graph(board) with its legal mask (data.legal [1, 4228]), from relational_fast: identical
    tensors (tests/python/test_distill.py); an item takes 0.6 ms instead of 1.0 here."""
    import numpy as np
    from torch_geometric.data import Data

    from relational_fast import build_arrays

    g = build_arrays(board)
    data = Data(x=torch.from_numpy(g.x), edge_index=torch.from_numpy(np.stack([g.src, g.dst])),
                edge_type=torch.from_numpy(g.edge_type), node_kind=torch.from_numpy(g.node_kind),
                num_nodes=int(g.x.shape[0]))
    legal = torch.zeros(1, POLICY_V3_DIM, dtype=torch.bool)
    legal[0, torch.from_numpy(g.legal)] = True
    data.legal = legal
    return data


class DistillDataset(Dataset):
    """Label rows -> relational graphs with distillation targets.

    Per graph: legal [1, 4228] bool, target_policy [1, 4228], target_wdl [1, 3],
    target_value [1], search_value [1], best [1] (v3 index of the best move).
    """

    def __init__(self, rows: List[Dict[str, Any]], params: Dict[str, Any]):
        self.rows = rows
        self.params = params

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        board = chess.Board(row["fen"])
        t = row_targets(row, self.params)
        if self.params.get("fast_graphs"):
            data = fast_graph(board)  # the same graph and mask, built faster
        else:
            data = build_graph(board)
            data.legal = legal_mask_v3(board).unsqueeze(0)
        target = torch.zeros(1, POLICY_V3_DIM)
        rest = data.legal.clone()  # the legal moves that were not scored
        for uci, p in t["policy"]:
            target[0, move_index_v3(uci)] += p
        for r in row["scored"]:
            rest[0, move_index_v3(r["move"])] = False
        data.target_policy = target
        data.rest = rest
        data.target_rest = torch.tensor([t["rest"]], dtype=torch.float)
        data.target_wdl = torch.tensor([t["wdl"]], dtype=torch.float)
        data.target_value = torch.tensor([t["value"]], dtype=torch.float)
        data.search_value = torch.tensor([t["search_value"]], dtype=torch.float)
        data.best = torch.tensor([move_index_v3(t["best"])], dtype=torch.long)
        return data


def valid_row(row: Dict[str, Any]) -> bool:
    """A label row whose scored moves are legal in its position."""
    try:
        board = chess.Board(row["fen"])
        scored = row.get("scored") or []
        if not scored:
            return False
        legal = set(board.legal_moves)
        for r in scored:
            if chess.Move.from_uci(r["move"]) not in legal or ("cp" not in r and "mate" not in r):
                return False
            if r.get("bound"):  # only a bound, not an exact score
                return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """log_softmax over the legal entries; illegal entries get a large negative (finite) value."""
    return torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)


def distill_losses(policy_logits: torch.Tensor, wdl_logits: torch.Tensor, score: Optional[torch.Tensor],
                   batch: Any) -> Dict[str, torch.Tensor]:
    """Per-position losses / statistics of a batch (not reduced)."""
    logp = masked_log_softmax(policy_logits, batch.legal)
    policy_nll = -(batch.target_policy * logp).sum(dim=1)
    # The unscored legal moves as one bucket: log of their summed probability.
    rest = getattr(batch, "rest", None)
    if rest is not None:
        has_rest = rest.any(dim=1)
        # (a finite fill: a row of -inf would give NaN gradients even where unused)
        log_rest = torch.logsumexp(logp.masked_fill(~rest, -1e9), dim=1)
        policy_nll = policy_nll - torch.where(has_rest, batch.target_rest * log_rest, torch.zeros_like(policy_nll))
    wdl_logp = torch.log_softmax(wdl_logits, dim=1)
    wdl_nll = -(batch.target_wdl * wdl_logp).sum(dim=1)
    p = wdl_logp.exp()
    value = p[:, 0] - p[:, 2]
    out = {"policy_nll": policy_nll, "wdl_nll": wdl_nll, "value_se": (value - batch.target_value) ** 2,
           "top1": (logp.argmax(dim=1) == batch.best).to(torch.float)}
    if score is not None:
        out["score_se"] = (score - batch.search_value) ** 2
    return out


# ----------------------------------------------------------------------------
# Labelling

def source_result(row: Dict[str, Any], board: chess.Board) -> Optional[float]:
    """Game result for the side to move from a source row ('value' in [-1, 0, 1] or a PGN 'result')."""
    v = row.get("value")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v in (-1, 0, 1):
        return float(v)
    r = row.get("result")
    if isinstance(r, str):
        white = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(r)
        if white is not None:
            return white if board.turn == chess.WHITE else -white
    return None


def label_board(engine: Any, board: chess.Board, multipv: int, nodes: int) -> Optional[Dict[str, Any]]:
    """Analyses one position with MultiPV `multipv` at `nodes` nodes; None if it has no legal move or no line."""
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None
    infos = engine.analyse(board, chess.engine.Limit(nodes=int(nodes)), multipv=int(multipv))
    if isinstance(infos, dict):
        infos = [infos]
    scored: List[Dict[str, Any]] = []
    seen = set()
    depths: List[int] = []
    seldepth = 0
    total_nodes = 0
    for info in infos:
        pv, score = info.get("pv"), info.get("score")
        if not pv or score is None:
            continue
        move = pv[0]
        if move in seen or move not in board.legal_moves:
            continue
        seen.add(move)
        rec: Dict[str, Any] = {"move": move.uci(), **score_record(score.pov(board.turn))}
        if info.get("lowerbound"):
            rec["bound"] = "lower"
        elif info.get("upperbound"):
            rec["bound"] = "upper"
        scored.append(rec)
        if "depth" in info:
            depths.append(int(info["depth"]))
        seldepth = max(seldepth, int(info.get("seldepth", 0)))
        total_nodes = max(total_nodes, int(info.get("nodes", 0)))
    if not scored:
        return None
    return {"fen": board.fen(), "legal": legal, "scored": scored, "multipv": int(multipv),
            "rest_upper_bound": len(scored) < len(legal), "depth": min(depths) if depths else None,
            "seldepth": seldepth or None, "nodes": total_nodes or None}
