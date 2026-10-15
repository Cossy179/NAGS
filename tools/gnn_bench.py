#!/usr/bin/env python3
"""Inference benchmark of the GNN evaluators: legacy 'nags-gnn-v2' and relational 'nags-gnn-v3'.

For each model, device and batch size it times, separately,
  * graph construction: builder.fens_to_batch on the CPU (FEN parsing, python-chess,
    tensor building) - the part rpc_server.py does before calling the model;
  * model: model.evaluate(batch, mc_samples) including the transfer to the device and
    the results back to the CPU - the rest of an rpc_server.py request;
and prints the median of --repeats runs (after --warmup runs) in milliseconds and
positions per second. Weights do not change the timing, so without --v2 / --v3
checkpoints it uses untrained models of the default sizes (v2: the pipeline's
model_params; v3: distill_params.model).

The CPU may be shared (e.g. with an engine match): the default is one torch
thread, and the load average is printed with the results.

  python tools/gnn_bench.py
  python tools/gnn_bench.py --batch-sizes 1,8 --repeats 5 --json bench.json
  python tools/gnn_bench.py --v2 models/production_model.pth --v3 models/gnn_v3_distill_X.pth
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import chess  # noqa: E402
import torch  # noqa: E402

from chess_graph import ChessGraph  # noqa: E402
from gnn_evaluator import GNNEvaluator  # noqa: E402
from gnn_v3 import GNNAdapter, GNNAdapterV2, GNNAdapterV3, RelationalGNNEvaluator, load_any_checkpoint  # noqa: E402
from relational_graph import RelationalChessGraph  # noqa: E402

PROMOTION_START = "1n2k1n1/PPP3PP/8/8/8/8/ppp3pp/1N2K1N1 w - - 0 1"


def positions(n: int, seed: int = 0) -> List[str]:
    """n distinct FENs from random games (every eighth from a promotion-heavy start)."""
    rng = random.Random(seed)
    out: List[str] = []
    seen = set()
    while len(out) < n:
        board = chess.Board(PROMOTION_START if len(out) % 8 == 7 else chess.STARTING_FEN)
        for _ in range(rng.randint(4, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if any(board.legal_moves) and board.fen() not in seen:
            seen.add(board.fen())
            out.append(board.fen())
    return out


def make_model(kind: str, device: torch.device, path: Optional[str], tiny: bool) -> GNNAdapter:
    if path:
        adapter = load_any_checkpoint(path, device=device)
        if adapter.format != {"v2": "nags-gnn-v2", "v3": "nags-gnn-v3"}[kind]:
            raise SystemExit(f"--{kind} {path} is a {adapter.format} checkpoint")
        return adapter
    torch.manual_seed(0)
    if kind == "v2":
        cfg = dict(hidden_dim=16, gnn_layers=1, policy_layers=1, value_layers=1) if tiny else \
            dict(hidden_dim=128, gnn_layers=6, policy_layers=4, value_layers=2)
        model = GNNEvaluator(device=device, **cfg)
        return GNNAdapterV2(model.eval(), ChessGraph(device=torch.device("cpu")))
    cfg = dict(hidden_dim=16, layers=1, num_bases=2) if tiny else {}
    model = RelationalGNNEvaluator(device=device, **cfg)
    return GNNAdapterV3(model.eval(), RelationalChessGraph(device=torch.device("cpu")))


def bench_one(adapter: GNNAdapter, fens: Sequence[str], batch: int, repeats: int, warmup: int,
              mc_samples: int, device: torch.device) -> Dict[str, Any]:
    builder = adapter.builder
    builder.device = torch.device("cpu")  # graphs are built on the CPU; evaluate() moves them
    chunk = list(fens[:batch])
    build, model = [], []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        graph = builder.fens_to_batch(chunk)
        t1 = time.perf_counter()
        adapter.evaluate(graph, mc_samples=mc_samples)  # returns CPU tensors
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= warmup:
            build.append(t1 - t0)
            model.append(t2 - t1)
    b, m = statistics.median(build), statistics.median(model)
    return {"model": adapter.format, "device": str(device), "batch": batch,
            "build_ms": 1000 * b, "model_ms": 1000 * m, "total_ms": 1000 * (b + m),
            "build_pos_s": batch / b, "model_pos_s": batch / m, "total_pos_s": batch / (b + m)}


def run(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--batch-sizes", default="1,8,64,256")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--mc-samples", type=int, default=3, help="MC-dropout samples (training_config rpc_mc_samples)")
    p.add_argument("--threads", type=int, default=1, help="torch threads (0: torch's default)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="auto: the CPU, and CUDA too when available")
    p.add_argument("--models", default="v2,v3")
    p.add_argument("--v2", help="a nags-gnn-v2 checkpoint (default: untrained, default size)")
    p.add_argument("--v3", help="a nags-gnn-v3 checkpoint (default: untrained, default size)")
    p.add_argument("--tiny", action="store_true", help="tiny models (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write the rows to this file")
    args = p.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
    devices = [torch.device("cpu")] if args.device in ("auto", "cpu") else []
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        devices.append(torch.device("cuda"))
    fens = positions(max(sizes), args.seed)
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    print(f"torch {torch.__version__}, {torch.get_num_threads()} thread(s), {os.cpu_count()} CPUs, "
          f"load average {load[0]:.2f}, " if load else f"torch {torch.__version__}, ", end="")
    print(f"mc_samples {args.mc_samples}, median of {args.repeats} run(s)", flush=True)
    rows: List[Dict[str, Any]] = []
    header = (f"{'model':<12} {'device':<6} {'batch':>5} {'build ms':>9} {'model ms':>9} {'total ms':>9} "
              f"{'build pos/s':>11} {'model pos/s':>11} {'total pos/s':>11}")
    print(header)
    for device in devices:
        for kind in [k.strip() for k in args.models.split(",") if k.strip()]:
            adapter = make_model(kind, device, getattr(args, kind), args.tiny)
            params = sum(t.numel() for t in adapter.model.parameters())
            for bs in sizes:
                row = bench_one(adapter, fens, bs, args.repeats, args.warmup, args.mc_samples, device)
                row["params"] = params
                rows.append(row)
                print(f"{row['model']:<12} {device.type:<6} {bs:>5} {row['build_ms']:>9.1f} {row['model_ms']:>9.1f} "
                      f"{row['total_ms']:>9.1f} {row['build_pos_s']:>11.1f} {row['model_pos_s']:>11.1f} "
                      f"{row['total_pos_s']:>11.1f}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"load_average": load, "threads": torch.get_num_threads(),
                                               "mc_samples": args.mc_samples, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run()
