#!/usr/bin/env python3
"""End-to-end throughput of the hybrid engine for each way of serving the GNN (docs/TESTING.md, "End to end").

tools/gnn_bench.py times the network alone. This times what the engine gets: for each
configuration (rpc_server.py backend, providers, fp16, request batching, engine-built graphs,
pipelining) one rpc_server.py is started, and `nags` searches the same positions with the same
move time, its MCTS arm always on. Reported per configuration, from the engine's own
`info string nags` line and its alpha-beta `info` lines:

  sims/s      MCTS simulations per second of MCTS time (the number that matters for the hybrid)
  pos/s       positions the server evaluated per second of MCTS time (cache hits excluded)
  rtt ms      mean round trip of a request, and positions per request
  ab depth    the alpha-beta arm's mean completed depth, and ab knps its speed: what the server's
              CPU use costs the rest of the engine

The first search of each configuration is a warm-up and is not counted (it includes ONNX
export and TensorRT engine builds, which can take minutes).

  python tools/e2e_bench.py --model models/gnn_v3.pth --device cuda --preset gpu --positions 40 \\
      --movetime 2000 --json e2e_gpu.json
  python tools/e2e_bench.py --model models/gnn_v3.pth --preset cpu --positions 10 --movetime 1000
  python tools/e2e_bench.py --model M.pth --config "mine;--backend onnx --providers cuda;NNBatch=16"

A configuration is "name;rpc_server.py arguments;engine options" (options NAME=VALUE, comma
separated). --only keeps the named configurations of a preset.
"""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid_eval import Engine, _exe, parse_summary  # noqa: E402

OPENINGS = str(ROOT / "tools" / "openings" / "nags_balanced.epd")


class Config(NamedTuple):
    name: str
    server: List[str]          # rpc_server.py arguments (besides --model, --device, --port)
    options: Dict[str, str]    # nags UCI options (besides the ones every run sets)


def _cfg(name: str, server: str, options: str) -> Config:
    opts = dict(o.split("=", 1) for o in options.split(",") if o) if options else {}
    return Config(name, shlex.split(server), opts)


# What changed the throughput on the way here (docs/ARCHITECTURE.md, "Neural inference"), cumulatively
# where it makes sense: the original one-request-per-simulation path first.
BATCHED = "NNBatch=32,NNPipeline=true"
PRESETS: Dict[str, List[Config]] = {
    "cpu": [
        _cfg("legacy-b1", "--backend legacy", "NNBatch=1"),
        _cfg("torch-b1", "--backend torch", "NNBatch=1"),
        _cfg("torch-b8", "--backend torch", "NNBatch=8"),
        _cfg("torch-b8-graphs", "--backend torch", "NNBatch=8,NNGraphs=true"),
        _cfg("onnx-cpu-b8-graphs", "--backend onnx --providers cpu", "NNBatch=8,NNGraphs=true"),
    ],
    "gpu": [
        _cfg("legacy-b1", "--backend legacy", "NNBatch=1"),
        _cfg("torch-b1", "--backend torch", "NNBatch=1"),
        _cfg("torch-b32-pipe", "--backend torch", BATCHED),
        _cfg("torch-b32-pipe-graphs", "--backend torch", BATCHED + ",NNGraphs=true"),
        _cfg("torch-fp16-b32-pipe-graphs", "--backend torch --fp16", BATCHED + ",NNGraphs=true"),
        _cfg("onnx-cuda-b32-pipe-graphs", "--backend onnx --providers cuda,cpu", BATCHED + ",NNGraphs=true"),
        _cfg("onnx-cuda-fp16-b32-pipe-graphs", "--backend onnx --providers cuda,cpu --fp16",
             BATCHED + ",NNGraphs=true"),
        _cfg("onnx-trt-fp16-b32-pipe-graphs", "--backend onnx --providers tensorrt,cuda,cpu --fp16",
             BATCHED + ",NNGraphs=true"),
        _cfg("onnx-trt-fp16-b64-pipe-graphs", "--backend onnx --providers tensorrt,cuda,cpu --fp16",
             "NNBatch=64,NNPipeline=true,NNGraphs=true"),
    ],
}


def parse_config(spec: str) -> Config:
    parts = spec.split(";")
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"configuration {spec!r}: expected 'name;server arguments;engine options'")
    return _cfg(*parts)


def last_info(lines: Sequence[str]) -> Dict[str, int]:
    """depth and nps of the last alpha-beta 'info depth' line."""
    for line in reversed(lines):
        if line.startswith("info depth ") and " nps " in line:
            f = line.split()
            return {"depth": int(f[f.index("depth") + 1]), "nps": int(f[f.index("nps") + 1])}
    return {"depth": 0, "nps": 0}


def summarise(name: str, runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mcts_ms = sum(r["mcts_ms"] for r in runs)
    sims = sum(r["mcts_sims"] for r in runs)
    positions = sum(r["rpc_positions"] for r in runs)
    requests = sum(r["rpc_requests"] for r in runs)
    wait = sum(r["rpc_wait_ms"] for r in runs)
    return {
        "config": name,
        "searches": len(runs),
        "sims_per_s": sims / (mcts_ms / 1000) if mcts_ms else 0.0,
        "server_positions_per_s": positions / (mcts_ms / 1000) if mcts_ms else 0.0,
        "rtt_ms": wait / requests if requests else 0.0,
        "positions_per_request": positions / requests if requests else 0.0,
        "cache_hits": sum(r["cache_hits"] for r in runs),
        "rpc_failures": sum(r["rpc_failures"] for r in runs),
        "ab_depth": statistics.mean(r["ab_depth"] for r in runs) if runs else 0.0,
        "ab_knps": statistics.mean(r["ab_nps"] for r in runs) / 1000 if runs else 0.0,
    }


def run_config(cfg: Config, args: argparse.Namespace, fens: Sequence[str], log_dir: Path) -> Dict[str, Any]:
    from training_pipeline import ServiceProcess

    server = ["--model", args.model, "--device", args.device, "--mc-samples", str(args.mc_samples), *cfg.server]
    options = {"UseNN": "true", "UseMetaLearner": "false", "MctsMinTime": "0", "Threads": "1",
               "Hash": str(args.hash), **cfg.options}
    with ServiceProcess("rpc_server.py", server, log_dir / f"e2e_{cfg.name}.log",
                        startup_timeout=args.startup_timeout) as rpc:
        options["NNPort"] = str(rpc.port)
        engine = Engine([args.engine], options)
        try:
            go = f"go movetime {args.movetime}"
            engine.go(fens[0], go)  # warm-up: model load, ONNX export, TensorRT engine build
            runs = []
            for fen in fens:
                lines = engine.go(fen, go)
                f = parse_summary(lines)
                info = last_info(lines)
                runs.append({"fen": fen, "mcts_sims": int(f.get("mcts_sims", 0)), "mcts_ms": float(f.get("mcts_ms", 0)),
                             "rpc_requests": int(f.get("rpc_requests", 0)),
                             "rpc_positions": int(f.get("rpc_positions", 0)),
                             "rpc_wait_ms": float(f.get("rpc_wait_ms", 0)), "cache_hits": int(f.get("cache_hits", 0)),
                             "rpc_failures": int(f.get("rpc_failures", 0)), "ab_depth": info["depth"],
                             "ab_nps": info["nps"]})
        finally:
            engine.close()
    out = summarise(cfg.name, runs)
    out["server_args"] = server
    out["engine_options"] = cfg.options
    if out["rpc_failures"]:
        out["warning"] = f"{out['rpc_failures']} failed requests: see {log_dir / f'e2e_{cfg.name}.log'}"
    return out


def format_table(rows: Sequence[Dict[str, Any]]) -> str:
    head = f"{'configuration':34} {'sims/s':>8} {'pos/s':>8} {'rtt ms':>7} {'pos/req':>7} {'ab depth':>8} {'ab knps':>8}"
    lines = [head, "-" * len(head)]
    base = rows[0]["sims_per_s"] if rows and rows[0]["sims_per_s"] else 0.0
    for r in rows:
        if "error" in r:
            lines.append(f"{r['config']:34} failed: {r['error']}")
            continue
        speed = f" ({r['sims_per_s'] / base:.1f}x)" if base else ""
        lines.append(f"{r['config']:34} {r['sims_per_s']:8.0f} {r['server_positions_per_s']:8.0f} {r['rtt_ms']:7.1f} "
                     f"{r['positions_per_request']:7.1f} {r['ab_depth']:8.1f} {r['ab_knps']:8.0f}{speed}"
                     + (f"  [{r['warning']}]" if "warning" in r else ""))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint served by rpc_server.py")
    parser.add_argument("--device", default="cpu", help="rpc_server.py --device (cuda for the GPU)")
    parser.add_argument("--engine", default=None, help="the hybrid engine (default: build/nags)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--only", nargs="+", help="configurations of the preset to run")
    parser.add_argument("--config", action="append", default=[], help="'name;server arguments;engine options'")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--plies", type=int, default=16, help="random plies after an opening")
    parser.add_argument("--openings", default=OPENINGS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--movetime", type=int, default=1000, help="ms per search")
    parser.add_argument("--mc-samples", type=int, default=1, help="rpc_server.py --mc-samples")
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--startup-timeout", type=float, default=600.0, help="s for a server to start")
    parser.add_argument("--log-dir", default=".", help="where the servers' logs go")
    parser.add_argument("--json", help="write every configuration's numbers here")
    args = parser.parse_args(argv)

    configs = list(PRESETS[args.preset]) if args.preset else []
    if args.only:
        unknown = set(args.only) - {c.name for c in configs}
        if unknown:
            parser.error(f"not in the preset: {', '.join(sorted(unknown))}")
        configs = [c for c in configs if c.name in args.only]
    configs += [parse_config(s) for s in args.config]
    if not configs:
        parser.error("give --preset and/or --config")
    args.engine = args.engine or _exe("nags")

    from search_benefit import sample_positions

    fens = sample_positions(args.openings, args.positions, args.plies, args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configs:
        start = time.time()
        print(f"== {cfg.name}: {' '.join(cfg.server)} | {cfg.options}", flush=True)
        try:
            rows.append(run_config(cfg, args, fens, log_dir))
        except Exception as e:  # a provider that is not installed, a server that does not start
            rows.append({"config": cfg.name, "error": str(e)})
        print(f"   {time.time() - start:.0f} s", flush=True)
    print(f"\n{len(fens)} positions, {args.movetime} ms a search, device {args.device}, "
          f"--mc-samples {args.mc_samples}\n")
    print(format_table(rows))
    if args.json:
        Path(args.json).write_text(json.dumps({"positions": len(fens), "movetime": args.movetime, "device": args.device,
                                               "results": rows}, indent=2))
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
