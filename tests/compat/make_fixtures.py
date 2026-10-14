#!/usr/bin/env python3
"""Writes the frozen backward-compatibility fixtures in tests/data/compat.

The fixtures record what the trainer and the GNN pipeline produced at commit
bb1e03c, while a network was being trained with that code. Future versions
of NAGS must keep loading them and keep interpreting them the same way
(tests/python/test_compat.py). They were generated once with

    python tests/compat/make_fixtures.py --root <checkout of bb1e03c> --out tests/data/compat

Do NOT regenerate them with newer code: that would turn the tests into a
comparison of the new code with itself. Add new fixtures for new formats
instead.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/8/1k6/8/8/8/6K1/7Q b - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="checkout of the code the fixtures describe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "tools" / "nnue"))
    sys.path.insert(0, str(root))
    import torch
    import train  # the baseline trainer

    # 1. A training checkpoint written part-way through a run (the state an
    #    interrupted run_round.py leaves behind), with the data it was made
    #    from. Design "current" of run_round.py, at a tiny hidden size.
    rng = np.random.default_rng(0)
    lines = [f"{FENS[i % len(FENS)]} | {int(rng.integers(-300, 300))} | {rng.choice(['1.0', '0.5', '0.0'])}"
             for i in range(320)]
    feats, stm, score, result = train.parse_chunk(lines)
    (out / "train_data.bin").write_bytes(train.encode(feats, stm, score, result).tobytes())
    net = out / "train_partial.nnue"
    train.main(["--data", str(out / "train_data.bin"), "--out", str(net), "--epochs", "3", "--stop-after", "1",
                "--hidden", "8", "--batch", "64", "--threads", "1", "--buckets", "8", "--device", "cpu"])
    net.unlink()  # only the checkpoint is a fixture (the network is re-exported after every epoch)

    # 2. A GNN checkpoint in the 'nags-gnn-v2' format, with the outputs it
    #    gives (legal-move priors and value) for a few positions.
    import chess
    from chess_graph import ChessGraph
    from gnn_evaluator import GNNEvaluator, legal_move_priors, save_checkpoint

    torch.manual_seed(0)
    model = GNNEvaluator(hidden_dim=16, gnn_layers=2, policy_layers=1, value_layers=1, device=torch.device("cpu"))
    save_checkpoint(model, str(out / "gnn_v2.pt"), {"note": "compatibility fixture"})
    builder = ChessGraph(device=torch.device("cpu"))
    expected = []
    for fen in FENS + ["8/P6k/8/8/8/8/6K1/8 w - - 0 1"]:  # the last one has promotions
        probs, value, _ = model.evaluate(builder.fen_to_graph(fen), mc_samples=1)
        moves = [m.uci() for m in chess.Board(fen).legal_moves]
        expected.append({"fen": fen, "moves": moves, "priors": legal_move_priors(probs[0], moves),
                         "value": float(value[0])})
    (out / "gnn_v2_expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote fixtures to {out}")


if __name__ == "__main__":
    main()
