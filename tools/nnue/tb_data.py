#!/usr/bin/env python3
"""Endgame supervision from Syzygy tablebases: training positions with exact results, and an endgame test.

  generate  random legal positions of given materials (--material KRPvKR KPvKP ...), side to move not in
            check and, by default, without a capture or promotion to make (quiet, like nags_datagen's kept
            positions). The result is the tablebase's, under the fifty-move rule: a win that needs more than
            100 plies to a capture or pawn move (a cursed win or blessed loss) is a draw, as it is at a
            halfmove clock of 0. The score is NAGS's search at --nodes without tablebases (what the network
            is trained to predict), and a position it scores as mate is dropped, as nags_datagen does. Writes
            32-byte records (no game index: the positions are independent) and <out>.tb.json with the counts
            by material and result.
  measure   the same kind of positions, a sample per material: does NAGS's score at --nodes (no tablebases)
            say the tablebase's result? A score beyond +-band is a win or a loss, within it a draw. Reports
            the share right per material and how it errs (draws scored as wins, wins scored as draws, ...):
            fortresses, wrong-coloured bishops and zugzwangs show up as draws called wins. With --json, the
            numbers to compare two networks (EvalFile) or builds on the same positions (same --seed).

  python tools/nnue/tb_data.py generate --syzygy /tb/345 --material KRvKB KRPvKR KBPvK KPvKP KQvKR \\
      --count 20000 --engine build/nags_enhanced --nodes 5000 --workers 5 --out data/tb/tb_1.bin --seed 1
  python tools/nnue/tb_data.py measure --syzygy /tb/345 --material KRvKB KRPvKR KBPvK KPvKP KQvKR \\
      --count 400 --engine build/nags_enhanced --nodes 10000 --workers 3 --json tb_net8.json

A material is written as python-chess and Syzygy name it: White's pieces, 'v', Black's (the stronger side
need not be White: both colours are played, the side to move at random).
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading

import chess
import chess.engine
import chess.syzygy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402

RESULTS = ("loss", "draw", "win")  # for the side to move


def parse_material(name):
    """(white pieces, black pieces) as piece letters, e.g. 'KRPvKR' -> ('KRP', 'KR')."""
    w, sep, b = name.partition("v")
    if not sep or not w.startswith("K") or not b.startswith("K") or any(c not in "KQRBNP" for c in w + b) \
            or w.count("K") != 1 or b.count("K") != 1:
        sys.exit(f"--material {name}: like KRPvKR (each side's king first)")
    return w, b


def random_position(material, rng, quiet=True, tries=1000):
    """A random legal position of the material, the side to move not in check (and with nothing to capture or
    promote when `quiet`), or None."""
    w, b = parse_material(material)
    for _ in range(tries):
        board = chess.Board(None)
        squares = rng.sample(range(64), len(w) + len(b))
        ok = True
        for letter, sq in zip(w.upper() + b.lower(), squares):
            if letter in "Pp" and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            board.set_piece_at(sq, chess.Piece.from_symbol(letter))
        if not ok:
            continue
        board.turn = rng.random() < 0.5
        if not board.is_valid() or board.is_check() or not any(board.legal_moves):
            continue
        if quiet and any(board.is_capture(m) or m.promotion for m in board.legal_moves):
            continue
        return board
    return None


def tb_result(tb, board):
    """The side to move's result under the fifty-move rule (halfmove clock 0): 'win', 'draw' or 'loss'."""
    wdl = tb.probe_wdl(board)
    return "win" if wdl == 2 else "loss" if wdl == -2 else "draw"


def command(text):
    return text.split() if " " in text and not os.path.exists(text) else [text]


class Workers:
    """One engine per thread, created on first use."""

    def __init__(self, args):
        self.args, self.local, self.all, self.lock = args, threading.local(), [], threading.Lock()

    def engine(self):
        if not hasattr(self.local, "eng"):
            eng = chess.engine.SimpleEngine.popen_uci(command(self.args.engine))
            opts = dict(o.split("=", 1) for o in self.args.option or [])
            opts.setdefault("SyzygyPath", "")  # the engine must not know the answer
            eng.configure({k: v for k, v in opts.items() if k in eng.options})
            self.local.eng = eng
            with self.lock:
                self.all.append(eng)
        return self.local.eng

    def score(self, board):
        """(centipawns or None for a mate, the mate's sign or 0): NAGS's score for the side to move at --nodes,
        from a new game."""
        info = self.engine().analyse(board, chess.engine.Limit(nodes=self.args.nodes), game=object())
        s = info["score"].relative
        return (None, 1 if s > chess.engine.Cp(0) else -1) if s.is_mate() else (s.score(), 0)

    def close(self):
        for e in self.all:
            e.quit()


def positions(args, tb, material, count, seed):
    rng = random.Random(f"{seed}:{material}")
    out = []
    while len(out) < count:
        board = random_position(material, rng, quiet=not args.any)
        if board is None:
            sys.exit(f"{material}: no legal quiet position found")
        out.append((board, tb_result(tb, board)))
    return out


def generate(args):
    work = Workers(args)
    rows, counts = [], {}
    try:
        with chess.syzygy.open_tablebase(args.syzygy) as tb:
            todo = [(m, b, r) for m in args.material for b, r in positions(args, tb, m, args.count, args.seed)]
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            scores = list(pool.map(lambda t: work.score(t[1]), todo))
    finally:
        work.close()
    for (material, board, result), (score, _) in zip(todo, scores):
        c = counts.setdefault(material, {"positions": 0, "dropped_mate": 0, "win": 0, "draw": 0, "loss": 0})
        if score is None:
            c["dropped_mate"] += 1
            continue
        c["positions"] += 1
        c[result] += 1
        white = board.turn == chess.WHITE
        res_white = {"win": 1.0, "draw": 0.5, "loss": 0.0}[result]
        rows.append((board.fen(), score if white else -score, res_white if white else 1.0 - res_white))
    order = np.random.default_rng(args.seed).permutation(len(rows))  # materials mixed through the file
    rows = [rows[i] for i in order]
    feats, stm, score, result = train.parse_chunk([f"{f} | {s} | {r}" for f, s, r in rows])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    train.encode(feats, stm, score, result).tofile(args.out)
    with open(args.out + ".tb.json", "w") as fh:
        json.dump({"materials": counts, "nodes": args.nodes, "engine": args.engine, "options": args.option or [],
                   "seed": args.seed, "quiet": not args.any, "positions": len(rows)}, fh, indent=1)
    for m, c in counts.items():
        print(f"{m}: {c['positions']} positions (win {c['win']}, draw {c['draw']}, loss {c['loss']}; "
              f"{c['dropped_mate']} dropped for mate scores)")
    print(f"wrote {args.out}: {len(rows)} positions")
    return counts


def measure(args):
    work = Workers(args)
    try:
        with chess.syzygy.open_tablebase(args.syzygy) as tb:
            todo = [(m, b, r) for m in args.material for b, r in positions(args, tb, m, args.count, args.seed)]
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            scores = list(pool.map(lambda t: work.score(t[1]), todo))
    finally:
        work.close()
    report = {"nodes": args.nodes, "band": args.band, "seed": args.seed, "options": args.option or [],
              "materials": {}}
    for (material, _, result), (score, mate) in zip(todo, scores):
        if score is None:  # a mate score: its sign says win or loss
            said = "win" if mate > 0 else "loss"
        else:
            said = "win" if score > args.band else "loss" if score < -args.band else "draw"
        m = report["materials"].setdefault(material, {"positions": 0, "right": 0, "confusion": {}})
        m["positions"] += 1
        m["right"] += said == result
        key = f"{result} as {said}"
        m["confusion"][key] = m["confusion"].get(key, 0) + 1
    total = sum(m["positions"] for m in report["materials"].values())
    right = sum(m["right"] for m in report["materials"].values())
    report["overall"] = right / total if total else 0.0
    print(f"tablebase result said by the score at {args.nodes} nodes (band +-{args.band} cp): "
          f"{right}/{total} ({report['overall']:.1%})")
    for material, m in report["materials"].items():
        wrong = {k: v for k, v in sorted(m["confusion"].items(), key=lambda kv: -kv[1])
                 if k.split(" as ")[0] != k.split(" as ")[1]}
        print(f"  {material:<8} {m['right']}/{m['positions']} ({m['right'] / m['positions']:.0%})"
              + (": wrong " + ", ".join(f"{k} {v}" for k, v in wrong.items()) if wrong else ""))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--syzygy", required=True, help="directory of Syzygy tables (WDL files suffice)")
        p.add_argument("--material", nargs="+", required=True)
        p.add_argument("--count", type=int, required=True, help="positions per material")
        p.add_argument("--engine", required=True, help="NAGS (a UCI command); probed without tablebases")
        p.add_argument("--option", action="append", metavar="NAME=VALUE", help="UCI option (e.g. EvalFile=...)")
        p.add_argument("--nodes", type=int, default=5000 if name == "generate" else 10000)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--any", action="store_true", help="also positions with a capture or promotion to make")
        if name == "generate":
            p.add_argument("--out", required=True, help="32-byte records (.bin)")
        else:
            p.add_argument("--band", type=int, default=100, help="a score within +-band is a draw")
            p.add_argument("--json")
    args = ap.parse_args(argv)
    for m in args.material:
        parse_material(m)
    if args.cmd == "generate":
        generate(args)
    else:
        measure(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
