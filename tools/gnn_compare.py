#!/usr/bin/env python3
"""Same positions, several ways to train a GNN move prior, compared on held-out games.

    A   legacy v2 GNN, supervised on the moves played (+ game result)   [step 'supervised']
    A2  A with the opt-in legal-move mask (supervised_legal_mask)
    B   v3 GNN distilled from MultiPV search scores                     [steps 'label' + 'distill']
    C   v3 GNN on the moves played (+ game result): B's architecture, A's targets

Positions come from nags_enhanced self-play games (fixed nodes per move,
random openings from tools/openings/nags_balanced.epd); a subset is labelled
with MultiPV search scores, and a tenth of the games is held out. Reported
on the held-out positions: top-1 agreement with the engine's best move and
with the move played, the negative log-likelihood of the engine's best
move, the prior mass on the engine's top-k, and the value's squared error
against the search's value and against the game result (baselines: a
uniform prior, a constant 0 value). Everything is written under --out
(steps are skipped when their output exists, so an interrupted run
continues); nothing else is touched.

    python tools/gnn_compare.py --out runs/gnn_compare --engine build/nags_enhanced

This measures move priors and values only, not playing strength (that
needs the hybrid with a trained network in a match).
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import training_pipeline as tp  # noqa: E402
from distillation import effective_cp, wdl_from_cp  # noqa: E402
from gnn_v3 import load_any_checkpoint  # noqa: E402

# Set from the command line in main().
EXP = Path(".")
ENGINE = ""
GAMES, PLAY_NODES = 300, 2000
LABEL_POSITIONS, MULTIPV, LABEL_NODES = 4000, 4, 15000
EPOCHS = 8
REST_SCALES = []
log = logging.getLogger("gnn_compare")


def selfplay(path):
    if path.exists():
        return
    rng = random.Random(7)
    openings = [l.split(";")[0].strip() for l in (ROOT / "tools" / "openings" / "nags_balanced.epd").read_text().splitlines() if l.strip()]
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    rows = []
    try:
        for g in range(GAMES):
            board = chess.Board(rng.choice(openings))
            game = []
            while not board.is_game_over(claim_draw=True) and board.ply() < 300:
                if board.ply() == 0:
                    engine.configure({"Clear Hash": None})
                move = engine.play(board, chess.engine.Limit(nodes=PLAY_NODES)).move
                game.append((board.fen(), move.uci(), board.turn))
                board.push(move)
            res = board.result(claim_draw=True)
            white = {"1-0": 1, "0-1": -1}.get(res, 0)
            for ply, (fen, mv, turn) in enumerate(game):
                if ply >= 6:
                    rows.append({"fen": fen, "move": mv, "value": white if turn == chess.WHITE else -white,
                                 "game_id": g, "ply": ply})
            if g % 25 == 0:
                log.info("self-play game %d/%d, %d positions", g, GAMES, len(rows))
    finally:
        engine.quit()
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def pipeline(name, distill=None, **cfg):
    d = EXP / name
    for sub in ("data", "models", "logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    config = {"data_dir": str(d / "data"), "model_dir": str(d / "models"), "logs_dir": str(d / "logs"),
              "engine_path": ENGINE, "epochs": EPOCHS, "seed": 0, "validation_fraction": 0.05,
              "distill_params": {"engine_path": ENGINE, "epochs": EPOCHS, **(distill or {})}, **cfg}
    path = d / "config.json"
    path.write_text(json.dumps(config))
    return tp.TrainingPipeline(str(path))


def main(argv=None):
    global EXP, ENGINE, GAMES, PLAY_NODES, LABEL_POSITIONS, MULTIPV, LABEL_NODES, EPOCHS, REST_SCALES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="working directory for games, labels, models and results")
    ap.add_argument("--engine", default=str(ROOT / "build" / "nags_enhanced"))
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--play-nodes", type=int, default=2000)
    ap.add_argument("--label-positions", type=int, default=4000)
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--label-nodes", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--rest-mass-scales", type=float, nargs="*", default=[],
                    help="extra B arms with these distillation rest_mass_scale values (B itself uses the default)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads")
    args = ap.parse_args(argv)
    EXP = Path(args.out).resolve()
    EXP.mkdir(parents=True, exist_ok=True)
    ENGINE = str(Path(args.engine).resolve())
    GAMES, PLAY_NODES = args.games, args.play_nodes
    LABEL_POSITIONS, MULTIPV, LABEL_NODES, EPOCHS = args.label_positions, args.multipv, args.label_nodes, args.epochs
    REST_SCALES = args.rest_mass_scales
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(EXP / "run.log"), logging.StreamHandler()])
    t0 = time.time()
    sp = EXP / "selfplay.jsonl"
    selfplay(sp)
    rows = tp.read_jsonl(sp)
    games = sorted({r["game_id"] for r in rows})
    random.Random(1).shuffle(games)
    test_games = set(games[: max(1, len(games) // 10)])
    log.info("%d positions from %d games; %d games held out", len(rows), len(games), len(test_games))

    # Label a random subset of positions (train and test games alike).
    labelled = EXP / "labels.jsonl"
    if not labelled.exists():
        lp = pipeline("label", distill={"multipv": MULTIPV, "nodes": LABEL_NODES, "label_positions": LABEL_POSITIONS})
        out = lp.label_positions([sp])
        out.rename(labelled)
    labels = tp.read_jsonl(labelled)
    played = {" ".join(r["fen"].split()[:4]): r["move"] for r in rows}
    for r in labels:
        r["played"] = played[" ".join(r["fen"].split()[:4])]
    train = [r for r in labels if r["game_id"] not in test_games]
    test = [r for r in labels if r["game_id"] in test_games]
    log.info("labelled: %d train, %d test positions (%.0fs so far)", len(train), len(test), time.time() - t0)

    def write(path, rs):
        with open(path, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
        return path

    models = {}
    # A: legacy supervised on the same training positions, played moves.
    pa = pipeline("A")
    if not list((EXP / "A/models").glob("supervised_model_*.pth")):
        pa.supervised_training(write(EXP / "A/data/train.jsonl",
                                     [{"fen": r["fen"], "move": r["played"], "value": r["result"], "game_id": r["game_id"]}
                                      for r in train]))
    models["A legacy v2, played moves"] = sorted((EXP / "A/models").glob("supervised_model_*.pth"))[-1]
    # A2: A with the opt-in legal-move mask on the policy loss.
    pa2 = pipeline("A2", supervised_legal_mask=True)
    if not list((EXP / "A2/models").glob("supervised_model_*.pth")):
        pa2.supervised_training(write(EXP / "A2/data/train.jsonl",
                                      [{"fen": r["fen"], "move": r["played"], "value": r["result"], "game_id": r["game_id"]}
                                       for r in train]))
    models["A2 legacy v2, played, legal mask"] = sorted((EXP / "A2/models").glob("supervised_model_*.pth"))[-1]
    # B: v3 distillation from the search's MultiPV scores.
    pb = pipeline("B")
    if not list((EXP / "B/models").glob("gnn_v3_distill_*.pth")):
        pb.distill_training(write(EXP / "B/data/train.jsonl", train))
    models["B v3, distilled from search"] = sorted((EXP / "B/models").glob("gnn_v3_distill_*.pth"))[-1]
    for scale in REST_SCALES:  # B with other weights for the unscored moves' pooled bucket
        name = f"B_rest{scale:g}"
        pbx = pipeline(name, distill={"rest_mass_scale": scale})
        if not list((EXP / name / "models").glob("gnn_v3_distill_*.pth")):
            pbx.distill_training(write(EXP / name / "data" / "train.jsonl", train))
        models[f"B v3, distilled, rest_mass_scale {scale:g}"] = sorted((EXP / name / "models").glob("gnn_v3_distill_*.pth"))[-1]
    # C: v3 on the played moves (one-hot policy) and game results.
    pc = pipeline("C", distill={"rest_mass_scale": 0.0, "result_weight": 1.0, "aux_score_weight": 0.0})
    if not list((EXP / "C/models").glob("gnn_v3_distill_*.pth")):
        pc.distill_training(write(EXP / "C/data/train.jsonl",
                                  [{**r, "scored": [{"move": r["played"], "cp": 0}], "rest_upper_bound": True} for r in train]))
    models["C v3, played moves"] = sorted((EXP / "C/models").glob("gnn_v3_distill_*.pth"))[-1]

    # Held-out evaluation.
    params = pb.config["distill_params"]
    kw = {"cp_clip": params["cp_clip"], "mate_cp": params["mate_cp"], "mate_step": params["mate_step"]}
    results = {}
    for name, path in models.items():
        adapter = load_any_checkpoint(str(path), device=torch.device("cpu"))
        adapter.model.eval()
        top_engine = top_played = 0
        nll = []
        mass_topk = []
        v_err = []
        r_err = []
        for i in range(0, len(test), 64):
            chunk = test[i : i + 64]
            with torch.no_grad():
                out = adapter.evaluate_fens([r["fen"] for r in chunk], [r["legal"] for r in chunk], mc_samples=1)
            for r, o in zip(chunk, out):
                pri = dict(zip(r["legal"], o["move_priors"]))
                best = max(r["scored"], key=lambda s: effective_cp(s, **kw))
                pick = max(pri, key=pri.get)
                top_engine += pick == best["move"]
                top_played += pick == r["played"]
                nll.append(-np.log(max(pri[best["move"]], 1e-12)))
                mass_topk.append(sum(pri[s["move"]] for s in r["scored"]))
                w = wdl_from_cp(effective_cp(best, **kw), params["wdl_a"], params["wdl_b"])
                v_err.append((o["value"] - (w[0] - w[2])) ** 2)
                r_err.append((o["value"] - r["result"]) ** 2)
        n = len(test)
        results[name] = {"top1_engine": top_engine / n, "top1_played": top_played / n, "nll_engine_best": float(np.mean(nll)),
                         "mass_on_engine_topk": float(np.mean(mass_topk)), "value_mse_vs_search": float(np.mean(v_err)),
                         "value_mse_vs_result": float(np.mean(r_err)),
                         "checkpoint": str(path)}
    # Reference: a uniform prior over the legal moves.
    results["constant 0 value"] = {"value_mse_vs_result": float(np.mean([r["result"] ** 2 for r in test]))}
    results["uniform prior"] = {"top1_engine": float(np.mean([1 / len(r["legal"]) for r in test])),
                                "nll_engine_best": float(np.mean([np.log(len(r["legal"])) for r in test])),
                                "mass_on_engine_topk": float(np.mean([len(r["scored"]) / len(r["legal"]) for r in test]))}
    (EXP / "results.json").write_text(json.dumps(results, indent=1))
    log.info("held-out positions: %d (from %d games); total %.0fs", len(test), len(test_games), time.time() - t0)
    for k, v in results.items():
        log.info("%-32s %s", k, {a: round(b, 4) for a, b in v.items() if isinstance(b, float)})


if __name__ == "__main__":
    main()
