"""Search-policy distillation: targets (distillation.py) and the pipeline steps 'label' and 'distill'.

The engine is tests/python/fake_uci_engine.py (a real UCI subprocess driven by
python-chess), so no engine binary is needed; with build/nags_enhanced (or
NAGS_ENGINE pointing at a UCI engine) a tiny real labelling run is added.
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch

import distillation
import training_pipeline as tp
import training_pipeline
from policy_encoding import POLICY_V3_DIM
from distillation import effective_cp, policy_target, row_targets, wdl_from_cp
from gnn_v3 import GNNAdapterV3, load_any_checkpoint

ROOT = Path(__file__).resolve().parents[2]
FAKE_ENGINE = [sys.executable, str(Path(__file__).with_name("fake_uci_engine.py"))]
MATE_IN_1 = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"       # Ra8#
PROMOTION = "8/P6k/8/8/8/8/6K1/8 w - - 0 1"
CHECKMATED = "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1"
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
DEFAULTS = tp.DISTILL_DEFAULTS


def tiny_config(tmp_path, **extra):
    distill = {"engine_command": FAKE_ENGINE, "multipv": 3, "nodes": 100, "epochs": 2, "batch_size": 4,
               "validation_fraction": 0.3, "model": {"hidden_dim": 16, "layers": 1, "num_bases": 2}}
    distill.update(extra.pop("distill_params", {}))
    cfg = {"data_dir": str(tmp_path / "data"), "model_dir": str(tmp_path / "models"), "logs_dir": str(tmp_path / "logs"),
           "pgn_file": str(tmp_path / "games.pgn"), "max_positions": 200, "batch_size": 16, "epochs": 1,
           "skip_opening_plies": 0, "model_params": {"hidden_dim": 16, "gnn_layers": 1, "policy_layers": 1, "value_layers": 1},
           "distill_params": distill}
    cfg.update(extra)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def source_rows():
    rows = [{"fen": chess.STARTING_FEN, "move": "e2e4", "value": 0.0, "game_id": 1, "ply": 0},
            {"fen": MATE_IN_1, "move": "a1a8", "value": 1.0, "game_id": 2, "ply": 30},
            {"fen": PROMOTION, "move": "a7a8q", "value": 1.0, "game_id": 3, "ply": 70},
            {"fen": CHECKMATED, "value": -1.0, "game_id": 2, "ply": 31},          # no legal move: skipped
            {"fen": STALEMATE, "result": "1/2-1/2", "game_id": 4},                 # no legal move: skipped
            {"fen": chess.STARTING_FEN, "move": "d2d4", "value": 1.0, "game_id": 5, "ply": 0},  # duplicate
            {"move": "e2e4"}]                                                       # no fen: ignored
    rng = random.Random(0)
    for g in range(6, 12):
        board = chess.Board()
        for ply in range(rng.randint(4, 30)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if any(board.legal_moves):
            rows.append({"fen": board.fen(), "result": ["1-0", "0-1", "1/2-1/2"][g % 3], "game_id": g})
    return rows


@pytest.fixture()
def labelled(tmp_path):
    pipe = tp.TrainingPipeline(tiny_config(tmp_path))
    src = tmp_path / "positions.jsonl"
    tp.write_jsonl(src, source_rows())
    return pipe, src, pipe.label_positions([src])


# ----------------------------------------------------------------------------
# Targets

def test_effective_cp_orders_mates_and_keeps_them_finite():
    kw = dict(cp_clip=2000.0, mate_cp=3000.0, mate_step=10.0)
    order = [{"mate": 1}, {"mate": 3}, {"cp": 5000}, {"cp": 150}, {"cp": 0}, {"cp": -150}, {"cp": -5000},
             {"mate": -5}, {"mate": -1}]
    values = [effective_cp(r, **kw) for r in order]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values)
    assert all(math.isfinite(v) for v in values)
    assert values[0] == 3000 and values[2] == 2000 and values[-1] == -3000
    assert effective_cp({"mate": 200}, **kw) > 2000  # a long mate still outranks every cp score


def test_wdl_model():
    for cp in (-1500, -400, -100, 0, 100, 400, 1500):
        w, d, l = wdl_from_cp(cp)
        assert abs(w + d + l - 1) < 1e-12 and min(w, d, l) >= 0
        assert wdl_from_cp(-cp) == pytest.approx((l, d, w))
        # expected score close to the NNUE trainer's sigmoid(cp / 400)
        assert abs((w + d / 2) - 1 / (1 + math.exp(-cp / 400))) < 0.03
    assert wdl_from_cp(0)[1] == pytest.approx(0.664, abs=1e-3)
    assert wdl_from_cp(3000)[0] > 0.9999
    with pytest.raises(ValueError):
        wdl_from_cp(0, a=-1)


def test_policy_target_only_on_scored_moves():
    scored = [{"move": "e2e4", "cp": 50}, {"move": "d2d4", "cp": 0}, {"move": "g1f3", "mate": 2}]
    target = dict(policy_target(scored, 50.0))
    assert set(target) == {"e2e4", "d2d4", "g1f3"} and abs(sum(target.values()) - 1) < 1e-12
    assert target["g1f3"] > 0.999 and target["e2e4"] / target["d2d4"] == pytest.approx(math.e)


def test_row_targets_blend_the_result():
    row = {"scored": [{"move": "e2e4", "cp": 0}], "result": 1.0}
    params = dict(DEFAULTS, result_weight=0.0)
    plain = row_targets(row, params)
    assert plain["wdl"] == pytest.approx(list(wdl_from_cp(0))) and plain["value"] == pytest.approx(0)
    blended = row_targets(row, dict(DEFAULTS, result_weight=0.5))
    w, d, l = wdl_from_cp(0)
    assert blended["wdl"] == pytest.approx([0.5 * w + 0.5, 0.5 * d, 0.5 * l])
    assert blended["search_value"] == pytest.approx(0) and blended["value"] > 0
    assert row_targets(dict(row, result=None), dict(DEFAULTS, result_weight=0.5))["wdl"] == pytest.approx(plain["wdl"])


# ----------------------------------------------------------------------------
# Pipeline steps

def test_defaults_live_in_code():
    assert "distill_params" not in tp.DEFAULT_CONFIG
    assert "distill_params" not in json.loads((ROOT / "training_config.json").read_text())


def test_distill_params_are_deep_merged(tmp_path):
    pipe = tp.TrainingPipeline(tiny_config(tmp_path))
    params = pipe.config["distill_params"]
    assert params["multipv"] == 3 and params["temperature_cp"] == DEFAULTS["temperature_cp"]
    assert params["model"]["hidden_dim"] == 16 and params["model"]["ffn_mult"] == DEFAULTS["model"]["ffn_mult"]


def test_label_rows(labelled):
    pipe, src, out = labelled
    assert out.parent == pipe.data_dir and out.name.startswith("distill_")
    rows = tp.read_jsonl(out)
    fens = [r["fen"] for r in rows]
    assert CHECKMATED not in fens and STALEMATE not in fens
    assert fens.count(chess.STARTING_FEN) == 1 and len(rows) >= 8
    for r in rows:
        board = chess.Board(r["fen"])
        assert r["legal"] == [m.uci() for m in board.legal_moves]
        assert 1 <= len(r["scored"]) <= 3 and r["multipv"] == 3
        assert all(chess.Move.from_uci(s["move"]) in board.legal_moves for s in r["scored"])
        assert all(("cp" in s) != ("mate" in s) for s in r["scored"])
        assert r["rest_upper_bound"] == (len(r["scored"]) < len(r["legal"]))
        assert r["nodes"] == 100 and r["depth"] == 1 and r["source"] == src.name and "game_id" in r
    by_fen = {r["fen"]: r for r in rows}
    mate = by_fen[MATE_IN_1]
    assert mate["scored"][0] == {"move": "a1a8", "mate": 1} and mate["result"] == 1.0
    promo = by_fen[PROMOTION]
    # a8=N / a8=B leave K+minor vs K (a draw: 0), so a king move (+100, pawn kept) comes third
    assert [s["move"] for s in promo["scored"]][:2] == ["a7a8q", "a7a8r"]
    assert [s["cp"] for s in promo["scored"]] == [900, 500, 100]
    assert by_fen[chess.STARTING_FEN]["result"] == 0.0
    for r in rows:
        g = r["game_id"]
        if g >= 6:  # PGN-style results "1-0" / "0-1" / "1/2-1/2" by g % 3, converted to the side to move
            white = {0: 1.0, 1: -1.0, 2: 0.0}[g % 3]
            assert r["result"] == (white if chess.Board(r["fen"]).turn == chess.WHITE else -white)


def test_label_limits_positions(tmp_path):
    pipe = tp.TrainingPipeline(tiny_config(tmp_path, distill_params={"label_positions": 3}))
    src = tmp_path / "positions.jsonl"
    tp.write_jsonl(src, source_rows())
    rows = tp.read_jsonl(pipe.label_positions([src]))
    assert 1 <= len(rows) <= 3


def test_distill_trains_a_v3_checkpoint(labelled):
    pipe, _, out = labelled
    torch.manual_seed(0)
    path = pipe.distill_training(out)
    assert path.exists() and path.name.startswith("gnn_v3_distill_")
    assert not list(pipe.model_dir.glob("*_model_*.pth"))  # the legacy steps never pick it up as theirs
    adapter = load_any_checkpoint(str(path), device=torch.device("cpu"))
    assert isinstance(adapter, GNNAdapterV3)
    extra = torch.load(path, weights_only=True)["extra"]
    assert extra["stage"] == "distill" and extra["dataset"] == str(out)
    for split in ("train", "val"):
        m = extra["metrics"][split]
        assert set(m) >= {"policy_nll", "top1", "wdl_nll", "value_mse", "positions"}
        assert all(math.isfinite(v) for v in m.values()) and 0 <= m["top1"] <= 1 and m["policy_nll"] >= 0
    # continue from the v3 checkpoint; a v2 checkpoint is refused
    again = pipe.distill_training(out, path)
    assert torch.load(again, weights_only=True)["extra"]["parent"] == str(path)
    with pytest.raises(RuntimeError):
        pipe.distill_training(out, ROOT / "tests" / "data" / "compat" / "gnn_v2.pt")


def test_distill_learns_the_search_move(tmp_path):
    """On a few positions repeated, the network's top move converges to the search's best move."""
    cfg = tiny_config(tmp_path, distill_params={"epochs": 30, "batch_size": 8, "learning_rate": 0.01,
                                                "validation_fraction": 0.0,
                                                "model": {"hidden_dim": 32, "layers": 2, "num_bases": 2}})
    pipe = tp.TrainingPipeline(cfg)
    src = tmp_path / "positions.jsonl"
    win_queen = "4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1"  # Qxd5: a unique best move, like the other two
    tp.write_jsonl(src, [{"fen": f} for f in (MATE_IN_1, PROMOTION, win_queen)])
    out = pipe.label_positions([src])
    torch.manual_seed(0)
    path = pipe.distill_training(out)
    metrics = torch.load(path, weights_only=True)["extra"]["metrics"]["train"]
    assert metrics["top1"] == 1.0, metrics


def test_cli_label_and_distill(tmp_path):
    cfg = tiny_config(tmp_path, distill_params={"epochs": 1})
    src = tmp_path / "positions.jsonl"
    tp.write_jsonl(src, source_rows())
    assert tp.main(["--config", cfg, "--step", "label", "--data", str(src)]) == 0
    data_dir = tmp_path / "data"
    labels = sorted(data_dir.glob("distill_*.jsonl"))
    assert len(labels) == 1
    assert tp.main(["--config", cfg, "--step", "distill"]) == 0  # newest distill_*.jsonl by default
    assert list((tmp_path / "models").glob("gnn_v3_distill_*.pth"))
    assert tp.main(["--config", cfg, "--step", "distill", "--data", str(tmp_path / "missing.jsonl")]) == 1


def test_label_without_sources_or_engine_is_an_error(tmp_path):
    pipe = tp.TrainingPipeline(tiny_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        pipe.label_positions()  # no data/training_data.jsonl or self_play_*.jsonl yet
    pipe = tp.TrainingPipeline(tiny_config(tmp_path, distill_params={"engine_command": None,
                                                                     "engine_path": str(tmp_path / "nope")}))
    with pytest.raises(FileNotFoundError):
        pipe.label_engine_command()


def test_label_sources_default_to_the_pipeline_datasets(tmp_path):
    pipe = tp.TrainingPipeline(tiny_config(tmp_path))
    tp.write_jsonl(pipe.data_dir / "training_data.jsonl", source_rows()[:3])
    tp.write_jsonl(pipe.data_dir / "self_play_20260101_000000.jsonl", source_rows()[6:9])
    assert [p.name for p in pipe.label_sources()] == ["training_data.jsonl", "self_play_20260101_000000.jsonl"]
    pipe.config["distill_params"]["label_sources"] = [str(pipe.data_dir / "self_play_*.jsonl")]
    assert [p.name for p in pipe.label_sources()] == ["self_play_20260101_000000.jsonl"]


def write_pgn(path, games=6):
    rng = random.Random(0)
    with open(path, "w") as f:
        for g in range(games):
            board = chess.Board()
            for _ in range(20):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            game = chess.pgn.Game.from_board(board)
            game.headers["Result"] = ["1-0", "0-1", "1/2-1/2"][g % 3]
            print(game, file=f, end="\n\n")


def test_supervised_legal_mask_is_opt_in(tmp_path):
    write_pgn(tmp_path / "games.pgn")
    pipe = tp.TrainingPipeline(tiny_config(tmp_path, supervised_legal_mask=True))
    torch.manual_seed(0)
    assert pipe.supervised_training(pipe.parse_pgn_to_dataset()).exists()
    assert "supervised_legal_mask" not in tp.DEFAULT_CONFIG  # default off: the step is unchanged


# ----------------------------------------------------------------------------
# Real engine (optional)

def _real_engine():
    env = os.environ.get("NAGS_ENGINE")
    if env and Path(env).exists():
        return Path(env)
    for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return c
    return None


@pytest.mark.skipif(_real_engine() is None, reason="nags_enhanced not built")
def test_label_with_the_real_engine(tmp_path):
    cfg = tiny_config(tmp_path, distill_params={"engine_command": None, "engine_path": str(_real_engine()),
                                                "nodes": 3000, "multipv": 3})
    pipe = tp.TrainingPipeline(cfg)
    src = tmp_path / "positions.jsonl"
    tp.write_jsonl(src, [{"fen": MATE_IN_1, "value": 1.0, "game_id": 1}, {"fen": PROMOTION, "game_id": 2},
                         {"fen": chess.STARTING_FEN, "game_id": 3}])
    rows = {r["fen"]: r for r in tp.read_jsonl(pipe.label_positions([src]))}
    assert len(rows) == 3
    assert rows[MATE_IN_1]["scored"][0] == {"move": "a1a8", "mate": 1}
    for r in rows.values():
        board = chess.Board(r["fen"])
        assert len(r["scored"]) == min(3, len(r["legal"])) and r["depth"] >= 1 and r["nodes"] >= 1
        assert all(chess.Move.from_uci(s["move"]) in board.legal_moves for s in r["scored"])
        assert all(isinstance(s.get("cp", s.get("mate")), int) for s in r["scored"])
        assert len({s["move"] for s in r["scored"]}) == len(r["scored"])


def test_unscored_moves_share_a_censored_rest_bucket():
    # 20 legal moves, 2 scored (30 and 20 cp): the 18 others score at most
    # 20 cp, so together they hold at most 18 * exp(-10 / 50) of the weight;
    # the target gives them rest_mass_scale of that as one bucket.
    row = {"fen": chess.STARTING_FEN, "legal": [m.uci() for m in chess.Board().legal_moves],
           "scored": [{"move": "e2e4", "cp": 30}, {"move": "d2d4", "cp": 20}]}
    params = dict(training_pipeline.DISTILL_DEFAULTS)
    assert params["rest_mass_scale"] == 0.02
    for scale in (0.02, 0.5):
        t = distillation.row_targets(row, {**params, "rest_mass_scale": scale})
        w = [1.0, math.exp(-10 / 50), scale * 18 * math.exp(-10 / 50)]
        assert abs(t["rest"] - w[2] / sum(w)) < 1e-9
    t = distillation.row_targets(row, params)
    w = [1.0, math.exp(-10 / 50), 0.02 * 18 * math.exp(-10 / 50)]
    assert abs(t["rest"] - w[2] / sum(w)) < 1e-9
    assert abs(dict(t["policy"])["e2e4"] - w[0] / sum(w)) < 1e-9
    assert abs(sum(p for _, p in t["policy"]) + t["rest"] - 1) < 1e-9
    # rest_mass_scale 0 gives the scored-moves-only target.
    t0 = distillation.row_targets(row, {**params, "rest_mass_scale": 0.0})
    assert t0["rest"] == 0 and abs(sum(p for _, p in t0["policy"]) - 1) < 1e-9
    # A row whose only scores are bounds is not used.
    assert not distillation.valid_row({**row, "scored": [{"move": "e2e4", "cp": 30, "bound": "lower"}]})


def test_rest_loss_is_finite_with_and_without_unscored_moves():
    from torch_geometric.data import Batch

    params = dict(training_pipeline.DISTILL_DEFAULTS)
    full = {"fen": "7k/8/8/8/8/8/8/K7 w - - 0 1"}  # every legal move scored: no rest bucket
    board = chess.Board(full["fen"])
    full["scored"] = [{"move": m.uci(), "cp": 0} for m in board.legal_moves]
    full["legal"] = [m.uci() for m in board.legal_moves]
    part = {"fen": chess.STARTING_FEN, "legal": [m.uci() for m in chess.Board().legal_moves],
            "scored": [{"move": "e2e4", "cp": 30}]}
    batch = Batch.from_data_list([distillation.DistillDataset([r], params)[0] for r in (full, part)])
    logits = torch.zeros(2, POLICY_V3_DIM, requires_grad=True)
    out = distillation.distill_losses(logits, torch.zeros(2, 3), None, batch)
    out["policy_nll"].sum().backward()
    assert torch.isfinite(out["policy_nll"]).all() and torch.isfinite(logits.grad).all()


def test_fast_graphs_give_identical_items():
    """distill_params.fast_graphs changes how the graphs are built, not one value of what is trained on."""
    import random

    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader  # the pipeline's loader

    rng = random.Random(3)
    rows = []
    for start in (chess.STARTING_FEN, "1n2k1n1/PPP3PP/8/8/8/8/ppp3pp/1N2K1N1 w - - 0 1",
                  "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"):
        for _ in range(12):
            board = chess.Board(start)
            for _ in range(rng.randint(0, 40)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            legal = [m.uci() for m in board.legal_moves]
            if len(legal) < 2:
                continue
            rows.append({"fen": board.fen(), "legal": legal, "result": 0.0,
                         "scored": [{"move": legal[0], "cp": 40}, {"move": legal[1], "mate": -3}]})
    params = dict(training_pipeline.DISTILL_DEFAULTS)
    assert params["fast_graphs"] is False and params["num_workers"] == 0  # today's behaviour by default
    ref = distillation.DistillDataset(rows, params)
    fast = distillation.DistillDataset(rows, {**params, "fast_graphs": True})
    for i in range(len(rows)):
        a, b = ref[i], fast[i]
        assert sorted(a.keys()) == sorted(b.keys())
        for k in a.keys():
            va, vb = a[k], b[k]
            if torch.is_tensor(va):
                assert va.dtype == vb.dtype and torch.equal(va, vb), k
            else:
                assert va == vb, k
    # And through a DataLoader with worker processes, batches as before.
    loader = DataLoader(fast, batch_size=8, num_workers=2)
    batches = list(loader)
    assert sum(b.num_graphs for b in batches) == len(rows)
    first = Batch.from_data_list([ref[i] for i in range(8)])
    assert torch.equal(batches[0].x, first.x) and torch.equal(batches[0].edge_index, first.edge_index)
