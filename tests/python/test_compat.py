"""Backward compatibility with what the code of commit bb1e03c produces.

A network was being trained with that code while NAGS kept changing, so
these tests pin, against frozen fixtures rather than the current code:

* every NNUE file format the engine reads (versions 1-3: output buckets,
  king buckets with mirroring, SCReLU), down to the exact integer evaluation
  of each position, in the trainer's reference and in the engine;
* the trainer's command-line defaults, constants, record layout and
  run_round.py designs, which an interrupted training round depends on;
* resuming a training checkpoint written by that trainer;
* loading and serving a 'nags-gnn-v2' GNN checkpoint with the same outputs.

Fixtures: tests/python/compat_nets.py (networks) and tests/data/compat
(written once by tests/compat/make_fixtures.py; never regenerate them).
See docs/COMPATIBILITY.md.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import compat_nets  # noqa: E402
import run_round  # noqa: E402
import train  # noqa: E402

FIXTURES = ROOT / "tests" / "data" / "compat"

# Integer evaluations (centipawns, side to move) of compat_nets.POSITIONS,
# recorded with the engine and the trainer's reference of commit bb1e03c,
# which agreed on every one.
GOLDEN_EVALS = {
    "v1": [6929, 6178, 6310, 2489, 2025, 3659, -1854, -32, 1796, 1060],
    "v2_b8": [-714, 3148, 1036, 8262, 8644, 778, 2922, 659, 4661, -2829],
    "v3_screlu_b8": [-6176, -2826, -4441, -6083, -1066, -5003, -3961, -5718, 3814, -2308],
    "v3_mirror": [-4395, -7803, -9608, -1819, -13080, 3287, -3733, 753, -4088, -1199],
    "v3_kb4_screlu_b8": [1523, 4332, 4180, 5684, 5854, -2963, 3547, 1276, -1329, -1963],
    "v3_kb16_screlu_b4": [-865, 2941, 3065, 7228, 4816, 55, 549, 448, -218, 3013],
}


def _final_fen(fen, moves):
    board = chess.Board(fen)
    for m in moves:
        board.push_uci(m)
    return board.fen()


@pytest.mark.parametrize("name", sorted(compat_nets.DESIGNS))
def test_reference_reads_every_format(tmp_path, name):
    path = tmp_path / f"{name}.nnue"
    path.write_bytes(compat_nets.network_bytes(name))
    net = train.read_net(str(path))
    got = [train.quantized_eval(net, _final_fen(f, m)) for f, m in compat_nets.POSITIONS]
    assert got == GOLDEN_EVALS[name]


def _engine():
    for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return c
    return None


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
@pytest.mark.parametrize("name", sorted(compat_nets.DESIGNS))
def test_engine_reads_every_format(tmp_path, name):
    path = tmp_path / f"{name}.nnue"
    path.write_bytes(compat_nets.network_bytes(name))
    cmds = f"setoption name EvalFile value {path}\n"
    for fen, moves in compat_nets.POSITIONS:
        cmds += f"position fen {fen}" + (" moves " + " ".join(moves) if moves else "") + "\neval\n"
    out = subprocess.run([str(_engine())], input=cmds, capture_output=True, text=True, timeout=60).stdout
    if "expects hidden size" in out and f"hidden size {compat_nets.HIDDEN}" not in out:
        pytest.skip("engine built with another NAGS_NNUE_HIDDEN")
    lines = out.splitlines()
    assert any("NNUE evaluation" in l for l in lines), out
    assert [int(l.split()[3]) for l in lines if l.startswith("info string eval")] == GOLDEN_EVALS[name]


def test_trainer_constants_and_layouts():
    assert (train.FEATURES, train.MAX_PIECES, train.QA, train.QB, train.SCALE) == (768, 32, 255, 64, 400)
    assert train.WEIGHT_CLIP == 1.98 and train.MAGIC == b"NAGSNNUE" and train.RECORD_BYTES == 32
    assert (train.MAX_BUCKETS, train.MAX_KING_BUCKETS) == (8, 32)
    assert train.KING_LAYOUTS["4"] == compat_nets.KB4 and train.KING_LAYOUTS["16"] == compat_nets.KB16
    assert train.KING_LAYOUTS["8"] == [0, 1, 2, 3, 4, 4, 5, 5] + [6] * 8 + [7] * 16
    assert train.KING_LAYOUTS["mirror"] == [0] * 32 and train.KING_LAYOUTS["none"] is None
    assert [train.bucket_of(p, 8) for p in range(1, 33)] == [(p - 1) * 8 // 32 for p in range(1, 33)]


def _trainer_parser():
    """The argparse parser train.main builds (captured, not run)."""

    class Captured(Exception):
        pass

    def capture(self, *a, **k):
        raise Captured(self)

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        train.main([])
    except Captured as c:
        return c.args[0]
    finally:
        argparse.ArgumentParser.parse_args = original
    raise AssertionError("train.main did not build a parser")


def test_trainer_defaults_unchanged():
    # A training command keeps its meaning (and --resume keeps working) only
    # if these defaults do not change. New options must default to the old
    # behaviour.
    defaults = {a.dest: a.default for a in _trainer_parser()._actions if a.dest != "help"}
    expected = {"cache": None, "hidden": 256, "buckets": 1, "king_buckets": "none", "activation": "crelu",
                "epochs": 30, "batch": 16384, "lr": 1e-3, "lam": 0.75, "val": 0.01, "threads": 0, "device": "auto",
                "seed": 1, "resume": False, "stop_after": 0}
    for k, v in expected.items():
        assert defaults.get(k) == v, f"--{k} default changed: {defaults.get(k)!r} (was {v!r})"


def test_run_round_designs_unchanged():
    frozen = {
        "current": ["--buckets", "8"],
        "kb4": ["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"],
        "kb8": ["--buckets", "8", "--king-buckets", "8", "--activation", "screlu"],
        "kb16": ["--buckets", "8", "--king-buckets", "16", "--activation", "screlu"],
    }
    for name, args in frozen.items():
        assert run_round.DESIGNS.get(name) == args, f"run_round.py design {name} changed"


def test_records_decode_as_written():
    raw = np.fromfile(FIXTURES / "train_data.bin", np.uint8).reshape(-1, 32)
    feats, stm, score, result = train.decode(torch.from_numpy(raw))
    assert len(raw) == 320 and set(stm.tolist()) <= {0, 1} and set(result.tolist()) <= {0.0, 0.5, 1.0}
    assert int((feats != train.PAD).sum(dim=1).min()) >= 2


def test_resumes_a_checkpoint_from_the_frozen_trainer(tmp_path, capsys):
    data = tmp_path / "train_data.bin"
    shutil.copyfile(FIXTURES / "train_data.bin", data)
    out = tmp_path / "train_partial.nnue"
    shutil.copyfile(FIXTURES / "train_partial.nnue.ckpt", str(out) + ".ckpt")
    train.main(["--data", str(data), "--out", str(out), "--epochs", "3", "--hidden", "8", "--batch", "64",
                "--threads", "1", "--buckets", "8", "--device", "cpu", "--resume"])
    assert "resuming after epoch 1" in capsys.readouterr().out
    ft_w, _, out_w, _, info = train.read_net(str(out))
    assert ft_w.shape == (768, 8) and out_w.shape == (8, 16) and info["king_buckets"] == 1
    assert torch.load(str(out) + ".ckpt", weights_only=False)["epoch"] == 3


def test_loads_and_evaluates_a_gnn_v2_checkpoint():
    from chess_graph import ChessGraph
    from gnn_evaluator import legal_move_priors, load_checkpoint

    model = load_checkpoint(str(FIXTURES / "gnn_v2.pt"), device=torch.device("cpu"))
    builder = ChessGraph(device=torch.device("cpu"))
    for case in json.loads((FIXTURES / "gnn_v2_expected.json").read_text()):
        probs, value, _ = model.evaluate(builder.fen_to_graph(case["fen"]), mc_samples=1)
        moves = [m.uci() for m in chess.Board(case["fen"]).legal_moves]
        assert sorted(moves) == sorted(case["moves"])
        assert np.allclose(legal_move_priors(probs[0], case["moves"]), case["priors"], atol=1e-5)
        assert abs(float(value[0]) - case["value"]) < 1e-5


def test_rpc_server_serves_a_gnn_v2_checkpoint():
    import rpc_server
    from chess_graph import ChessGraph
    from gnn_evaluator import load_checkpoint

    model = load_checkpoint(str(FIXTURES / "gnn_v2.pt"), device=torch.device("cpu"))
    service = rpc_server.EvaluationService(model, ChessGraph(device=torch.device("cpu")), mc_samples=1)
    cases = json.loads((FIXTURES / "gnn_v2_expected.json").read_text())
    reply = service.handle({"fens": [c["fen"] for c in cases], "moves": [c["moves"] for c in cases]})
    assert reply["policy_dim"] == 4096
    for case, r in zip(cases, reply["results"]):
        assert np.allclose(r["move_priors"], case["priors"], atol=1e-5)
        assert abs(r["value"] - case["value"]) < 1e-5
