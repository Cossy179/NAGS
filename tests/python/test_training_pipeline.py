"""Pipeline unit tests that run quickly on tiny data (no engine needed),
plus a self-play smoke test when a built engine is available."""

import json
import math
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch

import training_pipeline as tp

ROOT = Path(__file__).resolve().parents[2]


def tiny_config(tmp_path, **extra):
    cfg = {"data_dir": str(tmp_path / "data"), "model_dir": str(tmp_path / "models"), "logs_dir": str(tmp_path / "logs"),
           "pgn_file": str(tmp_path / "games.pgn"), "max_positions": 200, "batch_size": 16, "epochs": 1,
           "skip_opening_plies": 0, "model_params": {"hidden_dim": 16, "gnn_layers": 1, "policy_layers": 1, "value_layers": 1},
           "ppo_params": {"epochs": 1}, "meta_params": {"train_steps": 5}}
    cfg.update(extra)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def write_pgn(path, games=6):
    import random
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


def test_result_value_is_side_to_move_relative():
    assert tp.result_value("1-0", True) == 1.0
    assert tp.result_value("1-0", False) == -1.0
    assert tp.result_value("0-1", True) == -1.0
    assert tp.result_value("1/2-1/2", False) == 0.0
    assert tp.result_value("*", True) is None


def test_elo_and_time_control_helpers():
    elo, margin = tp.elo_from_score(0.5, 100)
    assert abs(elo) < 1e-6 and margin > 0
    assert tp.elo_from_score(0.75, 100)[0] == pytest.approx(-400 * math.log10(1 / 0.75 - 1))
    assert math.isfinite(tp.elo_from_score(1.0, 10)[0])  # clamped, no division by zero
    assert tp.parse_time_control("1+0.1") == (60.0, 0.1)
    assert tp.parse_time_control("5") == (300.0, 0.0)


def test_lfs_pointer_detection(tmp_path):
    p = tmp_path / "x.pgn"
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    assert tp.is_lfs_pointer(p)
    p.write_text('[Event "x"]\n')
    assert not tp.is_lfs_pointer(p)


def test_meta_sample_parsing():
    rec = {"fen": chess.STARTING_FEN, "info_string": "nags_meta time_left 30000 uncertainty 0.1 tactical 0.25 deltas 0.5,-0.25,0 meta on"}
    s = tp.TrainingPipeline._meta_sample(rec, -1.0)
    assert s["time_left_ms"] == 30000 and s["tactical_shot_ratio"] == 0.25 and s["reward"] == -1.0
    assert s["chosen_deltas"] == {"dfs_depth_delta": 0.5, "mcts_budget_delta": -0.25, "bandit_exploration_delta": 0.0}
    assert tp.TrainingPipeline._meta_sample({"fen": "x", "info_string": "something else"}, 0.0) is None


def test_parse_supervised_ppo_meta(tmp_path):
    torch.manual_seed(0)
    write_pgn(tmp_path / "games.pgn")
    pipe = tp.TrainingPipeline(tiny_config(tmp_path))
    dataset = pipe.parse_pgn_to_dataset()
    rows = tp.read_jsonl(dataset)
    assert rows and all(r["value"] in (-1.0, 0.0, 1.0) for r in rows)
    # values alternate sign with the side to move within a decisive game
    first_game = [r for r in rows if r["game_id"] == 1]
    assert first_game[0]["value"] == -first_game[1]["value"]

    model_path = pipe.supervised_training(dataset)
    assert model_path.exists()

    self_play = tmp_path / "data" / "self_play_test.jsonl"
    tp.write_jsonl(self_play, rows[:40])
    ppo_path = pipe.ppo_training(self_play, model_path)
    assert ppo_path.exists() and ppo_path != model_path

    meta_samples = tmp_path / "data" / "meta_samples_test.jsonl"
    tp.write_jsonl(meta_samples, [{"fen": r["fen"], "time_left_ms": 1000, "last_uncertainty": 0.1,
                                   "tactical_shot_ratio": 0.2, "reward": r["value"],
                                   "chosen_deltas": {"dfs_depth_delta": 0.1, "mcts_budget_delta": 0.0,
                                                     "bandit_exploration_delta": -0.1}} for r in rows[:20]])
    assert pipe.update_meta_learner(meta_samples) is not None
    assert pipe.meta_model_path.exists()


def test_missing_pgn_is_an_error(tmp_path):
    pipe = tp.TrainingPipeline(tiny_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        pipe.parse_pgn_to_dataset()


def _engine():
    for c in (ROOT / "build" / "nags", ROOT / "build" / "Release" / "nags.exe"):
        if c.exists():
            return c
    return None


@pytest.mark.skipif(_engine() is None, reason="engine not built")
def test_self_play_smoke(tmp_path):
    cfg = tiny_config(tmp_path, engine_path=str(_engine()), self_play_time=0.05, max_game_plies=16, rpc_mc_samples=1)
    pipe = tp.TrainingPipeline(cfg)
    out = pipe.run_self_play(1)
    rows = tp.read_jsonl(out)
    assert rows and all(chess.Move.from_uci(r["move"]) in chess.Board(r["fen"]).legal_moves for r in rows)
    meta = pipe.latest("meta_samples_*.jsonl", pipe.data_dir)
    assert meta is not None and tp.read_jsonl(meta)
