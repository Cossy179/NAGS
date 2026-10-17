#!/usr/bin/env python3
"""
NAGS Training Pipeline
======================

  parse       PGN games -> (position, move played, game result) samples
  supervised  train the GNN policy/value network on those samples
  selfplay    play real engine-vs-engine games with the current network
  ppo         clipped PPO update of the network on the self-play games
  evaluate    play a match against a baseline, estimate Elo, promote if better
  meta        train the meta-learner on (features, deltas, game result) samples
  full        all of the above in order

  label       v3: score positions from the jsonl datasets with the alpha-beta
              engine (MultiPV k, fixed nodes) -> data/distill_*.jsonl
  distill     v3: train the relational 'nags-gnn-v3' network on those labels
              (search-policy distillation; see distillation.py)

'label' and 'distill' are new and separate: the steps above, and 'full',
are unchanged and train 'nags-gnn-v2' networks. For v3 networks 'distill' is
the recommended objective (see ppo_training for why PPO on these games is not
an on-policy update).

Values are always from the side to move's point of view in [-1, 1]
(+1 win, 0 draw, -1 loss), matching the network's tanh value head and the
negamax convention of the engine.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn
import chess.polyglot
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from torch_geometric.loader import DataLoader
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch_geometric is required.") from e

from chess_graph import FEATURE_DIM, ChessGraph
from gnn_evaluator import CHECKPOINT_FORMAT, POLICY_DIM, GNNEvaluator, load_checkpoint, move_index, save_checkpoint
from meta_learner import DELTA_KEYS, MetaLearner

logger = logging.getLogger("nags.training")

# The v3 GNN modules (distillation, gnn_v3, relational_graph, policy_encoding)
# are imported only by the steps that use them ('label', 'distill'), so the
# legacy steps and their services never depend on the v3 code.

ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "data_dir": "data",
    "model_dir": "models",
    "logs_dir": "logs",
    "pgn_file": "AJ-CORR-PGN-000.pgn",
    "engine_path": "",              # empty: auto-detect build/nags or build/Release/nags.exe
    "max_positions": 100000,
    "skip_opening_plies": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10,
    "validation_fraction": 0.05,
    "self_play_games": 100,
    "self_play_time": 0.5,          # seconds per move
    "max_game_plies": 300,          # adjudicated as a draw beyond this
    "opening_random_plies": 4,
    "meta_exploration": 0.3,        # std-dev of noise the engine adds to meta deltas in self-play
    "rpc_mc_samples": 3,
    "elo_threshold": 25,
    "baseline_engine": "heuristic", # "heuristic" (nags without the network), "production", or a UCI engine command
    "seed": 0,
    "model_params": {"hidden_dim": 128, "gnn_layers": 6, "policy_layers": 4, "value_layers": 2},
    "ppo_params": {"epochs": 5, "clip_ratio": 0.2, "value_coef": 0.5, "entropy_coef": 0.01, "learning_rate": 0.0001},
    "meta_params": {"train_steps": 200},
    "evaluation": {"num_games": 50, "time_control": "1+0.1", "opening_book": "book.bin"},
    "notifications": {"slack_webhook": ""},
}

# Defaults of the v3 steps 'label' and 'distill'. They are merged under the
# 'distill_params' key of the loaded config (deep_merge), so a config file
# only needs the values it changes; DEFAULT_CONFIG (and so a newly written
# training_config.json) does not contain them.
DISTILL_DEFAULTS: Dict[str, Any] = {
    # label
    "label_sources": [],          # jsonl files / globs with a 'fen' field; empty: data/training_data.jsonl
                                  # and data/self_play_*.jsonl
    "label_positions": 20000,     # at most this many distinct positions per run (seeded random subset)
    "multipv": 5,                 # k: moves scored exactly per position
    "nodes": 50000,               # node budget per position
    "engine_path": "",            # empty: config 'engine_path', else build/nags_enhanced
    "engine_command": None,       # full command (string or list) overriding engine_path
    "hash_mb": 16,
    "clear_hash": True,           # clear the hash before each position (reproducible labels)
    # targets (see distillation.py)
    "temperature_cp": 50.0,       # policy target = softmax(cp / T) over the scored moves
    "rest_mass_scale": 0.02,      # unscored moves: this fraction of their upper-bound softmax mass (0 = none)
    "cp_clip": 2000.0,
    "mate_cp": 3000.0,            # mate in n -> mate_cp - mate_step * (n - 1)
    "mate_step": 10.0,
    "wdl_a": 400.0,               # P(win) = sigmoid((cp - a) / b), P(loss) = sigmoid((-cp - a) / b)
    "wdl_b": 250.0,
    "result_weight": 0.0,         # blend of the game result into the WDL target, when known
    # training
    "max_rows": 1000000,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "validation_fraction": 0.05,  # split by source game (by position when there is no game id)
    "value_weight": 1.0,
    "aux_score_weight": 0.25,
    "model": {"hidden_dim": 128, "layers": 6, "num_bases": 4, "ffn_mult": 2, "dropout": 0.0,
              "value_dropout": 0.2, "aux_score": True},
}


# ----------------------------------------------------------------------------
# Helpers

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def result_value(result: str, white_to_move: bool) -> Optional[float]:
    """Game result from the side to move's point of view, or None if unknown."""
    white = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result)
    if white is None:
        return None
    return white if white_to_move else -white


def is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def legal_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(POLICY_DIM, dtype=torch.bool)
    for mv in board.legal_moves:
        mask[mv.from_square * 64 + mv.to_square] = True
    return mask


def parse_time_control(tc: str) -> Tuple[float, float]:
    """'1+0.1' -> (60.0 s base, 0.1 s increment); base is in minutes."""
    base, _, inc = str(tc).partition("+")
    return float(base) * 60.0, float(inc or 0.0)


def elo_from_score(score: float, games: int) -> Tuple[float, float]:
    """Elo difference and an approximate 95% half-width."""
    eps = 0.5 / max(1, games)
    s = min(max(score, eps), 1.0 - eps)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(s * (1 - s) / max(1, games))
    slope = 400.0 / (math.log(10) * s * (1 - s))  # d elo / d score
    return elo, 1.96 * se * slope


class ServiceProcess:
    """Runs rpc_server.py / meta_learner.py as a subprocess on a free port."""

    def __init__(self, script: str, args: Sequence[str], log_path: Path, startup_timeout: float = 180.0):
        self.port = free_port()
        self.log_path = log_path
        cmd = [sys.executable, str(ROOT / script), "--port", str(self.port), *args]
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{script} exited during startup; see {log_path}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.25)
        self.close()
        raise RuntimeError(f"{script} did not start within {startup_timeout:.0f}s; see {log_path}")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def __enter__(self) -> "ServiceProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class MatchResult:
    games: int
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)


# ----------------------------------------------------------------------------
# Datasets

class PositionDataset(Dataset):
    """Samples {fen, move, value[, old_logp, advantage]} -> PyG Data objects."""

    def __init__(self, samples: List[Dict[str, Any]], with_mask: bool = False):
        self.samples = samples
        self.with_mask = with_mask
        self.builder = ChessGraph(device=torch.device("cpu"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        data = self.builder.fen_to_graph(s["fen"])
        data.y_policy = torch.tensor([move_index(s["move"])], dtype=torch.long)
        data.y_value = torch.tensor([float(s["value"])], dtype=torch.float)
        if self.with_mask:
            data.legal = legal_mask(chess.Board(s["fen"])).unsqueeze(0)
            data.old_logp = torch.tensor([float(s.get("old_logp", 0.0))])
            data.advantage = torch.tensor([float(s.get("advantage", 0.0))])
        return data


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out


def read_jsonl_settled(path: Path) -> List[Dict[str, Any]]:
    """read_jsonl for a file another process may still be writing (the legacy
    steps write data/*.jsonl line by line): a last line that does not parse
    (cut off mid-write) is skipped."""
    out = []
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            logger.info("%s: skipped an incomplete last line", path)
    return out


CHECKPOINT_FORMAT_V3 = "nags-gnn-v3"  # gnn_v3.CHECKPOINT_FORMAT_V3; not imported: legacy steps must not need gnn_v3


def checkpoint_format(path: Path) -> str:
    """The 'format' string of a GNN checkpoint ('nags-gnn-v2' when it has none)."""
    with open(path, "rb") as fh:
        payload = torch.load(fh, map_location="cpu", weights_only=False)
    fmt = payload.get("format") if isinstance(payload, dict) else None
    return fmt if isinstance(fmt, str) else CHECKPOINT_FORMAT


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------------
# Pipeline

class TrainingPipeline:
    def __init__(self, config_file: str = "training_config.json"):
        self.config = self.load_config(config_file)
        self.config["distill_params"] = deep_merge(DISTILL_DEFAULTS, self.config.get("distill_params") or {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = random.Random(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        self.data_dir = self._path(self.config["data_dir"])
        self.model_dir = self._path(self.config["model_dir"])
        self.logs_dir = self._path(self.config["logs_dir"])
        for d in (self.data_dir, self.model_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.production_path = self.model_dir / "production_model.pth"
        # The promoted 'nags-gnn-v3' network: a name the legacy '*_model_*.pth'
        # lookups (current_model, the 'evaluate' default) never match.
        self.production_v3_path = self.model_dir / "production_v3.pth"
        self.meta_model_path = self.model_dir / "meta_model.pth"
        self.meta_data_path = self.data_dir / "meta_training_data.json"
        logger.info("Using device: %s", self.device)

    # -- config / paths -------------------------------------------------------
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        path = Path(config_file)
        if path.exists():
            with open(path) as f:
                return deep_merge(DEFAULT_CONFIG, json.load(f))
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logger.info("Created default config: %s", path)
        return copy.deepcopy(DEFAULT_CONFIG)

    def _path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    def stamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def engine_path(self) -> Path:
        configured = self.config.get("engine_path")
        candidates = [self._path(configured)] if configured else [
            ROOT / "build" / "nags", ROOT / "build" / "Release" / "nags.exe", ROOT / "build" / "Release" / "nags",
            ROOT / "build" / "Debug" / "nags.exe", ROOT / "build" / "nags.exe"]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError("NAGS engine binary not found (looked at: " + ", ".join(map(str, candidates)) +
                                "). Build it with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release")

    def latest(self, pattern: str, directory: Optional[Path] = None) -> Optional[Path]:
        files = list((directory or self.model_dir).glob(pattern))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def current_model(self) -> Optional[Path]:
        """Production model if there is one, else the newest checkpoint."""
        if self.production_path.exists():
            return self.production_path
        return self.latest("*_model_*.pth")

    def new_model(self) -> GNNEvaluator:
        p = self.config["model_params"]
        return GNNEvaluator(in_dim=FEATURE_DIM, hidden_dim=p["hidden_dim"], gnn_layers=p["gnn_layers"],
                            policy_layers=p["policy_layers"], value_layers=p["value_layers"], device=self.device)

    # -- step 1: PGN -> samples ----------------------------------------------
    def parse_pgn_to_dataset(self) -> Path:
        pgn_path = self._path(self.config["pgn_file"])
        output_path = self.data_dir / "training_data.jsonl"
        if not pgn_path.exists():
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")
        if is_lfs_pointer(pgn_path):
            raise RuntimeError(f"{pgn_path} is a Git LFS pointer, not the actual PGN. Run 'git lfs pull' first.")

        max_positions = int(self.config["max_positions"])
        skip = int(self.config["skip_opening_plies"])
        logger.info("Parsing PGN file: %s", pgn_path)
        positions = games = skipped = 0
        with open(pgn_path, encoding="utf-8", errors="replace") as pgn, open(output_path, "w") as out:
            while positions < max_positions:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                games += 1
                result = game.headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2") or game.errors:
                    skipped += 1
                    continue
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if positions >= max_positions:
                        break
                    if ply >= skip:
                        out.write(json.dumps({"fen": board.fen(), "move": move.uci(),
                                              "value": result_value(result, board.turn == chess.WHITE),
                                              "game_id": games, "ply": ply}) + "\n")
                        positions += 1
                    board.push(move)
                if games % 1000 == 0:
                    logger.info("Processed %d games, extracted %d positions", games, positions)
        if positions == 0:
            raise RuntimeError(f"No usable positions found in {pgn_path} ({games} games read, {skipped} skipped)")
        logger.info("Extraction complete: %d positions from %d games (%d skipped)", positions, games, skipped)
        return output_path

    # -- step 2: supervised training --------------------------------------------
    def supervised_training(self, dataset_path: Path) -> Path:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path} (run the 'parse' step first)")
        samples = read_jsonl(dataset_path, limit=int(self.config["max_positions"]))
        if len(samples) < 2:
            raise RuntimeError(f"Dataset {dataset_path} has too few samples ({len(samples)})")
        # Split by game so positions from one game never leak into validation.
        game_ids = sorted({s.get("game_id", i) for i, s in enumerate(samples)})
        self.rng.shuffle(game_ids)
        n_val = max(1, int(len(game_ids) * float(self.config["validation_fraction"]))) if len(game_ids) > 1 else 0
        val_games = set(game_ids[:n_val])
        train = [s for i, s in enumerate(samples) if s.get("game_id", i) not in val_games]
        val = [s for i, s in enumerate(samples) if s.get("game_id", i) in val_games]
        logger.info("Supervised training on %d positions (%d validation)", len(train), len(val))
        # Opt-in (default off, which is the behaviour this step always had):
        # restrict the policy softmax to the legal moves.
        masked = bool(self.config.get("supervised_legal_mask", False))
        if masked:
            logger.info("supervised_legal_mask: policy loss over legal moves only")

        model = self.new_model()
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]), weight_decay=1e-4)
        bs = int(self.config["batch_size"])
        train_loader = DataLoader(PositionDataset(train, with_mask=masked), batch_size=bs, shuffle=True)
        val_loader = DataLoader(PositionDataset(val, with_mask=masked), batch_size=bs) if val else None

        best_val = float("inf")
        best_path = self.model_dir / f"supervised_model_{self.stamp()}.pth"
        for epoch in range(int(self.config["epochs"])):
            model.train()
            totals = [0.0, 0.0, 0]
            for step, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                logits, values = model(batch)
                if masked:
                    logits = logits.masked_fill(~batch.legal, -1e9)
                p_loss = F.cross_entropy(logits, batch.y_policy)
                v_loss = F.mse_loss(values, batch.y_value)
                loss = p_loss + v_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                totals[0] += p_loss.item() * batch.num_graphs
                totals[1] += v_loss.item() * batch.num_graphs
                totals[2] += batch.num_graphs
                if step % 100 == 0:
                    logger.info("Epoch %d step %d: policy %.4f value %.4f", epoch, step, p_loss.item(), v_loss.item())
            msg = f"Epoch {epoch}: train policy {totals[0] / totals[2]:.4f} value {totals[1] / totals[2]:.4f}"
            if val_loader is not None:
                vp, vv, acc = self._validate(model, val_loader, masked)
                msg += f" | val policy {vp:.4f} value {vv:.4f} top-1 {acc:.1%}"
                if vp + vv < best_val:
                    best_val = vp + vv
                    save_checkpoint(model, str(best_path), {"stage": "supervised", "epoch": epoch})
            else:
                save_checkpoint(model, str(best_path), {"stage": "supervised", "epoch": epoch})
            logger.info(msg)
        logger.info("Supervised model saved: %s", best_path)
        return best_path

    @torch.no_grad()
    def _validate(self, model: GNNEvaluator, loader: DataLoader, masked: bool = False) -> Tuple[float, float, float]:
        model.eval()
        p_sum = v_sum = correct = n = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            logits, values = model(batch)
            if masked:
                logits = logits.masked_fill(~batch.legal, -1e9)
            p_sum += F.cross_entropy(logits, batch.y_policy, reduction="sum").item()
            v_sum += F.mse_loss(values, batch.y_value, reduction="sum").item()
            correct += (logits.argmax(dim=1) == batch.y_policy).sum().item()
            n += batch.num_graphs
        return p_sum / n, v_sum / n, correct / n

    # -- engines and games ------------------------------------------------------
    def open_nags(self, nn_port: Optional[int], meta_port: Optional[int], meta_exploration: float = 0.0):
        engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path()))
        # MctsMinTime 0: run the MCTS arm even at the short self-play move times.
        options = {"UseNN": nn_port is not None, "UseMetaLearner": meta_port is not None,
                   "MetaExploration": int(round(meta_exploration * 100)), "MctsMinTime": 0}
        if nn_port is not None:
            options["NNPort"] = nn_port
        if meta_port is not None:
            options["MetaPort"] = meta_port
        engine.configure(options)
        return engine

    def opening(self, book: Optional[Path]) -> List[chess.Move]:
        """A few opening plies from the polyglot book if available, otherwise random legal moves."""
        board = chess.Board()
        moves: List[chess.Move] = []
        plies = int(self.config["opening_random_plies"])
        reader = chess.polyglot.open_reader(str(book)) if book and book.exists() else None
        try:
            for _ in range(plies):
                choice = None
                if reader is not None:
                    try:
                        choice = reader.weighted_choice(board, random=self.rng).move
                    except IndexError:
                        choice = None
                if choice is None:
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    choice = self.rng.choice(legal)
                board.push(choice)
                moves.append(choice)
        finally:
            if reader is not None:
                reader.close()
        return moves

    def play_game(self, white, black, opening: Sequence[chess.Move], move_time: Optional[float] = None,
                  clock: Optional[Tuple[float, float]] = None) -> Tuple[str, List[Dict[str, Any]], chess.Board]:
        """Plays one game. Returns (result, per-move records, final board)."""
        board = chess.Board()
        for mv in opening:
            board.push(mv)
        records: List[Dict[str, Any]] = []
        remaining = {chess.WHITE: clock[0], chess.BLACK: clock[0]} if clock else None
        max_plies = int(self.config["max_game_plies"])
        while not board.is_game_over(claim_draw=True):
            if board.ply() >= max_plies:
                return "1/2-1/2", records, board
            engine = white if board.turn == chess.WHITE else black
            if remaining is not None:
                limit = chess.engine.Limit(white_clock=remaining[chess.WHITE], black_clock=remaining[chess.BLACK],
                                           white_inc=clock[1], black_inc=clock[1])
            else:
                limit = chess.engine.Limit(time=move_time)
            start = time.monotonic()
            result = engine.play(board, limit, info=chess.engine.INFO_ALL)
            elapsed = time.monotonic() - start
            if remaining is not None:
                remaining[board.turn] -= elapsed
                if remaining[board.turn] < 0:
                    return ("0-1" if board.turn == chess.WHITE else "1-0"), records, board
                remaining[board.turn] += clock[1]
            if result.move is None or result.move not in board.legal_moves:
                logger.warning("Engine returned illegal move %s in %s", result.move, board.fen())
                return ("0-1" if board.turn == chess.WHITE else "1-0"), records, board
            records.append({"fen": board.fen(), "move": result.move.uci(), "white": board.turn == chess.WHITE,
                            "info_string": result.info.get("string", "")})
            board.push(result.move)
        return board.result(claim_draw=True), records, board

    # -- step 3: self-play --------------------------------------------------------
    def run_self_play(self, num_games: int, model_path: Optional[Path] = None) -> Path:
        model_path = model_path or self.current_model()
        if model_path is None:
            logger.warning("No trained model found: self-play will use an untrained network")
        stamp = self.stamp()
        out_path = self.data_dir / f"self_play_{stamp}.jsonl"
        meta_path = self.data_dir / f"meta_samples_{stamp}.jsonl"
        move_time = float(self.config["self_play_time"])
        rpc_args = ["--mc-samples", str(self.config["rpc_mc_samples"]),
                    "--hidden-dim", str(self.config["model_params"]["hidden_dim"])]
        rpc_args += ["--model", str(model_path)] if model_path else ["--model", ""]
        meta_args = ["--model", str(self.meta_model_path), "--data", ""]
        positions = meta_samples = 0
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        logger.info("Starting self-play: %d games at %.2fs per move (model: %s)", num_games, move_time, model_path)
        with ServiceProcess("rpc_server.py", rpc_args, self.logs_dir / "selfplay_rpc_server.log") as rpc, \
                ServiceProcess("meta_learner.py", meta_args, self.logs_dir / "selfplay_meta_learner.log") as meta, \
                open(out_path, "w") as out, open(meta_path, "w") as meta_out:
            engines = [self.open_nags(rpc.port, meta.port, float(self.config["meta_exploration"])) for _ in range(2)]
            try:
                for game_id in range(num_games):
                    for e in engines:
                        e.configure({"Clear Hash": None})
                    white, black = (engines[0], engines[1]) if game_id % 2 == 0 else (engines[1], engines[0])
                    result, records, _ = self.play_game(white, black, self.opening(None), move_time=move_time)
                    results[result] = results.get(result, 0) + 1
                    for rec in records:
                        value = result_value(result, rec["white"])
                        out.write(json.dumps({"fen": rec["fen"], "move": rec["move"], "value": value,
                                              "game_id": game_id}) + "\n")
                        positions += 1
                        sample = self._meta_sample(rec, value)
                        if sample:
                            meta_out.write(json.dumps(sample) + "\n")
                            meta_samples += 1
                    logger.info("Self-play game %d/%d: %s (%d plies)", game_id + 1, num_games, result, len(records))
            finally:
                for e in engines:
                    e.quit()
        logger.info("Self-play complete: %s; %d positions -> %s; %d meta samples -> %s",
                    results, positions, out_path, meta_samples, meta_path)
        return out_path

    @staticmethod
    def _meta_sample(record: Dict[str, Any], reward: float) -> Optional[Dict[str, Any]]:
        """Parses 'nags_meta time_left T uncertainty U tactical X deltas a,b,c meta on'."""
        parts = record.get("info_string", "").split()
        if not parts or parts[0] != "nags_meta":
            return None
        try:
            kv = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
            deltas = [float(v) for v in kv["deltas"].split(",")]
            return {"fen": record["fen"], "time_left_ms": float(kv["time_left"]),
                    "last_uncertainty": float(kv["uncertainty"]), "tactical_shot_ratio": float(kv["tactical"]),
                    "chosen_deltas": dict(zip(DELTA_KEYS, deltas)), "reward": reward}
        except (KeyError, ValueError):
            return None

    # -- step 4: PPO -----------------------------------------------------------
    def ppo_training(self, self_play_path: Path, model_path: Optional[Path] = None) -> Path:
        """Clipped PPO-style update of a 'nags-gnn-v2' network on self-play games (unchanged).

        Caveat: this is not a valid on-policy (PPO) update. The self-play moves
        are chosen by the engine's alpha-beta search (with the network only
        guiding the MCTS arm), not sampled from the network's policy, so the
        behaviour policy is not the network: the ratio exp(logp - old_logp)
        is not an importance ratio against the policy that produced the data,
        and the clipped surrogate does not estimate the policy gradient. In
        effect it is advantage-weighted imitation of the engine's moves. Kept
        as it was for the running training; for the v3 network the recommended
        objective is search-policy distillation (steps 'label' + 'distill').
        """
        logger.info("ppo: the self-play moves come from the alpha-beta search, not from the network's "
                    "policy, so this is not an on-policy PPO update; 'distill' is the recommended objective "
                    "(see ppo_training's docstring)")
        if not self_play_path.exists():
            raise FileNotFoundError(f"Self-play data not found: {self_play_path}")
        samples = read_jsonl(self_play_path)
        if not samples:
            raise RuntimeError(f"No self-play positions in {self_play_path}")
        model_path = model_path or self.current_model()
        model = load_checkpoint(str(model_path), device=self.device) if model_path else self.new_model()
        params = self.config["ppo_params"]
        bs = int(self.config["batch_size"])
        logger.info("PPO on %d self-play positions starting from %s", len(samples), model_path or "a fresh model")

        # Old policy: log-probability of the played move (over legal moves) and value estimate.
        model.eval()
        old_logp: List[float] = []
        old_value: List[float] = []
        with torch.no_grad():
            for batch in DataLoader(PositionDataset(samples, with_mask=True), batch_size=bs):
                batch = batch.to(self.device)
                logits, values = model(batch)
                logp = F.log_softmax(logits.masked_fill(~batch.legal, -1e9), dim=1)
                old_logp.extend(logp.gather(1, batch.y_policy.unsqueeze(1)).squeeze(1).tolist())
                old_value.extend(values.tolist())
        adv = torch.tensor([s["value"] for s in samples]) - torch.tensor(old_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(samples) > 1 else adv
        for s, lp, a in zip(samples, old_logp, adv.tolist()):
            s["old_logp"], s["advantage"] = lp, a

        opt = torch.optim.AdamW(model.parameters(), lr=float(params.get("learning_rate", 1e-4)))
        clip = float(params["clip_ratio"])
        loader = DataLoader(PositionDataset(samples, with_mask=True), batch_size=bs, shuffle=True)
        for epoch in range(int(params["epochs"])):
            model.train()
            stats = [0.0, 0.0, 0.0, 0]
            for batch in loader:
                batch = batch.to(self.device)
                logits, values = model(batch)
                logp_all = F.log_softmax(logits.masked_fill(~batch.legal, -1e9), dim=1)
                logp = logp_all.gather(1, batch.y_policy.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp - batch.old_logp)
                surrogate = torch.min(ratio * batch.advantage, ratio.clamp(1 - clip, 1 + clip) * batch.advantage)
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(values, batch.y_value)
                probs = logp_all.exp()
                entropy = -(probs * logp_all.masked_fill(~batch.legal, 0.0)).sum(dim=1).mean()
                loss = policy_loss + float(params["value_coef"]) * value_loss - float(params["entropy_coef"]) * entropy
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                stats[0] += policy_loss.item() * batch.num_graphs
                stats[1] += value_loss.item() * batch.num_graphs
                stats[2] += entropy.item() * batch.num_graphs
                stats[3] += batch.num_graphs
            n = stats[3]
            logger.info("PPO epoch %d: policy %.4f value %.4f entropy %.3f", epoch, stats[0] / n, stats[1] / n, stats[2] / n)

        out = self.model_dir / f"ppo_model_{self.stamp()}.pth"
        save_checkpoint(model, str(out), {"stage": "ppo", "parent": str(model_path) if model_path else None})
        logger.info("PPO model saved: %s", out)
        return out

    # -- step 5: evaluation -----------------------------------------------------
    def evaluate_against_baseline(self, model_path: Path) -> Optional[Tuple[float, float, MatchResult]]:
        ev = self.config["evaluation"]
        num_games = int(ev["num_games"])
        clock = parse_time_control(ev.get("time_control", "1+0.1"))
        book = self._path(ev["opening_book"]) if ev.get("opening_book") else None
        baseline = str(self.config["baseline_engine"])
        mc = ["--mc-samples", str(self.config["rpc_mc_samples"])]
        logger.info("Evaluating %s against baseline '%s': %d games at %s", model_path, baseline, num_games,
                    ev.get("time_control"))

        services: List[ServiceProcess] = []
        try:
            cand_rpc = ServiceProcess("rpc_server.py", ["--model", str(model_path), *mc], self.logs_dir / "eval_candidate_rpc.log")
            services.append(cand_rpc)
            candidate = self.open_nags(cand_rpc.port, None)
            # A v3 candidate plays the promoted v3 network once there is one
            # (else the v2 production, as any candidate does).
            production = self.production_path
            if self.production_v3_path.exists() and checkpoint_format(model_path) == CHECKPOINT_FORMAT_V3:
                production = self.production_v3_path
            if baseline == "production" and production.exists() and \
                    production.resolve() != Path(model_path).resolve():
                base_rpc = ServiceProcess("rpc_server.py", ["--model", str(production), *mc],
                                          self.logs_dir / "eval_baseline_rpc.log")
                services.append(base_rpc)
                opponent = self.open_nags(base_rpc.port, None)
            elif baseline in ("heuristic", "production"):
                if baseline == "production":
                    logger.info("No separate production model yet; using the heuristic engine as baseline")
                opponent = self.open_nags(None, None)
            else:
                cmd = shutil.which(baseline) or (baseline if Path(baseline).exists() else None)
                if cmd is None:
                    logger.error("Baseline engine '%s' not found; skipping evaluation", baseline)
                    candidate.quit()
                    return None
                opponent = chess.engine.SimpleEngine.popen_uci(cmd)
            try:
                wins = draws = losses = 0
                for g in range(num_games):
                    if g % 2 == 0:
                        opening = self.opening(book)  # each opening is played with both colours
                    cand_white = g % 2 == 0
                    for e in (candidate, opponent):
                        try:
                            e.configure({"Clear Hash": None})
                        except chess.engine.EngineError:
                            pass
                    white, black = (candidate, opponent) if cand_white else (opponent, candidate)
                    result, _, _ = self.play_game(white, black, opening, clock=clock)
                    value = result_value(result, cand_white)
                    wins += value == 1.0
                    draws += value == 0.0
                    losses += value == -1.0
                    logger.info("Eval game %d/%d: %s (candidate %s)", g + 1, num_games, result,
                                "white" if cand_white else "black")
            finally:
                candidate.quit()
                opponent.quit()
        finally:
            for s in services:
                s.close()

        match = MatchResult(num_games, int(wins), int(draws), int(losses))
        elo, margin = elo_from_score(match.score, match.games)
        logger.info("Evaluation: +%d =%d -%d, score %.1f%%, Elo %+.0f ± %.0f", match.wins, match.draws, match.losses,
                    100 * match.score, elo, margin)
        return elo, margin, match

    # -- promotion / notifications ------------------------------------------------
    def promote_model(self, model_path: Path, elo: float, margin: float) -> None:
        # Only a 'nags-gnn-v2' checkpoint may become production_model.pth: the
        # legacy steps (self-play, PPO) and older code load that file as v2. A
        # v3 network that passes goes to its own slot.
        target = self.production_path
        fmt = checkpoint_format(model_path)
        if fmt == CHECKPOINT_FORMAT_V3:
            target = self.production_v3_path
            logger.info("%s is a %s checkpoint: promoted to %s, not to %s", model_path, fmt, target,
                        self.production_path)
        shutil.copy2(model_path, target)
        record = {"timestamp": datetime.now().isoformat(), "model_path": str(model_path), "elo_gain": elo,
                  "elo_margin_95": margin, "promoted_to": str(target), "format": fmt}
        with open(self.logs_dir / "promotions.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Model promoted to %s", target)
        self.notify_team(f"New NAGS model promoted: {elo:+.0f} ± {margin:.0f} Elo vs baseline ({model_path.name})")

    def notify_team(self, message: str) -> None:
        logger.info("NOTIFICATION: %s", message)
        webhook = self.config.get("notifications", {}).get("slack_webhook")
        if not webhook:
            return
        try:
            req = urllib.request.Request(webhook, data=json.dumps({"text": message}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            logger.warning("Slack notification failed: %s", e)

    def evaluate_and_maybe_promote(self, model_path: Path) -> Optional[float]:
        outcome = self.evaluate_against_baseline(model_path)
        if outcome is None:
            return None
        elo, margin, _ = outcome
        threshold = float(self.config["elo_threshold"])
        if elo > threshold:
            self.promote_model(model_path, elo, margin)
        else:
            logger.info("Model not promoted (Elo %+.0f ± %.0f, threshold %+.0f)", elo, margin, threshold)
        return elo

    # -- step 6: meta-learner -----------------------------------------------------
    def update_meta_learner(self, meta_samples_path: Optional[Path] = None) -> Optional[float]:
        meta_samples_path = meta_samples_path or self.latest("meta_samples_*.jsonl", self.data_dir)
        meta = MetaLearner(model_path=str(self.meta_model_path), data_path=str(self.meta_data_path))
        meta.load_training_data()
        added = 0
        if meta_samples_path and meta_samples_path.exists():
            for s in read_jsonl(meta_samples_path):
                meta.add_training_sample(s["fen"], s["time_left_ms"], s["last_uncertainty"],
                                         s["tactical_shot_ratio"], s["chosen_deltas"], s["reward"])
                added += 1
        logger.info("Meta-learner: added %d samples from %s", added, meta_samples_path)
        loss = meta.train_epoch(int(self.config["meta_params"]["train_steps"]))
        meta.save_model()
        return loss

    # -- v3 step 'label': search scores for distillation --------------------------
    def label_sources(self, sources: Optional[Sequence[Path]] = None) -> List[Path]:
        if sources:
            return [Path(s) for s in sources]
        patterns = self.config["distill_params"].get("label_sources") or []
        if patterns:
            found: List[Path] = []
            for pattern in patterns:
                path = self._path(pattern)
                matches = sorted(path.parent.glob(path.name)) if any(c in path.name for c in "*?[") else [path]
                found.extend(p for p in matches if p not in found)
            return found
        default = [self.data_dir / "training_data.jsonl"] + sorted(self.data_dir.glob("self_play_*.jsonl"))
        return [p for p in default if p.exists()]

    def label_engine_command(self) -> Any:
        """The alpha-beta engine for labelling: distill_params engine_command / engine_path, the
        config's engine_path, else build/nags_enhanced (or nags, run without the network)."""
        params = self.config["distill_params"]
        if params.get("engine_command"):
            return params["engine_command"]
        if params.get("engine_path"):
            path = self._path(params["engine_path"])
            if not path.exists():
                raise FileNotFoundError(f"distill_params.engine_path not found: {path}")
            return str(path)
        if self.config.get("engine_path"):
            return str(self.engine_path())
        candidates = [ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe",
                      ROOT / "build" / "Release" / "nags_enhanced", ROOT / "build" / "nags_enhanced.exe",
                      ROOT / "build" / "nags", ROOT / "build" / "Release" / "nags.exe"]
        for c in candidates:
            if c.exists():
                return str(c)
        raise FileNotFoundError("No engine for labelling (looked at: " + ", ".join(map(str, candidates)) +
                                "). Build it with: cmake -B build -DCMAKE_BUILD_TYPE=Release && "
                                "cmake --build build --config Release")

    def open_label_engine(self):
        params = self.config["distill_params"]
        engine = chess.engine.SimpleEngine.popen_uci(self.label_engine_command())
        options: Dict[str, Any] = {}
        for name, value in (("Threads", 1), ("Hash", int(params["hash_mb"])), ("UseNN", False),
                            ("UseMetaLearner", False)):
            if name in engine.options:
                options[name] = value
        try:
            engine.configure(options)
        except Exception:
            engine.quit()
            raise
        return engine

    def label_positions(self, sources: Optional[Sequence[Path]] = None) -> Path:
        """Scores positions with the alpha-beta engine for 'distill' (row format: distillation.py).

        Reads 'fen' rows from jsonl files (parse / self-play outputs or any jsonl
        with a 'fen' field), keeps distinct positions, and writes the engine's
        MultiPV top-k with exact scores to data/distill_<stamp>.jsonl, flushing
        as it goes so an interrupted run keeps what it labelled.
        """
        from distillation import label_board, source_result
        params = self.config["distill_params"]
        paths = self.label_sources(sources)
        if not paths:
            raise FileNotFoundError(f"No positions to label (no data/training_data.jsonl or self_play_*.jsonl in "
                                    f"{self.data_dir}; pass --data or set distill_params.label_sources)")
        candidates: List[Tuple[Path, Dict[str, Any]]] = []
        seen = set()
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Label source not found: {path}")
            for row in read_jsonl_settled(path):
                fen = row.get("fen")
                if not isinstance(fen, str):
                    continue
                key = " ".join(fen.split()[:4])
                if key not in seen:
                    seen.add(key)
                    candidates.append((path, row))
        limit = int(params["label_positions"])
        if len(candidates) > limit:
            keep = sorted(self.rng.sample(range(len(candidates)), limit))
            candidates = [candidates[i] for i in keep]
        k, nodes = int(params["multipv"]), int(params["nodes"])
        out_path = self.data_dir / f"distill_{self.stamp()}.jsonl"
        logger.info("Labelling %d positions from %s: MultiPV %d at %d nodes -> %s", len(candidates),
                    ", ".join(map(str, paths)), k, nodes, out_path)
        written = skipped = 0
        start = time.time()
        engine = self.open_label_engine()
        clear = bool(params["clear_hash"]) and "Clear Hash" in engine.options
        try:
            with open(out_path, "w") as out:
                for path, row in candidates:
                    try:
                        board = chess.Board(row["fen"])
                    except ValueError:
                        skipped += 1
                        continue
                    if clear:
                        engine.configure({"Clear Hash": None})
                    rec = label_board(engine, board, k, nodes)
                    if rec is None:
                        skipped += 1
                        continue
                    rec["result"] = source_result(row, board)
                    rec["source"] = path.name
                    for key in ("game_id", "ply"):
                        if key in row:
                            rec[key] = row[key]
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    written += 1
                    if written % 100 == 0:
                        logger.info("Labelled %d/%d positions (%.1f/s)", written, len(candidates),
                                    written / max(1e-9, time.time() - start))
        except chess.engine.EngineError as e:
            logger.error("Engine failed after %d positions: %s", written, e)
        finally:
            try:
                engine.quit()
            except chess.engine.EngineError:
                pass
        if written == 0:
            raise RuntimeError(f"No position could be labelled ({skipped} skipped)")
        logger.info("Labelling complete: %d positions (%d skipped) -> %s", written, skipped, out_path)
        return out_path

    # -- v3 step 'distill': search-policy distillation ----------------------------
    def new_v3_model(self) -> "RelationalGNNEvaluator":
        from gnn_v3 import RelationalGNNEvaluator
        return RelationalGNNEvaluator(**self.config["distill_params"]["model"], device=self.device)

    def distill_training(self, dataset_path: Optional[Path] = None, model_path: Optional[Path] = None) -> Path:
        """Trains a 'nags-gnn-v3' network on 'label' rows; returns the best checkpoint.

        Policy loss: cross-entropy against the soft search target over the legal
        moves (log_softmax restricted to the legal mask). Value loss: WDL
        cross-entropy against the logistic WDL of the search score (optionally
        blended with the game result). Targets: distillation.py. Validation
        (split by source game) reports policy NLL, top-1 agreement with the
        search's best move, WDL NLL and value MSE.
        """
        from distillation import DistillDataset, distill_losses, valid_row
        from gnn_v3 import load_checkpoint_v3, save_checkpoint_v3
        params = self.config["distill_params"]
        dataset_path = dataset_path or self.latest("distill_*.jsonl", self.data_dir)
        if dataset_path is None or not Path(dataset_path).exists():
            raise FileNotFoundError(f"Distillation data not found: {dataset_path or self.data_dir / 'distill_*.jsonl'} "
                                    f"(run the 'label' step first)")
        rows = read_jsonl(Path(dataset_path), limit=int(params["max_rows"]))
        usable = [r for r in rows if valid_row(r)]
        if len(usable) < len(rows):
            logger.warning("Ignoring %d label rows without legal scored moves", len(rows) - len(usable))
        if len(usable) < 2:
            raise RuntimeError(f"{dataset_path} has too few usable rows ({len(usable)})")

        def group(i: int, r: Dict[str, Any]) -> str:
            if r.get("game_id") is not None:
                return f"{r.get('source', '')}#{r['game_id']}"
            return f"row#{i}"

        groups = sorted({group(i, r) for i, r in enumerate(usable)})
        self.rng.shuffle(groups)
        fraction = float(params["validation_fraction"])
        n_val = max(1, round(len(groups) * fraction)) if fraction > 0 and len(groups) > 1 else 0
        val_groups = set(groups[:n_val])
        train = [r for i, r in enumerate(usable) if group(i, r) not in val_groups]
        val = [r for i, r in enumerate(usable) if group(i, r) in val_groups]
        logger.info("Distillation on %d positions (%d validation, %d groups)", len(train), len(val), len(groups))

        if model_path:
            try:
                model = load_checkpoint_v3(str(model_path), device=self.device)
            except ValueError as e:
                raise RuntimeError(f"'distill' continues only from a nags-gnn-v3 checkpoint: {e}") from e
        else:
            model = self.new_v3_model()
        opt = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]),
                                weight_decay=float(params["weight_decay"]))
        bs = int(params["batch_size"])
        train_loader = DataLoader(DistillDataset(train, params), batch_size=bs, shuffle=True)
        val_loader = DataLoader(DistillDataset(val, params), batch_size=bs) if val else None
        value_w, aux_w = float(params["value_weight"]), float(params["aux_score_weight"])

        best_val = float("inf")
        best_path = self.model_dir / f"gnn_v3_distill_{self.stamp()}.pth"  # not matched by '*_model_*.pth'
        for epoch in range(int(params["epochs"])):
            model.train()
            sums: Dict[str, float] = {}
            n = 0
            for step, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                policy_logits, wdl_logits, score = model(batch)
                parts = distill_losses(policy_logits, wdl_logits, score, batch)
                loss = parts["policy_nll"].mean() + value_w * parts["wdl_nll"].mean()
                if "score_se" in parts and aux_w > 0:
                    loss = loss + aux_w * parts["score_se"].mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                for key, v in parts.items():
                    sums[key] = sums.get(key, 0.0) + float(v.detach().sum())
                n += batch.num_graphs
                if step % 100 == 0:
                    logger.info("Distill epoch %d step %d: policy %.4f wdl %.4f", epoch, step,
                                parts["policy_nll"].mean().item(), parts["wdl_nll"].mean().item())
            metrics: Dict[str, Any] = {"train": self._distill_summary(sums, n)}
            if val_loader is not None:
                metrics["val"] = self._distill_metrics(model, val_loader)
                score_val = metrics["val"]["policy_nll"] + value_w * metrics["val"]["wdl_nll"]
            else:
                score_val = -float(epoch)  # no validation: keep the latest epoch
            logger.info("Distill epoch %d: %s", epoch, json.dumps(metrics))
            if score_val < best_val:
                best_val = score_val
                save_checkpoint_v3(model, str(best_path), {"stage": "distill", "epoch": epoch, "metrics": metrics,
                                                           "dataset": str(dataset_path),
                                                           "parent": str(model_path) if model_path else None})
        logger.info("Distilled 'nags-gnn-v3' model saved: %s", best_path)
        return best_path

    @staticmethod
    def _distill_summary(sums: Dict[str, float], n: int) -> Dict[str, float]:
        n = max(1, n)
        out = {"positions": n, "policy_nll": sums.get("policy_nll", 0.0) / n, "top1": sums.get("top1", 0.0) / n,
               "wdl_nll": sums.get("wdl_nll", 0.0) / n, "value_mse": sums.get("value_se", 0.0) / n}
        if "score_se" in sums:
            out["score_mse"] = sums["score_se"] / n
        return out

    @torch.no_grad()
    def _distill_metrics(self, model: "RelationalGNNEvaluator", loader: DataLoader) -> Dict[str, float]:
        from distillation import distill_losses
        was_training = model.training
        model.eval()
        sums: Dict[str, float] = {}
        n = 0
        try:
            for batch in loader:
                batch = batch.to(self.device)
                parts = distill_losses(*model(batch), batch)
                for key, v in parts.items():
                    sums[key] = sums.get(key, 0.0) + float(v.sum())
                n += batch.num_graphs
        finally:
            model.train(was_training)
        return self._distill_summary(sums, n)

    # -- full pipeline ---------------------------------------------------------
    def run_full_pipeline(self) -> None:
        start = time.time()
        logger.info("Starting full training pipeline")
        dataset = self.parse_pgn_to_dataset()
        supervised = self.supervised_training(dataset)
        self_play = self.run_self_play(int(self.config["self_play_games"]), supervised)
        ppo = self.ppo_training(self_play, supervised)
        self.evaluate_and_maybe_promote(ppo)
        self.update_meta_learner()
        logger.info("Pipeline completed in %.1f seconds", time.time() - start)


def setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(logs_dir / "training.log"), logging.StreamHandler()])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NAGS Training Pipeline")
    parser.add_argument("--config", default="training_config.json", help="Config file path")
    parser.add_argument("--step", choices=["parse", "supervised", "selfplay", "ppo", "evaluate", "meta", "full",
                                           "label", "distill"],
                        default="full", help="Pipeline step to run ('label' and 'distill' train the v3 network)")
    parser.add_argument("--model", help="Checkpoint to use instead of the latest one")
    parser.add_argument("--games", type=int, help="Override the number of self-play / evaluation games")
    parser.add_argument("--data", nargs="+", help="label: jsonl files with 'fen' rows to label; "
                                                  "distill: the labelled jsonl (default: the newest data/distill_*.jsonl)")
    args = parser.parse_args(argv)

    config = TrainingPipeline.load_config(args.config)
    logs_dir = Path(config["logs_dir"])
    setup_logging(logs_dir if logs_dir.is_absolute() else ROOT / logs_dir)
    pipeline = TrainingPipeline(args.config)
    if args.games is not None:
        pipeline.config["self_play_games"] = args.games
        pipeline.config["evaluation"]["num_games"] = args.games
    model = Path(args.model) if args.model else None

    try:
        if args.step == "full":
            pipeline.run_full_pipeline()
        elif args.step == "parse":
            pipeline.parse_pgn_to_dataset()
        elif args.step == "supervised":
            pipeline.supervised_training(pipeline.data_dir / "training_data.jsonl")
        elif args.step == "selfplay":
            pipeline.run_self_play(int(pipeline.config["self_play_games"]), model)
        elif args.step == "ppo":
            data = pipeline.latest("self_play_*.jsonl", pipeline.data_dir)
            if data is None:
                raise FileNotFoundError("No self-play data found (run the 'selfplay' step first)")
            pipeline.ppo_training(data, model)
        elif args.step == "evaluate":
            target = model or pipeline.latest("*_model_*.pth")
            if target is None:
                raise FileNotFoundError("No model to evaluate (run 'supervised' or 'ppo' first)")
            pipeline.evaluate_and_maybe_promote(target)
        elif args.step == "meta":
            pipeline.update_meta_learner()
        elif args.step == "label":
            pipeline.label_positions([Path(p) for p in args.data] if args.data else None)
        elif args.step == "distill":
            pipeline.distill_training(Path(args.data[0]) if args.data else None, model)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
