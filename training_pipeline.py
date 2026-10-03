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
from gnn_evaluator import POLICY_DIM, GNNEvaluator, load_checkpoint, move_index, save_checkpoint
from meta_learner import DELTA_KEYS, MetaLearner

logger = logging.getLogger("nags.training")

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = random.Random(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        self.data_dir = self._path(self.config["data_dir"])
        self.model_dir = self._path(self.config["model_dir"])
        self.logs_dir = self._path(self.config["logs_dir"])
        for d in (self.data_dir, self.model_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.production_path = self.model_dir / "production_model.pth"
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

        model = self.new_model()
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]), weight_decay=1e-4)
        bs = int(self.config["batch_size"])
        train_loader = DataLoader(PositionDataset(train), batch_size=bs, shuffle=True)
        val_loader = DataLoader(PositionDataset(val), batch_size=bs) if val else None

        best_val = float("inf")
        best_path = self.model_dir / f"supervised_model_{self.stamp()}.pth"
        for epoch in range(int(self.config["epochs"])):
            model.train()
            totals = [0.0, 0.0, 0]
            for step, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                logits, values = model(batch)
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
                vp, vv, acc = self._validate(model, val_loader)
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
    def _validate(self, model: GNNEvaluator, loader: DataLoader) -> Tuple[float, float, float]:
        model.eval()
        p_sum = v_sum = correct = n = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            logits, values = model(batch)
            p_sum += F.cross_entropy(logits, batch.y_policy, reduction="sum").item()
            v_sum += F.mse_loss(values, batch.y_value, reduction="sum").item()
            correct += (logits.argmax(dim=1) == batch.y_policy).sum().item()
            n += batch.num_graphs
        return p_sum / n, v_sum / n, correct / n

    # -- engines and games ------------------------------------------------------
    def open_nags(self, nn_port: Optional[int], meta_port: Optional[int], meta_exploration: float = 0.0):
        engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path()))
        options = {"UseNN": nn_port is not None, "UseMetaLearner": meta_port is not None,
                   "MetaExploration": int(round(meta_exploration * 100))}
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
            if baseline == "production" and self.production_path.exists() and \
                    self.production_path.resolve() != Path(model_path).resolve():
                base_rpc = ServiceProcess("rpc_server.py", ["--model", str(self.production_path), *mc],
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
        shutil.copy2(model_path, self.production_path)
        record = {"timestamp": datetime.now().isoformat(), "model_path": str(model_path), "elo_gain": elo,
                  "elo_margin_95": margin, "promoted_to": str(self.production_path)}
        with open(self.logs_dir / "promotions.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Model promoted to %s", self.production_path)
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
    parser.add_argument("--step", choices=["parse", "supervised", "selfplay", "ppo", "evaluate", "meta", "full"],
                        default="full", help="Pipeline step to run")
    parser.add_argument("--model", help="Checkpoint to use instead of the latest one")
    parser.add_argument("--games", type=int, help="Override the number of self-play / evaluation games")
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
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
