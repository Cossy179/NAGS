"""Meta-learner: predicts per-move search hyperparameter adjustments.

The NAGS engine asks for three deltas in [-1, 1] before each search (DFS depth
cap, MCTS simulation budget, PUCT exploration). Training data are
(position features, deltas actually used, reward) triples; the training
pipeline produces them from self-play, where the engine adds exploration
noise to the deltas (UCI option MetaExploration) and the reward is the game
result from the mover's point of view.

Training is advantage-weighted regression: the model regresses towards the
deltas that were used, weighted by exp(advantage / beta), so deltas that led to
better-than-average results pull the policy towards them and worse ones are
nearly ignored.

Protocol (newline-delimited JSON over TCP, persistent connections):
  {"command": "predict", "fen", "time_left_ms", "last_uncertainty", "tactical_shot_ratio"}
      -> {"status": "ok", "deltas": {...}}
  {"command": "add_sample", ..., "chosen_deltas": {...}, "reward": r}
      -> {"status": "ok", "samples": n}
  {"command": "train", "steps": n}  -> {"status": "ok", "loss": x}
  {"command": "save"}               -> {"status": "ok"}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socketserver
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

DELTA_KEYS = ("dfs_depth_delta", "mcts_budget_delta", "bandit_exploration_delta")
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}


class MetaFeatures:
    dim = 8

    def extract(self, fen: str, time_left_ms: float, last_uncertainty: float = 0.1,
                tactical_shot_ratio: float = 0.2) -> torch.Tensor:
        fields = fen.split()
        if len(fields) < 2:
            raise ValueError(f"invalid FEN {fen!r}")
        board_str = fields[0]
        white_to_move = fields[1] == 'w'
        castling = fields[2] if len(fields) > 2 else '-'

        material = 0
        active = 0
        pieces = 0
        for c in board_str:
            if c.lower() in PIECE_VALUES:
                v = PIECE_VALUES[c.lower()]
                material += v if c.isupper() else -v
                pieces += 1
                if c.lower() in 'nbrq':
                    active += 1
        if not white_to_move:
            material = -material  # side-to-move perspective

        f = torch.zeros(self.dim)
        f[0] = max(-1.0, min(1.0, material / 10.0))
        f[1] = min(1.0, active / 14.0)
        f[2] = min(1.0, math.log(max(1.0, float(time_left_ms))) / math.log(300000.0))
        f[3] = max(0.0, min(1.0, float(last_uncertainty)))
        f[4] = max(0.0, min(1.0, float(tactical_shot_ratio)))
        f[5] = max(0.0, 1.0 - pieces / 32.0)
        f[6] = 1.0 if white_to_move else 0.0
        f[7] = len([c for c in castling if c in 'KQkq']) / 4.0
        return f


class MetaLearnerMLP(nn.Module):
    def __init__(self, input_dim: int = MetaFeatures.dim, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, len(DELTA_KEYS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


Sample = Tuple[List[float], List[float], float]  # features, deltas, reward


class MetaLearner:
    def __init__(self, model_path: str = "meta_model.pth", data_path: str = "meta_training_data.json",
                 max_samples: int = 20000, awr_beta: float = 1.0, seed: int = 0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = MetaFeatures()
        self.model = MetaLearnerMLP().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model_path = model_path
        self.data_path = data_path
        self.max_samples = max_samples
        self.awr_beta = awr_beta
        self.rng = random.Random(seed)
        self.training_data: List[Sample] = []
        self.lock = threading.RLock()

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                print(f"Loaded meta-learner model from {model_path}")
            except Exception as e:  # incompatible/old file: start fresh rather than crash
                print(f"Could not load {model_path} ({e}); starting with a fresh model")
        else:
            print("No existing meta-learner model found, starting fresh")

    # -- inference ---------------------------------------------------------
    def predict(self, fen: str, time_left_ms: float, last_uncertainty: float = 0.1,
                tactical_shot_ratio: float = 0.2) -> Dict[str, float]:
        x = self.features.extract(fen, time_left_ms, last_uncertainty, tactical_shot_ratio).unsqueeze(0).to(self.device)
        with self.lock, torch.no_grad():
            self.model.eval()
            deltas = self.model(x).squeeze(0).cpu().tolist()
        return dict(zip(DELTA_KEYS, (float(d) for d in deltas)))

    # -- data --------------------------------------------------------------
    def add_training_sample(self, fen: str, time_left_ms: float, last_uncertainty: float,
                            tactical_shot_ratio: float, chosen_deltas: Dict[str, float], reward: float) -> int:
        feats = self.features.extract(fen, time_left_ms, last_uncertainty, tactical_shot_ratio).tolist()
        deltas = [max(-1.0, min(1.0, float(chosen_deltas[k]))) for k in DELTA_KEYS]
        with self.lock:
            self.training_data.append((feats, deltas, float(reward)))
            if len(self.training_data) > self.max_samples:
                del self.training_data[: len(self.training_data) - self.max_samples]
            return len(self.training_data)

    def save_training_data(self) -> None:
        with self.lock:
            data = list(self.training_data)
        with open(self.data_path, "w") as f:
            json.dump(data, f)

    def load_training_data(self) -> None:
        if not self.data_path or not os.path.exists(self.data_path):
            print("No existing meta-learner training data found")
            return
        with open(self.data_path) as f:
            raw = json.load(f)
        with self.lock:
            self.training_data = [(list(map(float, a)), list(map(float, b)), float(r)) for a, b, r in raw]
        print(f"Loaded {len(self.training_data)} meta-learner training samples")

    # -- training ----------------------------------------------------------
    def train_step(self, batch_size: int = 64) -> Optional[float]:
        with self.lock:
            n = len(self.training_data)
            if n < 2:
                return None
            batch = self.rng.sample(self.training_data, min(batch_size, n))
            feats = torch.tensor([b[0] for b in batch], device=self.device)
            targets = torch.tensor([b[1] for b in batch], device=self.device)
            rewards = torch.tensor([b[2] for b in batch], device=self.device)

            std = rewards.std()
            adv = (rewards - rewards.mean()) / std if std > 1e-6 else torch.zeros_like(rewards)
            weights = torch.exp((adv / self.awr_beta).clamp(max=3.0))
            weights = weights / weights.mean()

            self.model.train()
            pred = self.model(feats)
            loss = (weights.unsqueeze(1) * (pred - targets) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return float(loss.item())

    def train_epoch(self, steps: int = 100) -> Optional[float]:
        losses = [l for l in (self.train_step() for _ in range(max(0, steps))) if l is not None]
        if not losses:
            print("Meta-learner training skipped: not enough samples")
            return None
        avg = sum(losses) / len(losses)
        print(f"Meta-learner training: {len(losses)} steps, avg loss = {avg:.6f}")
        return avg

    def save_model(self) -> None:
        with self.lock:
            torch.save(self.model.state_dict(), self.model_path)
        self.save_training_data()
        print(f"Meta-learner model saved to {self.model_path}")


def make_server(meta: MetaLearner, host: str, port: int) -> socketserver.ThreadingTCPServer:
    def dispatch(payload: Dict) -> Dict:
        if not isinstance(payload, dict):
            return {"status": "error", "message": "request must be a JSON object"}
        cmd = payload.get('command')
        if cmd == 'predict':
            deltas = meta.predict(payload.get('fen', ''), payload.get('time_left_ms', 30000),
                                  payload.get('last_uncertainty', 0.1), payload.get('tactical_shot_ratio', 0.2))
            return {"status": "ok", "deltas": deltas}
        if cmd == 'add_sample':
            reward = payload.get('reward', payload.get('elo_gain_per_sec'))
            if reward is None:
                return {"status": "error", "message": "missing 'reward'"}
            n = meta.add_training_sample(payload.get('fen', ''), payload.get('time_left_ms', 30000),
                                         payload.get('last_uncertainty', 0.1), payload.get('tactical_shot_ratio', 0.2),
                                         payload.get('chosen_deltas', {}), reward)
            return {"status": "ok", "samples": n}
        if cmd == 'train':
            loss = meta.train_epoch(int(payload.get('steps', 100)))
            meta.save_model()
            return {"status": "ok", "loss": loss}
        if cmd == 'save':
            meta.save_model()
            return {"status": "ok"}
        return {"status": "error", "message": f"unknown command {cmd!r}"}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for raw in self.rfile:  # persistent connection: one JSON request per line
                line = raw.strip()
                if not line:
                    continue
                try:
                    resp = dispatch(json.loads(line.decode('utf-8')))
                except Exception as e:
                    resp = {"status": "error", "message": f"{type(e).__name__}: {e}"}
                try:
                    self.wfile.write((json.dumps(resp) + "\n").encode('utf-8'))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server((host, port), Handler)


def selftest(steps: int = 300) -> float:
    """Trains on synthetic data in memory (nothing is saved) and returns the
    mean distance of the learned deltas from the rewarded target; a working
    learner moves its predictions towards the deltas that earned high reward."""
    meta = MetaLearner(model_path="", data_path="", seed=1)
    rng = random.Random(2)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    target = [0.6, -0.4, 0.2]
    for _ in range(2000):
        deltas = [max(-1.0, min(1.0, rng.gauss(0.0, 0.5))) for _ in DELTA_KEYS]
        reward = -sum((d - t) ** 2 for d, t in zip(deltas, target)) + rng.gauss(0, 0.05)
        meta.add_training_sample(fen, 60000, 0.1, 0.2, dict(zip(DELTA_KEYS, deltas)), reward)
    meta.train_epoch(steps)
    pred = meta.predict(fen, 60000, 0.1, 0.2)
    return sum(abs(pred[k] - t) for k, t in zip(DELTA_KEYS, target)) / len(target)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="NAGS meta-learner service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--model", default="meta_model.pth")
    parser.add_argument("--data", default="meta_training_data.json")
    parser.add_argument("--selftest", action="store_true", help="train on synthetic data in memory and report")
    args = parser.parse_args(argv)

    if args.selftest:
        err = selftest()
        print(f"selftest: mean |prediction - rewarded target| = {err:.3f} (untrained ~0.4)")
        return

    meta = MetaLearner(model_path=args.model, data_path=args.data)
    meta.load_training_data()
    with make_server(meta, args.host, args.port) as srv:
        print(f"Meta-learner RPC server listening on tcp://{args.host}:{args.port}", flush=True)
        srv.serve_forever()


if __name__ == '__main__':
    main()
