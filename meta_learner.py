import json
import math
import random
import socketserver
from typing import Dict, List, Tuple
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Meta-learning features
class MetaFeatures:
    def __init__(self):
        self.dim = 8  # Feature vector dimension
    
    def extract(self, fen: str, time_left_ms: int, last_uncertainty: float = 0.1, 
                tactical_shot_ratio: float = 0.2) -> torch.Tensor:
        """Extract meta-learning features from position"""
        features = torch.zeros(self.dim)
        
        # Parse basic position info
        fields = fen.split()
        board_str = fields[0]
        side_to_move = fields[1] == 'w'
        
        # Feature 0: Material balance (normalized)
        material_balance = self._calculate_material_balance(board_str)
        features[0] = material_balance / 10.0  # Normalize by ~queen value
        
        # Feature 1: Piece activity (center control proxy)
        center_activity = self._estimate_center_activity(board_str)
        features[1] = center_activity
        
        # Feature 2: Time pressure (log-scaled)
        time_factor = math.log(max(1, time_left_ms)) / math.log(300000)  # 5min reference
        features[2] = min(1.0, time_factor)
        
        # Feature 3: Last move uncertainty
        features[3] = min(1.0, last_uncertainty)
        
        # Feature 4: Tactical shot ratio (captures/checks available)
        features[4] = min(1.0, tactical_shot_ratio)
        
        # Feature 5: Game phase (endgame indicator)
        piece_count = sum(1 for c in board_str if c.isalpha())
        features[5] = max(0.0, 1.0 - piece_count / 32.0)
        
        # Feature 6: Side to move
        features[6] = 1.0 if side_to_move else 0.0
        
        # Feature 7: King safety proxy (castling rights)
        castling = fields[2] if len(fields) > 2 else '-'
        king_safety = len([c for c in castling if c in 'KQkq']) / 4.0
        features[7] = king_safety
        
        return features
    
    def _calculate_material_balance(self, board_str: str) -> float:
        values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
                  'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        balance = 0
        for c in board_str:
            if c in values:
                balance += values[c] if c.isupper() else -values[c]
        return balance
    
    def _estimate_center_activity(self, board_str: str) -> float:
        # Rough estimate: count pieces that could influence center
        activity = 0
        for c in board_str:
            if c.lower() in 'nbrq':  # Active pieces
                activity += 1
        return min(1.0, activity / 16.0)


class MetaLearnerMLP(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # 3 outputs: DFS depth delta, MCTS budget delta, bandit exploration delta
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))  # Output in [-1, 1]


class MetaLearner:
    def __init__(self, model_path: str = "meta_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = MetaFeatures()
        self.model = MetaLearnerMLP().to(self.device)
        self.model_path = model_path
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training data storage
        self.training_data: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        
        # Try to load existing model
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded meta-learner model from {model_path}")
        except FileNotFoundError:
            print("No existing meta-learner model found, starting fresh")
    
    def predict(self, fen: str, time_left_ms: int, last_uncertainty: float = 0.1, 
                tactical_shot_ratio: float = 0.2) -> Dict[str, float]:
        """Predict hyperparameter adjustments for current position"""
        features = self.features.extract(fen, time_left_ms, last_uncertainty, tactical_shot_ratio)
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            deltas = self.model(features).squeeze(0).cpu()
        
        return {
            'dfs_depth_delta': float(deltas[0]),      # -1 to +1, multiply by max_depth_adjustment
            'mcts_budget_delta': float(deltas[1]),    # -1 to +1, multiply by budget_multiplier
            'bandit_exploration_delta': float(deltas[2])  # -1 to +1, adjust exploration constant
        }
    
    def add_training_sample(self, fen: str, time_left_ms: int, last_uncertainty: float,
                           tactical_shot_ratio: float, chosen_deltas: Dict[str, float], 
                           elo_gain_per_sec: float):
        """Add a training sample from self-play"""
        features = self.features.extract(fen, time_left_ms, last_uncertainty, tactical_shot_ratio)
        deltas_tensor = torch.tensor([
            chosen_deltas['dfs_depth_delta'],
            chosen_deltas['mcts_budget_delta'], 
            chosen_deltas['bandit_exploration_delta']
        ])
        
        self.training_data.append((features, deltas_tensor, elo_gain_per_sec))
        
        # Keep only recent samples (sliding window)
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-8000:]
    
    def train_step(self, batch_size: int = 64) -> float:
        """Perform one training step"""
        if len(self.training_data) < batch_size:
            return 0.0
        
        # Sample batch
        batch_indices = random.sample(range(len(self.training_data)), batch_size)
        batch_features = []
        batch_targets = []
        batch_rewards = []
        
        for idx in batch_indices:
            features, deltas, reward = self.training_data[idx]
            batch_features.append(features)
            batch_targets.append(deltas)
            batch_rewards.append(reward)
        
        features_batch = torch.stack(batch_features).to(self.device)
        targets_batch = torch.stack(batch_targets).to(self.device)
        rewards_batch = torch.tensor(batch_rewards, device=self.device)
        
        # Normalize rewards to [0, 1] for weighting
        if rewards_batch.std() > 1e-6:
            rewards_normalized = (rewards_batch - rewards_batch.mean()) / rewards_batch.std()
            weights = torch.sigmoid(rewards_normalized)  # Convert to [0, 1]
        else:
            weights = torch.ones_like(rewards_batch)
        
        # Forward pass
        self.model.train()
        predictions = self.model(features_batch)
        
        # Weighted MSE loss (emphasize samples with higher Elo gain)
        loss = (weights.unsqueeze(1) * (predictions - targets_batch) ** 2).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return float(loss.item())
    
    def train_epoch(self, steps: int = 100) -> float:
        """Train for multiple steps"""
        total_loss = 0.0
        for _ in range(steps):
            loss = self.train_step()
            total_loss += loss
        
        avg_loss = total_loss / steps if steps > 0 else 0.0
        print(f"Meta-learner training: {steps} steps, avg loss = {avg_loss:.6f}")
        return avg_loss
    
    def save_model(self):
        """Save model and training data"""
        torch.save(self.model.state_dict(), self.model_path)
        with open("meta_training_data.pkl", "wb") as f:
            pickle.dump(self.training_data, f)
        print(f"Meta-learner model saved to {self.model_path}")
    
    def load_training_data(self):
        """Load existing training data"""
        try:
            with open("meta_training_data.pkl", "rb") as f:
                self.training_data = pickle.load(f)
            print(f"Loaded {len(self.training_data)} training samples")
        except FileNotFoundError:
            print("No existing training data found")


# RPC Server for meta-learner inference
def main():
    meta_learner = MetaLearner()
    meta_learner.load_training_data()
    
    class MetaHandler(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline()
            try:
                payload = json.loads(line.decode('utf-8'))
                
                if payload.get('command') == 'predict':
                    # Prediction request
                    fen = payload.get('fen', '')
                    time_left = payload.get('time_left_ms', 30000)
                    uncertainty = payload.get('last_uncertainty', 0.1)
                    tactical_ratio = payload.get('tactical_shot_ratio', 0.2)
                    
                    result = meta_learner.predict(fen, time_left, uncertainty, tactical_ratio)
                    resp = {"status": "ok", "deltas": result}
                    
                elif payload.get('command') == 'add_sample':
                    # Training sample
                    fen = payload.get('fen', '')
                    time_left = payload.get('time_left_ms', 30000)
                    uncertainty = payload.get('last_uncertainty', 0.1)
                    tactical_ratio = payload.get('tactical_shot_ratio', 0.2)
                    deltas = payload.get('chosen_deltas', {})
                    elo_gain = payload.get('elo_gain_per_sec', 0.0)
                    
                    meta_learner.add_training_sample(fen, time_left, uncertainty, 
                                                   tactical_ratio, deltas, elo_gain)
                    resp = {"status": "ok", "samples": len(meta_learner.training_data)}
                    
                elif payload.get('command') == 'train':
                    # Training request
                    steps = payload.get('steps', 100)
                    loss = meta_learner.train_epoch(steps)
                    meta_learner.save_model()
                    resp = {"status": "ok", "loss": loss}
                    
                else:
                    resp = {"status": "error", "message": "Unknown command"}
                
                out = (json.dumps(resp) + "\n").encode('utf-8')
                self.wfile.write(out)
                
            except Exception as e:
                out = (json.dumps({"status": "error", "message": str(e)}) + "\n").encode('utf-8')
                self.wfile.write(out)
    
    with socketserver.ThreadingTCPServer(("127.0.0.1", 5556), MetaHandler) as srv:
        print("Meta-learner RPC server listening on tcp://127.0.0.1:5556")
        srv.serve_forever()


# Self-play simulation for testing
def simulate_self_play(meta_learner: MetaLearner, games: int = 100):
    """Simulate self-play to generate training data"""
    print(f"Simulating {games} games for meta-learner training...")
    
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Opening
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Middlegame
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # Endgame
    ]
    
    for game in range(games):
        fen = random.choice(test_positions)
        time_left = random.randint(5000, 120000)  # 5sec to 2min
        uncertainty = random.uniform(0.05, 0.4)
        tactical_ratio = random.uniform(0.0, 0.6)
        
        # Get prediction
        deltas = meta_learner.predict(fen, time_left, uncertainty, tactical_ratio)
        
        # Simulate game outcome (random with some correlation to hyperparams)
        base_elo_gain = random.uniform(-2.0, 2.0)
        # Reward balanced hyperparameters in middlegame, extreme ones in tactical positions
        balance_penalty = sum(abs(d) for d in deltas.values()) * 0.1
        if tactical_ratio > 0.4:
            balance_penalty *= -1  # Reward extremes in tactical positions
        
        elo_gain_per_sec = (base_elo_gain - balance_penalty) / max(1, time_left / 1000)
        
        # Add training sample
        meta_learner.add_training_sample(fen, time_left, uncertainty, tactical_ratio, 
                                       deltas, elo_gain_per_sec)
        
        if game % 20 == 0:
            print(f"Game {game}: Elo/sec = {elo_gain_per_sec:.4f}, deltas = {deltas}")
    
    # Train the model
    meta_learner.train_epoch(200)
    meta_learner.save_model()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'simulate':
        # Training mode
        meta_learner = MetaLearner()
        meta_learner.load_training_data()
        simulate_self_play(meta_learner, 500)
    else:
        # Server mode
        main()
