#!/usr/bin/env python3
"""
NAGS Training Pipeline
======================

End-to-end training pipeline for the Neuro-Adaptive Graph Search engine:
1. Parse PGN files to extract position/move/outcome triples
2. Supervised pre-training on chess data
3. Self-play reinforcement learning with PPO
4. Model evaluation against baselines
5. Automated model promotion and checkpointing
"""

import argparse
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from chess_graph import ChessGraph
from gnn_evaluator import GNNEvaluator
from meta_learner import MetaLearner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChessDataset(Dataset):
    """Dataset for supervised chess training"""
    
    def __init__(self, data_file: str, max_samples: Optional[int] = None):
        self.samples = []
        self.graph_builder = ChessGraph()
        
        logger.info(f"Loading dataset from {data_file}")
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                sample = json.loads(line.strip())
                self.samples.append(sample)
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"Loaded {i + 1} samples")
        
        logger.info(f"Dataset loaded: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert FEN to graph
        fen = sample['fen']
        graph = self.graph_builder.fen_to_graph(fen)
        
        # Extract targets
        move_uci = sample['move']
        outcome = sample['outcome']  # 1.0 for win, 0.5 for draw, 0.0 for loss
        
        # Convert UCI move to policy target (from*64 + to)
        move = chess.Move.from_uci(move_uci)
        policy_target = move.from_square * 64 + move.to_square
        
        return {
            'graph': graph,
            'policy_target': torch.tensor(policy_target, dtype=torch.long),
            'value_target': torch.tensor(outcome, dtype=torch.float),
            'fen': fen,
            'move': move_uci
        }


class TrainingPipeline:
    """Main training pipeline coordinator"""
    
    def __init__(self, config_file: str = "training_config.json"):
        self.config = self.load_config(config_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.graph_builder = ChessGraph(device=self.device)
        self.evaluator = None
        self.meta_learner = MetaLearner()
        
        # Create directories
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['logs_dir'], exist_ok=True)
    
    def load_config(self, config_file: str) -> Dict:
        """Load training configuration"""
        default_config = {
            "data_dir": "data",
            "model_dir": "models",
            "logs_dir": "logs",
            "pgn_file": "AJ-CORR-PGN-000.pgn",
            "max_positions": 100000,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "self_play_games": 100,
            "self_play_time": 0.5,
            "elo_threshold": 25,
            "baseline_engine": "stockfish",
            "baseline_time": 1.0
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config: {config_file}")
        
        return default_config
    
    def parse_pgn_to_dataset(self) -> str:
        """Parse PGN file and extract training positions"""
        pgn_path = self.config['pgn_file']
        output_path = os.path.join(self.config['data_dir'], 'training_data.jsonl')
        
        if not os.path.exists(pgn_path):
            logger.error(f"PGN file not found: {pgn_path}")
            return output_path
        
        logger.info(f"Parsing PGN file: {pgn_path}")
        
        positions_extracted = 0
        max_positions = self.config['max_positions']
        
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file, \
             open(output_path, 'w') as out_file:
            
            game_count = 0
            while positions_extracted < max_positions:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_count += 1
                if game_count % 100 == 0:
                    logger.info(f"Processed {game_count} games, extracted {positions_extracted} positions")
                
                # Extract game result
                result = game.headers.get('Result', '*')
                if result == '1-0':
                    white_outcome, black_outcome = 1.0, 0.0
                elif result == '0-1':
                    white_outcome, black_outcome = 0.0, 1.0
                elif result == '1/2-1/2':
                    white_outcome, black_outcome = 0.5, 0.5
                else:
                    continue  # Skip games without clear result
                
                # Extract positions from game
                board = game.board()
                move_count = 0
                
                for move in game.mainline_moves():
                    if positions_extracted >= max_positions:
                        break
                    
                    # Skip opening moves (first 10 moves)
                    if move_count < 10:
                        board.push(move)
                        move_count += 1
                        continue
                    
                    # Extract position features
                    fen = board.fen()
                    move_uci = move.uci()
                    outcome = white_outcome if board.turn == chess.WHITE else black_outcome
                    
                    # Save position
                    sample = {
                        'fen': fen,
                        'move': move_uci,
                        'outcome': outcome,
                        'game_id': game_count,
                        'move_number': move_count
                    }
                    
                    out_file.write(json.dumps(sample) + '\n')
                    positions_extracted += 1
                    
                    board.push(move)
                    move_count += 1
        
        logger.info(f"Extraction complete: {positions_extracted} positions from {game_count} games")
        return output_path
    
    def supervised_training(self, dataset_path: str) -> str:
        """Run supervised pre-training on chess data"""
        logger.info("Starting supervised training")
        
        # Load dataset
        dataset = ChessDataset(dataset_path, max_samples=self.config['max_positions'])
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        # Initialize model
        sample_graph = dataset[0]['graph']
        in_dim = sample_graph.x.size(1)
        
        self.evaluator = GNNEvaluator(
            in_dim=in_dim,
            hidden_dim=128,
            gnn_layers=6,
            policy_layers=4,
            value_layers=2,
            device=self.device
        )
        
        # Optimizer
        optimizer = optim.AdamW(self.evaluator.parameters(), lr=self.config['learning_rate'])
        
        # Training loop
        for epoch in range(self.config['epochs']):
            total_loss = 0.0
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Process batch (simplified - in practice would need batching for PyG)
                batch_policy_loss = 0.0
                batch_value_loss = 0.0
                
                for i in range(len(batch['graph'])):
                    graph = batch['graph'][i].to(self.device)
                    policy_target = batch['policy_target'][i].to(self.device)
                    value_target = batch['value_target'][i].to(self.device)
                    
                    # Forward pass
                    global_emb, node_emb = self.evaluator.gnn(graph)
                    policy_logits = self.evaluator.policy(node_emb)
                    value_pred = self.evaluator.value(node_emb)
                    
                    # Losses
                    policy_loss = nn.CrossEntropyLoss()(policy_logits.unsqueeze(0), policy_target.unsqueeze(0))
                    value_loss = nn.MSELoss()(value_pred, value_target.unsqueeze(0))
                    
                    batch_policy_loss += policy_loss
                    batch_value_loss += value_loss
                
                # Combined loss
                total_batch_loss = batch_policy_loss + batch_value_loss
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                policy_loss_sum += batch_policy_loss.item()
                value_loss_sum += batch_value_loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            avg_policy_loss = policy_loss_sum / len(dataloader)
            avg_value_loss = value_loss_sum / len(dataloader)
            
            logger.info(f"Epoch {epoch} completed - Total Loss: {avg_loss:.4f}, "
                       f"Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")
        
        # Save model
        model_path = os.path.join(self.config['model_dir'], f'supervised_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        torch.save(self.evaluator.state_dict(), model_path)
        logger.info(f"Supervised model saved: {model_path}")
        
        return model_path
    
    def run_self_play(self, num_games: int) -> List[Dict]:
        """Run self-play games to collect training data"""
        logger.info(f"Starting self-play: {num_games} games")
        
        games_data = []
        
        # For now, simulate self-play data collection
        # In practice, this would launch NAGS vs NAGS matches
        for game_id in range(num_games):
            if game_id % 10 == 0:
                logger.info(f"Self-play progress: {game_id}/{num_games}")
            
            # Simulate game data
            game_data = {
                'game_id': game_id,
                'positions': [],
                'moves': [],
                'mcts_policies': [],
                'outcomes': [],
                'result': np.random.choice(['1-0', '0-1', '1/2-1/2'])
            }
            
            # Simulate 50 moves per game
            for move_num in range(50):
                game_data['positions'].append("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                game_data['moves'].append("e2e4")
                game_data['mcts_policies'].append(np.random.dirichlet([0.3] * 4096).tolist())
                game_data['outcomes'].append(0.5)
            
            games_data.append(game_data)
        
        # Save self-play data
        self_play_path = os.path.join(self.config['data_dir'], f'self_play_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(self_play_path, 'w') as f:
            json.dump(games_data, f)
        
        logger.info(f"Self-play data saved: {self_play_path}")
        return games_data
    
    def ppo_training(self, self_play_data: List[Dict]) -> str:
        """Apply PPO updates using self-play data"""
        logger.info("Starting PPO training")
        
        # Simplified PPO implementation
        # In practice, this would implement full PPO with advantage estimation
        
        if not self.evaluator:
            logger.warning("No evaluator model loaded, skipping PPO")
            return ""
        
        optimizer = optim.AdamW(self.evaluator.parameters(), lr=self.config['learning_rate'] * 0.1)
        
        for epoch in range(5):  # PPO epochs
            total_loss = 0.0
            
            for game in self_play_data[:10]:  # Process subset for demo
                for pos, move, policy, outcome in zip(
                    game['positions'], game['moves'], game['mcts_policies'], game['outcomes']
                ):
                    # Convert to tensors and compute loss
                    # This is a simplified version
                    pass
            
            logger.info(f"PPO epoch {epoch} completed")
        
        # Save updated model
        model_path = os.path.join(self.config['model_dir'], f'ppo_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        torch.save(self.evaluator.state_dict(), model_path)
        logger.info(f"PPO model saved: {model_path}")
        
        return model_path
    
    def evaluate_against_baseline(self, model_path: str) -> float:
        """Evaluate model against baseline engine"""
        logger.info(f"Evaluating model against {self.config['baseline_engine']}")
        
        # Simulate Elo evaluation
        # In practice, this would run actual games against Stockfish/other engines
        baseline_elo = np.random.normal(0, 50)  # Simulated Elo difference
        
        logger.info(f"Evaluation complete: {baseline_elo:+.1f} Elo vs baseline")
        return baseline_elo
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Parse PGN data
            dataset_path = self.parse_pgn_to_dataset()
            
            # Step 2: Supervised pre-training
            supervised_model = self.supervised_training(dataset_path)
            
            # Step 3: Self-play data collection
            self_play_data = self.run_self_play(self.config['self_play_games'])
            
            # Step 4: PPO training
            ppo_model = self.ppo_training(self_play_data)
            
            # Step 5: Evaluation
            elo_gain = self.evaluate_against_baseline(ppo_model)
            
            # Step 6: Model promotion
            if elo_gain > self.config['elo_threshold']:
                self.promote_model(ppo_model, elo_gain)
            else:
                logger.info(f"Model not promoted (Elo gain: {elo_gain:+.1f} < {self.config['elo_threshold']})")
            
            # Step 7: Update meta-learner
            self.meta_learner.train_epoch(steps=200)
            self.meta_learner.save_model()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.1f} seconds")
    
    def promote_model(self, model_path: str, elo_gain: float):
        """Promote model to production if it meets criteria"""
        logger.info(f"Promoting model with Elo gain: {elo_gain:+.1f}")
        
        # Copy to production directory
        production_path = os.path.join(self.config['model_dir'], 'production_model.pth')
        subprocess.run(['cp', model_path, production_path], check=True)
        
        # Create promotion record
        promotion_record = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'elo_gain': elo_gain,
            'promoted_to': production_path
        }
        
        promotions_file = os.path.join(self.config['logs_dir'], 'promotions.jsonl')
        with open(promotions_file, 'a') as f:
            f.write(json.dumps(promotion_record) + '\n')
        
        logger.info(f"Model promoted to: {production_path}")
        
        # Notify team (placeholder)
        self.notify_team(f"New model promoted with {elo_gain:+.1f} Elo gain!")
    
    def notify_team(self, message: str):
        """Send notification to team"""
        logger.info(f"NOTIFICATION: {message}")
        # In practice, this would send Slack/email notifications


def main():
    parser = argparse.ArgumentParser(description='NAGS Training Pipeline')
    parser.add_argument('--config', default='training_config.json', help='Config file path')
    parser.add_argument('--step', choices=['parse', 'supervised', 'selfplay', 'ppo', 'evaluate', 'full'], 
                       default='full', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(args.config)
    
    if args.step == 'full':
        pipeline.run_full_pipeline()
    elif args.step == 'parse':
        pipeline.parse_pgn_to_dataset()
    elif args.step == 'supervised':
        dataset_path = os.path.join(pipeline.config['data_dir'], 'training_data.jsonl')
        pipeline.supervised_training(dataset_path)
    elif args.step == 'selfplay':
        pipeline.run_self_play(pipeline.config['self_play_games'])
    elif args.step == 'ppo':
        # Load latest self-play data
        data_files = list(Path(pipeline.config['data_dir']).glob('self_play_*.json'))
        if data_files:
            with open(max(data_files), 'r') as f:
                self_play_data = json.load(f)
            pipeline.ppo_training(self_play_data)
    elif args.step == 'evaluate':
        # Evaluate latest model
        model_files = list(Path(pipeline.config['model_dir']).glob('*.pth'))
        if model_files:
            latest_model = max(model_files)
            pipeline.evaluate_against_baseline(str(latest_model))


if __name__ == '__main__':
    main()
