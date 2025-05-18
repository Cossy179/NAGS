#!/usr/bin/env python3
"""
Train the NAGS chess model on GM correspondence games in supervised mode.
"""

import os
import sys
import argparse
import time
import logging
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import chess.engine
from tqdm import tqdm
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gnn.board_encoder import ChessHypergraphBuilder, GNNEncoder
from gnn.evaluator_model import ChessModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

class PGNDataset(Dataset):
    """Dataset for loading PGN games."""
    
    def __init__(self, pgn_file, max_games=None, filter_min_elo=2500, use_gm_only=True):
        self.pgn_file = pgn_file
        self.max_games = max_games
        self.filter_min_elo = filter_min_elo
        self.use_gm_only = use_gm_only
        
        # Initialize hypergraph builder
        self.graph_builder = ChessHypergraphBuilder()
        
        # Load and preprocess games
        self.positions = []
        self.load_games()
        
    def load_games(self):
        """Load games from PGN file and extract positions."""
        logging.info(f"Loading games from {self.pgn_file}")
        
        count = 0
        with open(self.pgn_file, "r") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                # Apply filters
                if self.use_gm_only and not self._is_gm_game(game):
                    continue
                    
                if self.filter_min_elo > 0 and not self._meets_elo_threshold(game):
                    continue
                
                # Process the game
                self._extract_positions(game)
                
                count += 1
                if count % 100 == 0:
                    logging.info(f"Processed {count} games, extracted {len(self.positions)} positions")
                
                if self.max_games and count >= self.max_games:
                    break
        
        logging.info(f"Finished loading {count} games, extracted {len(self.positions)} positions")
    
    def _is_gm_game(self, game):
        """Check if at least one player is a GM."""
        headers = game.headers
        return ("GM" in headers.get("WhiteTitle", "") or 
                "GM" in headers.get("BlackTitle", ""))
    
    def _meets_elo_threshold(self, game):
        """Check if at least one player meets the Elo threshold."""
        headers = game.headers
        white_elo = int(headers.get("WhiteElo", "0") or "0")
        black_elo = int(headers.get("BlackElo", "0") or "0")
        return white_elo >= self.filter_min_elo or black_elo >= self.filter_min_elo
    
    def _extract_positions(self, game):
        """Extract training positions from a game."""
        result_str = game.headers.get("Result", "*")
        result_val = self._parse_result(result_str)
        
        # Skip games without a clear result
        if result_val is None:
            return
        
        board = game.board()
        moves = list(game.mainline_moves())
        
        for move_idx, move in enumerate(moves):
            # Skip early opening moves
            if move_idx < 10:
                board.push(move)
                continue
            
            # Extract (position, move) as training data
            fen = board.fen()
            
            # Create move label (one-hot encoded)
            legal_moves = list(board.legal_moves)
            move_idx_in_legal = legal_moves.index(move) if move in legal_moves else -1
            
            if move_idx_in_legal >= 0:
                # Position, move index, result value
                self.positions.append((board.copy(), move_idx_in_legal, result_val))
            
            board.push(move)
    
    def _parse_result(self, result_str):
        """Parse the game result string to a value."""
        if result_str == "1-0":
            return 1.0  # White win
        elif result_str == "0-1":
            return -1.0  # Black win
        elif result_str == "1/2-1/2":
            return 0.0  # Draw
        else:
            return None  # Unknown/incomplete
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        board, move_idx, result = self.positions[idx]
        
        # Convert board to graph
        graph = self.graph_builder.board_to_graph(board)
        
        # Create move target tensor
        num_legal_moves = len(list(board.legal_moves))
        move_target = torch.zeros(4096)  # 64*64 possible moves
        move_target[move_idx] = 1.0
        
        # Create legal moves mask
        legal_moves_mask = torch.zeros(4096, dtype=torch.bool)
        for i, move in enumerate(board.legal_moves):
            from_square = move.from_square
            to_square = move.to_square
            move_idx = from_square * 64 + to_square
            legal_moves_mask[move_idx] = True
        
        # Get result from side-to-move perspective
        if board.turn == chess.BLACK:
            result = -result
        
        return graph, move_target, legal_moves_mask, torch.tensor([result], dtype=torch.float)


def collate_fn(batch):
    """Custom collate function for batching graphs."""
    from torch_geometric.data import Batch
    
    graphs, move_targets, legal_moves_masks, results = zip(*batch)
    
    # Batch the graphs
    batched_graph = Batch.from_data_list(graphs)
    
    # Stack the targets and masks
    stacked_move_targets = torch.stack(move_targets)
    stacked_legal_moves_masks = torch.stack(legal_moves_masks)
    stacked_results = torch.stack(results)
    
    return batched_graph, stacked_move_targets, stacked_legal_moves_masks, stacked_results


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (graph_batch, move_targets, legal_moves_masks, results) in enumerate(pbar):
        # Move data to device
        graph_batch = graph_batch.to(device)
        move_targets = move_targets.to(device)
        legal_moves_masks = legal_moves_masks.to(device)
        results = results.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        move_logits, value, _ = model(graph_batch)
        
        # Compute policy loss (masked cross-entropy)
        move_logits = move_logits.view(-1, 4096)
        masked_move_logits = move_logits.clone()
        masked_move_logits[~legal_moves_masks] = float('-inf')
        
        policy_loss = F.cross_entropy(masked_move_logits, move_targets.argmax(dim=1))
        
        # Compute value loss (MSE)
        value_loss = F.mse_loss(value, results)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_policy_loss += policy_loss.item() * len(results)
        total_value_loss += value_loss.item() * len(results)
        total_samples += len(results)
        
        # Update progress bar
        pbar.set_postfix({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        })
    
    return total_policy_loss / total_samples, total_value_loss / total_samples


def validate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_policy_loss = 0
    total_value_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for graph_batch, move_targets, legal_moves_masks, results in tqdm(dataloader, desc="Validating"):
            # Move data to device
            graph_batch = graph_batch.to(device)
            move_targets = move_targets.to(device)
            legal_moves_masks = legal_moves_masks.to(device)
            results = results.to(device)
            
            # Forward pass
            move_logits, value, _ = model(graph_batch)
            
            # Compute policy loss (masked cross-entropy)
            move_logits = move_logits.view(-1, 4096)
            masked_move_logits = move_logits.clone()
            masked_move_logits[~legal_moves_masks] = float('-inf')
            
            policy_loss = F.cross_entropy(masked_move_logits, move_targets.argmax(dim=1))
            
            # Compute value loss (MSE)
            value_loss = F.mse_loss(value, results)
            
            # Track metrics
            total_policy_loss += policy_loss.item() * len(results)
            total_value_loss += value_loss.item() * len(results)
            total_samples += len(results)
    
    return total_policy_loss / total_samples, total_value_loss / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train NAGS model on GM games")
    parser.add_argument("--pgn_path", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_games", type=int, default=10000, help="Maximum games to process")
    parser.add_argument("--min_elo", type=int, default=2500, help="Minimum Elo filter")
    parser.add_argument("--gm_only", type=bool, default=True, help="Only use GM games")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for models")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create datasets
    full_dataset = PGNDataset(
        args.pgn_path,
        max_games=args.max_games,
        filter_min_elo=args.min_elo,
        use_gm_only=args.gm_only
    )
    
    # Split into train/val
    val_size = min(10000, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Create model
    gnn_encoder = GNNEncoder(
        node_features=16,
        edge_features=5,
        hidden_dim=args.hidden_dim,
        num_layers=6
    )
    
    model = ChessModel(
        gnn_encoder=gnn_encoder,
        node_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim * 2
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Create summary writer for TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}")
        
        # Train
        train_policy_loss, train_value_loss = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Validate
        val_policy_loss, val_value_loss = validate(model, val_loader, device)
        
        # Log metrics
        writer.add_scalar("Loss/train_policy", train_policy_loss, epoch)
        writer.add_scalar("Loss/train_value", train_value_loss, epoch)
        writer.add_scalar("Loss/val_policy", val_policy_loss, epoch)
        writer.add_scalar("Loss/val_value", val_value_loss, epoch)
        
        # Save checkpoint
        val_loss = val_policy_loss + val_value_loss
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_policy_loss": train_policy_loss,
            "train_value_loss": train_value_loss,
            "val_policy_loss": val_policy_loss,
            "val_value_loss": val_value_loss,
        }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model with val_loss={val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Training completed. Saved final model to {final_model_path}")


if __name__ == "__main__":
    main() 