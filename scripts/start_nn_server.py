#!/usr/bin/env python3
"""
Start a neural network inference server for NAGS.
"""

import os
import sys
import argparse
import logging
import time
import signal
import threading
import json
import zmq
import torch
import chess
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gnn.board_encoder import ChessHypergraphBuilder, GNNEncoder
from gnn.evaluator_model import ChessModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("nn_server.log"),
        logging.StreamHandler()
    ]
)

class NNServer:
    """Neural Network server using ZeroMQ."""
    
    def __init__(self, model_path, host="0.0.0.0", port=50051, batch_size=16, device=None):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.address = f"tcp://{host}:{port}"
        
        # Graph builder
        self.graph_builder = ChessHypergraphBuilder()
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
    
    def _load_model(self):
        """Load the model from the specified path."""
        logging.info(f"Loading model from {self.model_path}")
        
        # Create model
        gnn_encoder = GNNEncoder(
            node_features=16,
            edge_features=5,
            hidden_dim=128,
            num_layers=6
        )
        
        model = ChessModel(
            gnn_encoder=gnn_encoder,
            node_dim=128,
            hidden_dim=256
        )
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        logging.info(f"Model loaded successfully, using device: {self.device}")
        return model
    
    def start(self):
        """Start the server."""
        logging.info(f"Starting NN server on {self.address}")
        self.socket.bind(self.address)
        self.running = True
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        try:
            self._run_server_loop()
        except Exception as e:
            logging.error(f"Error in server loop: {e}")
        finally:
            self.shutdown()
    
    def _run_server_loop(self):
        """Main server loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Wait for request with timeout
                if self.socket.poll(100) == 0:
                    continue
                
                # Receive request
                request = self.socket.recv_json()
                logging.debug(f"Received request: {request}")
                
                # Process request
                response = self._process_request(request)
                
                # Send response
                self.socket.send_json(response)
            except zmq.ZMQError as e:
                logging.error(f"ZMQ error: {e}")
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error processing request: {e}")
                # Send error response
                self.socket.send_json({
                    "error": str(e)
                })
    
    def _process_request(self, request):
        """Process a client request."""
        request_type = request.get("type", "")
        
        if request_type == "evaluate":
            return self._handle_evaluate_request(request)
        elif request_type == "batch_evaluate":
            return self._handle_batch_evaluate_request(request)
        elif request_type == "ping":
            return {"status": "ok", "message": "pong"}
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def _handle_evaluate_request(self, request):
        """Handle a single position evaluation request."""
        fen = request.get("fen", "")
        
        if not fen:
            return {"error": "Missing FEN string"}
        
        try:
            # Create board from FEN
            board = chess.Board(fen)
            
            # Convert to graph
            graph = self.graph_builder.board_to_graph(board)
            
            # Add batch dimension and move to device
            graph_batch = graph.unsqueeze(0).to(self.device)
            
            # Get legal moves mask
            legal_moves_mask = torch.zeros(4096, dtype=torch.bool)
            legal_moves_indices = []
            
            for move in board.legal_moves:
                from_square = move.from_square
                to_square = move.to_square
                move_idx = from_square * 64 + to_square
                legal_moves_mask[move_idx] = True
                legal_moves_indices.append(move_idx)
            
            legal_moves_mask = legal_moves_mask.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                move_logits, value, uncertainty = self.model(graph_batch, mc_samples=5)
                move_probs = self.model.get_move_probabilities(graph_batch, legal_moves_mask)
            
            # Extract results
            value_cp = value.item()
            uncertainty_cp = uncertainty.item()
            
            # Get move probabilities for legal moves
            move_probs = move_probs.cpu().numpy()[0]
            legal_moves_probs = [(idx, float(move_probs[idx])) for idx in legal_moves_indices]
            legal_moves_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Convert move indices to UCI strings
            uci_probs = []
            for idx, prob in legal_moves_probs:
                from_square = idx // 64
                to_square = idx % 64
                move = chess.Move(from_square, to_square)
                uci_probs.append({
                    "uci": move.uci(),
                    "probability": prob
                })
            
            return {
                "value": value_cp,
                "uncertainty": uncertainty_cp,
                "moves": uci_probs
            }
        
        except Exception as e:
            logging.error(f"Error evaluating position: {e}")
            return {"error": str(e)}
    
    def _handle_batch_evaluate_request(self, request):
        """Handle a batch evaluation request."""
        fens = request.get("fens", [])
        
        if not fens:
            return {"error": "Missing FEN strings"}
        
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(fens), self.batch_size):
                batch_fens = fens[i:i+self.batch_size]
                batch_graphs = []
                batch_legal_masks = []
                batch_legal_indices = []
                
                # Process each position
                for fen in batch_fens:
                    board = chess.Board(fen)
                    graph = self.graph_builder.board_to_graph(board)
                    batch_graphs.append(graph)
                    
                    # Get legal moves mask
                    legal_moves_mask = torch.zeros(4096, dtype=torch.bool)
                    legal_moves_idx = []
                    
                    for move in board.legal_moves:
                        from_square = move.from_square
                        to_square = move.to_square
                        move_idx = from_square * 64 + to_square
                        legal_moves_mask[move_idx] = True
                        legal_moves_idx.append((move_idx, move.uci()))
                    
                    batch_legal_masks.append(legal_moves_mask)
                    batch_legal_indices.append(legal_moves_idx)
                
                # Create batch
                from torch_geometric.data import Batch
                graph_batch = Batch.from_data_list(batch_graphs).to(self.device)
                legal_moves_mask = torch.stack(batch_legal_masks).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    move_logits, values, uncertainties = self.model(graph_batch, mc_samples=5)
                    move_probs = self.model.get_move_probabilities(graph_batch, legal_moves_mask)
                
                # Extract results
                values_np = values.cpu().numpy()
                uncertainties_np = uncertainties.cpu().numpy()
                move_probs_np = move_probs.cpu().numpy()
                
                # Format results
                for j, (value, uncertainty, probs, legal_indices) in enumerate(
                    zip(values_np, uncertainties_np, move_probs_np, batch_legal_indices)
                ):
                    # Get move probabilities for legal moves
                    uci_probs = []
                    for idx, uci in legal_indices:
                        uci_probs.append({
                            "uci": uci,
                            "probability": float(probs[idx])
                        })
                    
                    # Sort by probability
                    uci_probs.sort(key=lambda x: x["probability"], reverse=True)
                    
                    results.append({
                        "value": float(value),
                        "uncertainty": float(uncertainty),
                        "moves": uci_probs
                    })
            
            return {"results": results}
        
        except Exception as e:
            logging.error(f"Error evaluating batch: {e}")
            return {"error": str(e)}
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        logging.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
    
    def shutdown(self):
        """Shutdown the server."""
        logging.info("Shutting down NN server...")
        self.running = False
        
        # Close ZeroMQ socket and context
        if self.socket:
            self.socket.close()
        
        if self.context:
            self.context.term()
        
        logging.info("Server shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Start NAGS neural network server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50051, help="Port to bind to")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum batch size")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()
    
    # Determine device
    if args.cuda:
        device = "cuda"
    elif args.cpu:
        device = "cpu"
    else:
        device = None  # Auto-detect
    
    # Create and start server
    server = NNServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        batch_size=args.batch_size,
        device=device
    )
    
    server.start()


if __name__ == "__main__":
    main() 