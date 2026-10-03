"""GNN evaluation service for the NAGS engine.

Protocol: newline-delimited JSON over TCP; a client may send any number of
requests on one connection.

Request:  {"fens": ["<fen>", ...], "moves": [["e2e4", ...], ...]}   ("moves" optional)
Response: {"results": [{"value": v, "uncertainty": u, "move_priors": [p, ...]}, ...],
           "policy_dim": 4096}
  * value is in [-1, 1] from the side to move's point of view;
  * move_priors has one probability per requested move (summing to 1);
  * without "moves", each result carries the full 4096-entry "policy" instead.
On a bad request the response is {"error": "..."} and the connection stays open.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socketserver
import threading
from typing import Any, Dict, List, Optional

import torch

from chess_graph import FEATURE_DIM, ChessGraph
from gnn_evaluator import POLICY_DIM, GNNEvaluator, legal_move_priors, load_checkpoint

logger = logging.getLogger("rpc_server")

DEFAULT_MODEL = os.path.join("models", "production_model.pth")
MAX_BATCH = 256


class EvaluationService:
    def __init__(self, model: GNNEvaluator, builder: ChessGraph, mc_samples: int = 5):
        self.model = model
        self.builder = builder
        self.mc_samples = mc_samples
        # MC dropout flips the value head between train/eval mode, so
        # concurrent requests must not interleave.
        self.lock = threading.Lock()

    def handle(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        fens = payload.get("fens", [])
        if isinstance(fens, str):
            fens = [fens]
        if not isinstance(fens, list) or not all(isinstance(f, str) for f in fens):
            raise ValueError("'fens' must be a list of FEN strings")
        if len(fens) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} positions per request")
        moves = payload.get("moves")
        if moves is not None:
            if not isinstance(moves, list) or len(moves) != len(fens) or not all(isinstance(m, list) for m in moves):
                raise ValueError("'moves' must be a list with one move list per FEN")
        if not fens:
            return {"results": [], "policy_dim": POLICY_DIM}

        batch = self.builder.fens_to_batch(fens)  # raises ValueError for malformed FENs
        with self.lock:
            probs, values, uncertainty = self.model.evaluate(batch, mc_samples=self.mc_samples)

        results: List[Dict[str, Any]] = []
        for i in range(len(fens)):
            item: Dict[str, Any] = {"value": float(values[i]), "uncertainty": float(uncertainty[i])}
            if moves is not None:
                item["move_priors"] = legal_move_priors(probs[i], [str(m) for m in moves[i]])
            else:
                item["policy"] = probs[i].tolist()
            results.append(item)
        return {"results": results, "policy_dim": POLICY_DIM}


def load_model(path: Optional[str], device: torch.device, hidden_dim: int = 128) -> GNNEvaluator:
    if path and os.path.exists(path):
        model = load_checkpoint(path, device=device)
        logger.info("Loaded model from %s (%s)", path, model.config)
        return model
    if path:
        logger.warning("Model file %s not found.", path)
    logger.warning("Serving an UNTRAINED model: priors and values carry no chess knowledge. "
                   "Train one with training_pipeline.py and pass --model.")
    model = GNNEvaluator(in_dim=FEATURE_DIM, hidden_dim=hidden_dim, device=device)
    model.eval()
    return model


def make_server(service: EvaluationService, host: str, port: int) -> socketserver.ThreadingTCPServer:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for raw in self.rfile:
                line = raw.strip()
                if not line:
                    continue
                try:
                    response = service.handle(json.loads(line.decode('utf-8')))
                except Exception as e:  # report and keep the connection open
                    response = {"error": f"{type(e).__name__}: {e}"}
                try:
                    self.wfile.write((json.dumps(response) + "\n").encode('utf-8'))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server((host, port), Handler)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="NAGS GNN evaluation server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint written by training_pipeline.py")
    parser.add_argument("--mc-samples", type=int, default=5, help="MC-dropout samples for the value uncertainty")
    parser.add_argument("--hidden-dim", type=int, default=128, help="width of the untrained fallback model")
    parser.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = load_model(args.model, device, args.hidden_dim)
    service = EvaluationService(model, ChessGraph(device=device), mc_samples=args.mc_samples)
    with make_server(service, args.host, args.port) as srv:
        print(f"RPC server listening on tcp://{args.host}:{args.port} (line-delimited JSON over TCP)", flush=True)
        srv.serve_forever()


if __name__ == '__main__':
    main()
