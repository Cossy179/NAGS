import json
import socketserver
from typing import List

import torch

from chess_graph import ChessGraph
from gnn_evaluator import GNNEvaluator


def evaluate_batch(fens: List[str], evaluator: GNNEvaluator, builder: ChessGraph):
    results = []
    for fen in fens:
        data = builder.fen_to_graph(fen)
        in_dim = data.x.size(1)
        # Ensure evaluator in-dim matches
        if evaluator.gnn.proj_in.in_features != in_dim:
            # Recreate evaluator with correct input dimension
            evaluator = GNNEvaluator(in_dim=in_dim, hidden_dim=64, gnn_layers=6, policy_layers=4, value_layers=2, device=builder.device)
        policy_probs, value_est, uncertainty = evaluator.evaluate(data, mc_samples=5)
        # Return only policy vector length and top-k preview to keep payload small
        # Full 4096 vector can be large; but we return all entries to meet spec
        results.append({
            "policy": policy_probs.tolist(),
            "value": value_est,
            "uncertainty": uncertainty,
        })
    return results, evaluator


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    builder = ChessGraph(device=device)
    # Dummy init with empty graph to get in_dim; will adapt on first request
    dummy = builder.fen_to_graph("8/8/8/8/8/8/8/8 w - - 0 1")
    evaluator = GNNEvaluator(in_dim=dummy.x.size(1), hidden_dim=64, gnn_layers=6, policy_layers=4, value_layers=2, device=device)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            nonlocal evaluator
            line = self.rfile.readline()
            try:
                payload = json.loads(line.decode('utf-8'))
                fens = payload.get("fens", [])
                if isinstance(fens, str):
                    fens = [fens]
                results, evaluator = evaluate_batch(fens, evaluator, builder)
                resp = {"results": results, "policy_dim": 64 * 64}
                out = (json.dumps(resp) + "\n").encode('utf-8')
                self.wfile.write(out)
            except Exception as e:
                out = (json.dumps({"error": str(e)}) + "\n").encode('utf-8')
                self.wfile.write(out)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 5555), Handler) as srv:
        print("RPC server listening on tcp://127.0.0.1:5555 (line-delimited JSON over TCP)")
        srv.serve_forever()


if __name__ == '__main__':
    main()


