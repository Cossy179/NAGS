"""GNN evaluation service for the NAGS engine.

Protocol: newline-delimited JSON over TCP; a client may send any number of
requests on one connection.

Request:  {"fens": ["<fen>", ...], "moves": [["e2e4", ...], ...]}   ("moves" optional)
Response: {"results": [{"value": v, "uncertainty": u, "move_priors": [p, ...]}, ...],
           "policy_dim": 4096, "model_format": "nags-gnn-v2"}
  * value is in [-1, 1] from the side to move's point of view;
  * move_priors has one probability per requested move (summing to 1);
  * without "moves", each result carries the full "policy" instead;
  * policy_dim is the served network's policy size: 4096 for 'nags-gnn-v2'
    checkpoints and the untrained fallback, 4228 for 'nags-gnn-v3'
    (policy_encoding.py: underpromotions get entries of their own);
  * model_format names the served checkpoint format.
On a bad request the response is {"error": "..."} and the connection stays open.

Optional "history": one entry per FEN, null or {"fen": <fen>, "moves": [<uci>, ...]}: the moves
since an earlier position that lead to this one (the engine sends those since the last capture or
pawn move). Only a v3 network trained with histories uses them (its replies then carry
"history": true); every other network ignores the field, exactly as before.

--model takes either checkpoint format (gnn_v3.load_any_checkpoint); each is
served with its own graph builder and move encoding.

Speed (none of these changes a reply's fields or meaning):
  * --backend: 'nags-gnn-v3' networks are served by default through
    gnn_infer's torch backend on relational_fast graphs, which gives
    bit-identical replies to the reference path ('legacy') in a third
    less time; 'onnx' runs the network with ONNX Runtime (--providers
    cpu, cuda, tensorrt). v2 networks always use the legacy path.
  * --threads: torch / ONNX Runtime threads (default 1). A request is one
    or a few positions, so more threads mostly synchronise: 4 threads made
    a single-position request 3x slower than 1 on a 4-core machine.
  * --batch-wait-ms: requests from all connections that arrive within
    this window are evaluated as one batch (a GPU evaluates a batch of 64
    in about the time of one position), and each connection may send its
    next request before the reply to the previous one arrives.
  * --cache-size: replies to repeated requests (same FEN, same moves) come
    from a cache. With --mc-samples > 1 a cached value repeats one draw of
    the Monte-Carlo dropout.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import queue
import socketserver
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from chess_graph import FEATURE_DIM, ChessGraph
from gnn_evaluator import POLICY_DIM, GNNEvaluator, legal_move_priors, load_checkpoint
logger = logging.getLogger("rpc_server")

try:
    from gnn_v3 import GNNAdapter, GNNAdapterV2, as_adapter, load_any_checkpoint
except Exception as _v3_error:  # the v3 code must never stop v2 networks being served
    _V3_IMPORT_ERROR: Optional[Exception] = _v3_error

    class GNNAdapter:  # type: ignore[no-redef]
        """Legacy-only stand-in for gnn_v3's adapters: 'nags-gnn-v2' served exactly as before."""
        format = "nags-gnn-v2"
        policy_dim = POLICY_DIM

        def __init__(self, model: GNNEvaluator, builder: ChessGraph):
            self.model = model
            self.builder = builder

        def build_batch(self, positions):
            return self.builder.fens_to_batch(positions)

        def evaluate(self, batch, mc_samples: int = 5):
            return self.model.evaluate(batch, mc_samples=mc_samples)

        def move_priors(self, policy_row, moves):
            return legal_move_priors(policy_row, moves)

    GNNAdapterV2 = GNNAdapter  # type: ignore[misc]

    def as_adapter(model, builder=None):  # type: ignore[no-redef]
        if isinstance(model, GNNAdapter):
            return model
        return GNNAdapter(model, builder if builder is not None else ChessGraph(device=model.device))

    def load_any_checkpoint(path, device=None):  # type: ignore[no-redef]
        raise RuntimeError(f"gnn_v3 unavailable ({_V3_IMPORT_ERROR})")
else:
    _V3_IMPORT_ERROR = None

DEFAULT_MODEL = os.path.join("models", "production_model.pth")
MAX_BATCH = 256


class _Batcher:
    """Evaluates requests from all connections that arrive within `wait_ms` of each other as one batch."""

    def __init__(self, backend: Any, mc_samples: int, wait_ms: float, max_batch: int, lock: threading.Lock):
        from relational_fast import collate

        self.collate = collate
        self.backend = backend
        self.mc_samples = mc_samples
        self.wait = max(0.0, wait_ms) / 1000.0
        self.max_batch = max(1, max_batch)
        self.lock = lock
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.batches = 0
        self.positions = 0
        # Open connections: once each has a request in the batch, waiting longer gains nothing.
        self.connections = 0
        self.connections_lock = threading.Lock()
        threading.Thread(target=self._loop, name="batcher", daemon=True).start()

    def connected(self, delta: int) -> None:
        with self.connections_lock:
            self.connections += delta

    def submit(self, graphs: Sequence[Any]) -> Future:
        future: Future = Future()
        self.jobs.put((list(graphs), future))
        return future

    def _loop(self) -> None:
        while True:
            jobs = [self.jobs.get()]
            count = len(jobs[0][0])
            deadline = time.monotonic() + self.wait
            while count < self.max_batch and len(jobs) < self.connections:
                remaining = deadline - time.monotonic()
                try:
                    job = self.jobs.get(timeout=remaining) if remaining > 0 else self.jobs.get_nowait()
                except queue.Empty:
                    break
                jobs.append(job)
                count += len(job[0])
            try:
                graphs = [g for gs, _ in jobs for g in gs]
                with self.lock:
                    probs, values, unc = self.backend.run(self.collate(graphs), mc_samples=self.mc_samples)
                self.batches += 1
                self.positions += len(graphs)
                start = 0
                for gs, future in jobs:
                    end = start + len(gs)
                    future.set_result((probs[start:end], values[start:end], unc[start:end]))
                    start = end
            except Exception as e:  # every waiting request gets the error
                for _, future in jobs:
                    if not future.done():
                        future.set_exception(e)


class EvaluationService:
    """Serves a GNNEvaluator with its ChessGraph (as before) or any GNNAdapter ('nags-gnn-v2' / 'nags-gnn-v3').

    backend: None serves through the adapter exactly as before; a gnn_infer
    backend (v3 only) evaluates relational_fast graphs instead. cache_size
    > 0 keeps that many replies; batch_wait_ms > 0 batches concurrent
    requests (see the module docstring).
    """

    def __init__(self, model: GNNEvaluator | GNNAdapter, builder: Optional[ChessGraph] = None, mc_samples: int = 5,
                 backend: Any = None, cache_size: int = 0, batch_wait_ms: float = 0.0, max_batch: int = MAX_BATCH):
        self.adapter = as_adapter(model, builder)
        self.model = self.adapter.model
        self.builder = self.adapter.builder
        self.mc_samples = mc_samples
        if backend is not None and self.adapter.format != "nags-gnn-v3":
            raise ValueError(f"backend {getattr(backend, 'name', backend)!r} serves nags-gnn-v3 networks only, "
                             f"not {self.adapter.format}")
        self.backend = backend
        if backend is not None:
            from relational_fast import build_arrays
            self._build_arrays = build_arrays
        # MC dropout flips the value head between train/eval mode, so
        # concurrent requests must not interleave.
        self.lock = threading.Lock()
        self.batcher = _Batcher(backend, mc_samples, batch_wait_ms, max_batch, self.lock) \
            if backend is not None and batch_wait_ms > 0 else None
        try:
            from gnn_v3 import is_history_aware
            self.history_aware = self.adapter.format == "nags-gnn-v3" and is_history_aware(self.model)
        except Exception:  # gnn_v3 unavailable: only v2 networks are served
            self.history_aware = False
        self.cache_size = max(0, int(cache_size))
        self.cache: "collections.OrderedDict[Any, Dict[str, Any]]" = collections.OrderedDict()
        self.cache_lock = threading.Lock()
        self.cache_hits = 0

    @staticmethod
    def _parse(payload: Any):
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
        history = payload.get("history")
        if history is not None and (not isinstance(history, list) or len(history) != len(fens)):
            raise ValueError("'history' must be a list with one entry (or null) per FEN")
        return fens, moves, history

    def handle(self, payload: Any) -> Dict[str, Any]:
        return self.handle_async(payload)()

    def handle_async(self, payload: Any) -> Callable[[], Dict[str, Any]]:
        """Validates the request and starts its evaluation; the returned function waits for the reply.

        Raises ValueError for a bad request (before anything is evaluated).
        """
        fens, moves, history = self._parse(payload)
        if not fens:
            reply = self._reply([])
            return lambda: reply
        # A network trained with histories gets each position as a board with its moves.
        positions: List[Any] = list(fens)
        use_history = history is not None and self.history_aware
        if use_history:
            from gnn_v3 import history_board
            positions = [history_board(f, h) if h is not None else f for f, h in zip(fens, history)]

        keys = None
        cached: List[Optional[Dict[str, Any]]] = [None] * len(fens)
        if self.cache_size:
            keys = [(f, tuple(str(m) for m in moves[i]) if moves is not None else None,
                     json.dumps(history[i], sort_keys=True) if use_history else None) for i, f in enumerate(fens)]
            with self.cache_lock:
                for i, k in enumerate(keys):
                    item = self.cache.get(k)
                    if item is not None:
                        self.cache.move_to_end(k)
                        cached[i] = item
                        self.cache_hits += 1
        todo = [i for i in range(len(fens)) if cached[i] is None]
        wanted = [positions[i] for i in todo]

        if not wanted:
            evaluated = None
        elif self.backend is None:
            batch = self.adapter.build_batch(wanted)  # raises ValueError for malformed FENs
            with self.lock:
                evaluated = self.adapter.evaluate(batch, mc_samples=self.mc_samples)
        else:
            graphs = [self._build_arrays(f) for f in wanted]  # ValueError for malformed FENs, as above
            if self.batcher is not None:
                evaluated = self.batcher.submit(graphs)
            else:
                from relational_fast import collate
                with self.lock:
                    evaluated = self.backend.run(collate(graphs), mc_samples=self.mc_samples)

        def finish() -> Dict[str, Any]:
            results = list(cached)
            if todo:
                probs, values, uncertainty = evaluated.result() if isinstance(evaluated, Future) else evaluated
            for j, i in enumerate(todo):
                item: Dict[str, Any] = {"value": float(values[j]), "uncertainty": float(uncertainty[j])}
                if moves is not None:
                    item["move_priors"] = self.adapter.move_priors(probs[j], [str(m) for m in moves[i]])
                else:
                    item["policy"] = probs[j].tolist()
                results[i] = item
            if keys is not None and todo:
                with self.cache_lock:
                    for i in todo:
                        self.cache[keys[i]] = results[i]
                        self.cache.move_to_end(keys[i])
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            return self._reply(results)

        return finish

    def _reply(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # model_format is new (additive); the other fields are as they always were.
        reply = {"results": results, "policy_dim": self.adapter.policy_dim, "model_format": self.adapter.format}
        if self.history_aware:
            reply["history"] = True  # send histories (only such networks say so)
        return reply


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


def load_served_model(path: Optional[str], device: torch.device, hidden_dim: int = 128) -> GNNAdapter:
    """load_model for either checkpoint format: a 'nags-gnn-v2' or 'nags-gnn-v3' file is served
    with its own graph builder and move encoding; without a file, the untrained legacy model."""
    if path and os.path.exists(path):
        if _V3_IMPORT_ERROR is not None:
            logger.warning("gnn_v3 could not be imported (%s); serving %s as a nags-gnn-v2 checkpoint.",
                           _V3_IMPORT_ERROR, path)
            return GNNAdapterV2(load_model(path, device, hidden_dim), ChessGraph(device=device))
        adapter = load_any_checkpoint(path, device=device)
        logger.info("Loaded %s model from %s (%s)", adapter.format, path, adapter.model.config)
        return adapter
    return GNNAdapterV2(load_model(path, device, hidden_dim), ChessGraph(device=device))


def make_server(service: EvaluationService, host: str, port: int) -> socketserver.ThreadingTCPServer:
    class Handler(socketserver.StreamRequestHandler):
        # TCP_NODELAY: with Nagle's algorithm a reply that follows another one
        # (pipelined requests) waits for the client's ACK of the first, which
        # a client with nothing more to send delays by up to 40 ms.
        disable_nagle_algorithm = True

        def handle(self) -> None:
            if service.batcher is not None:
                return self.handle_pipelined()
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

        def handle_pipelined(self) -> None:
            # Requests are read (and their graphs built) while earlier ones are still being
            # evaluated; replies go out in request order.
            pending: "queue.Queue[Optional[Callable[[], Dict[str, Any]]]]" = queue.Queue()
            broken = threading.Event()

            def writer() -> None:
                while True:
                    finish = pending.get()
                    if finish is None:
                        return
                    try:
                        response = finish()
                    except Exception as e:
                        response = {"error": f"{type(e).__name__}: {e}"}
                    if broken.is_set():
                        continue
                    try:
                        self.wfile.write((json.dumps(response) + "\n").encode('utf-8'))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        broken.set()

            out = threading.Thread(target=writer, daemon=True)
            out.start()
            service.batcher.connected(+1)
            try:
                for raw in self.rfile:
                    if broken.is_set():
                        break
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        pending.put(service.handle_async(json.loads(line.decode('utf-8'))))
                    except Exception as e:
                        error = {"error": f"{type(e).__name__}: {e}"}
                        pending.put(lambda error=error: error)
            except (ConnectionResetError, OSError):
                pass
            finally:
                service.batcher.connected(-1)
                pending.put(None)
                out.join()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server((host, port), Handler)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="NAGS GNN evaluation server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="checkpoint written by training_pipeline.py ('nags-gnn-v2' or 'nags-gnn-v3')")
    parser.add_argument("--mc-samples", type=int, default=5, help="MC-dropout samples for the value uncertainty")
    parser.add_argument("--hidden-dim", type=int, default=128, help="width of the untrained fallback model")
    parser.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    parser.add_argument("--backend", default="auto", choices=["auto", "legacy", "torch", "onnx"],
                        help="how nags-gnn-v3 networks are evaluated: auto = torch (bit-identical to legacy, faster); "
                             "v2 networks always use legacy")
    parser.add_argument("--providers", default=None,
                        help="ONNX Runtime providers for --backend onnx, in order of preference "
                             "(tensorrt,cuda,cpu; default: cuda,cpu on a CUDA device, else cpu)")
    parser.add_argument("--onnx-cache", default=os.path.join("models", "onnx_cache"),
                        help="directory for exported .onnx files and TensorRT engines")
    parser.add_argument("--fp16", action="store_true",
                        help="half precision on the GPU (torch autocast / TensorRT); outputs change slightly")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for the network (0: library default)")
    parser.add_argument("--batch-wait-ms", type=float, default=0.0,
                        help="batch requests arriving within this many ms (v3; 0 = off)")
    parser.add_argument("--max-batch", type=int, default=MAX_BATCH, help="largest batch formed by --batch-wait-ms")
    parser.add_argument("--cache-size", type=int, default=0, help="replies kept for repeated requests (0 = off)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    adapter = load_served_model(args.model, device, args.hidden_dim)
    backend = None
    kind = args.backend
    if kind == "auto":
        kind = "torch" if adapter.format == "nags-gnn-v3" else "legacy"
    if kind != "legacy":
        if adapter.format != "nags-gnn-v3":
            logger.warning("--backend %s serves nags-gnn-v3 networks only; serving %s through the legacy path",
                           kind, adapter.format)
        else:
            from gnn_infer import make_backend

            providers = (args.providers or ("cuda,cpu" if device.type == "cuda" else "cpu")).split(",")
            backend = make_backend(adapter.model, kind, providers=providers, threads=args.threads, fp16=args.fp16,
                                   cache_dir=args.onnx_cache)
            detail = f" ({', '.join(backend.providers)})" if kind == "onnx" else ""
            logger.info("Backend %s%s on %s, %d thread(s)", kind, detail, device, torch.get_num_threads())
    if args.batch_wait_ms > 0 and backend is None:
        logger.warning("--batch-wait-ms needs a nags-gnn-v3 network and a fast backend; requests are not batched")
    service = EvaluationService(adapter, mc_samples=args.mc_samples, backend=backend, cache_size=args.cache_size,
                                batch_wait_ms=args.batch_wait_ms, max_batch=args.max_batch)
    with make_server(service, args.host, args.port) as srv:
        print(f"RPC server listening on tcp://{args.host}:{args.port} (line-delimited JSON over TCP)", flush=True)
        srv.serve_forever()


if __name__ == '__main__':
    main()
