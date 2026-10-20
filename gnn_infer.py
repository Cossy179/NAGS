"""Inference backends for 'nags-gnn-v3' networks (gnn_v3.RelationalGNNEvaluator).

The checkpoint format, the graph (relational-v1) and the meaning of the
outputs are those of gnn_v3; this module only computes them faster, from
the precomputed arrays of relational_fast.collate:

  * TorchBackendV3: the model's own layers, called in the order
    RelationalGNNEvaluator.evaluate calls them, without switching the
    module between train and eval mode on every call (MC dropout uses
    F.dropout, which draws the same random numbers nn.Dropout does). The
    results are bit-identical to evaluate(), Monte-Carlo samples included
    (tests/python/test_gnn_infer.py). Optional half precision on CUDA
    (autocast; not identical, opt-in).
  * OnnxBackendV3: the network exported to ONNX (export_onnx) and run by
    ONNX Runtime with the execution providers asked for (CPU, CUDA,
    TensorRT). The graph has no data-dependent shapes or boolean
    indexing (square/global nodes come as gather indices), so the same
    file works with every provider. Results agree with torch to about
    1e-6; MC-dropout samples are drawn with numpy, so their values
    differ from torch's draws while following the same distribution.

Both return what evaluate() returns: (policy probabilities [B, 4228] over
the legal moves, value mean [B], value std over MC-dropout samples [B]),
as CPU tensors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from gnn_v3 import RelationalGNNEvaluator
from policy_encoding import POLICY_V3_DIM
from relational_fast import BatchArrays

logger = logging.getLogger("gnn_infer")

# Bumped whenever the exported graph changes, so cached .onnx files are rebuilt.
ONNX_EXPORT_VERSION = 1
ONNX_INPUTS = ("x", "src", "dst", "edge_type", "inv_deg", "square_idx", "global_idx", "batch", "node_count", "legal")
ONNX_OUTPUTS = ("policy", "wdl", "pooled")

Result = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _wdl_layers(model: RelationalGNNEvaluator) -> Tuple[float, nn.Linear, nn.Linear]:
    """(dropout p, first linear, second linear) of the WDL head: Dropout, Linear, ReLU, Dropout, Linear."""
    head = model.wdl_head
    kinds = [type(m) for m in head]
    if kinds != [nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear] or head[0].p != head[3].p:
        raise ValueError(f"unexpected WDL head layout {kinds}")
    return head[0].p, head[1], head[4]


def _mask_policy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """evaluate()'s legal masking: rows without a legal move are left unmasked."""
    any_legal = legal.any(dim=1, keepdim=True)
    usable = legal & any_legal
    return logits.masked_fill(~usable & any_legal, float("-inf"))


class TorchBackendV3:
    """RelationalGNNEvaluator.evaluate on relational_fast batches, bit for bit."""

    name = "torch"

    def __init__(self, model: RelationalGNNEvaluator, fp16: bool = False):
        self.model = model.eval()
        self.device = model.device
        self.p, self.lin1, self.lin2 = _wdl_layers(model)
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        if fp16 and not self.fp16:
            logger.warning("--fp16 needs a CUDA device; computing in float32")
        # Reduced precision: torch.autocast on this device with this dtype (the tests use the CPU's bfloat16).
        self.autocast_device = "cuda"
        self.autocast_dtype = torch.float16

    @staticmethod
    def _conv_autocast(conv: nn.Module, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                       edge_type: torch.Tensor, inv_deg: torch.Tensor) -> torch.Tensor:
        # gnn_v3.RelationalConv.forward with the edge coefficients in the activations' dtype:
        # under autocast `hs * coef` would be float32 and index_add_ into a half tensor refuses it.
        hs = h.index_select(0, src)
        coef = (conv.comp.index_select(0, edge_type) * inv_deg.unsqueeze(1)).to(h.dtype)
        aggs = [torch.zeros_like(h).index_add_(0, dst, hs * coef[:, b:b + 1]) for b in range(conv.num_bases)]
        return conv.root(h) + conv.basis(torch.cat(aggs, dim=1))

    def _tensors(self, b: BatchArrays) -> Dict[str, torch.Tensor]:
        dev = self.device
        t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        return {"x": t(b.x), "src": t(b.src), "dst": t(b.dst), "edge_type": t(b.edge_type),
                "inv_deg": t(b.inv_deg), "batch": t(b.batch), "square_idx": t(b.square_idx),
                "global_idx": t(b.global_idx), "legal": t(b.legal)}

    def _wdl(self, pooled: torch.Tensor, dropout: bool) -> torch.Tensor:
        # model.wdl_head(pooled), with the dropout layers on (MC samples) or off (eval).
        h = F.dropout(pooled, self.p, training=True) if dropout else pooled
        h = torch.relu(self.lin1(h))
        h = F.dropout(h, self.p, training=True) if dropout else h
        return self.lin2(h)

    @torch.inference_mode()
    def run(self, b: BatchArrays, mc_samples: int = 5) -> Result:
        m = self.model
        g = self._tensors(b)
        with torch.autocast(self.autocast_device, dtype=self.autocast_dtype, enabled=self.fp16):
            h = m.proj_in(g["x"])
            for i, conv in enumerate(m.convs):
                args = (h, g["src"], g["dst"], g["edge_type"], g["inv_deg"])
                c = self._conv_autocast(conv, *args) if self.fp16 else conv(*args)
                h = m.norms[i](h + m.act(c))
                if m.ffns is not None:
                    h = m.ffn_norms[i](h + m.ffns[i](h))
            glob = h.index_select(0, g["global_idx"])
            n = glob.size(0)
            squares = h.index_select(0, g["square_idx"]).view(n, 64, -1)
            pooled = torch.cat([glob, global_mean_pool(h, g["batch"], size=n)], dim=1)
            logits = m._policy(squares, glob)
        logits = _mask_policy(logits.float(), g["legal"])
        probs = F.softmax(logits, dim=-1)
        pooled = pooled.float()
        samples = max(1, mc_samples)
        with torch.autocast(self.autocast_device, dtype=self.autocast_dtype, enabled=self.fp16):
            values = torch.stack([m.wdl_value(self._wdl(pooled, samples > 1).float()) for _ in range(samples)])
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False) if samples > 1 else torch.zeros_like(mean)
        return probs.cpu(), mean.cpu(), std.cpu()


class _ExportModule(nn.Module):
    """The network as a function of plain index tensors (what export_onnx writes)."""

    def __init__(self, model: RelationalGNNEvaluator):
        super().__init__()
        self.m = model

    def forward(self, x, src, dst, edge_type, inv_deg, square_idx, global_idx, batch, node_count, legal):
        m = self.m
        h = m.proj_in(x)
        for i, conv in enumerate(m.convs):
            hs = h.index_select(0, src)
            coef = conv.comp.index_select(0, edge_type) * inv_deg.unsqueeze(1)          # [E, bases]
            msg = (hs.unsqueeze(1) * coef.unsqueeze(2)).reshape(hs.size(0), -1)        # [E, bases * H]
            agg = torch.zeros(h.size(0), msg.size(1), dtype=h.dtype, device=h.device).index_add(0, dst, msg)
            h = m.norms[i](h + m.act(conv.root(h) + conv.basis(agg)))
            if m.ffns is not None:
                h = m.ffn_norms[i](h + m.ffns[i](h))
        glob = h.index_select(0, global_idx)
        n = glob.size(0)
        squares = h.index_select(0, square_idx).view(n, 64, -1)
        total = torch.zeros(n, h.size(1), dtype=h.dtype, device=h.device).index_add(0, batch, h)
        pooled = torch.cat([glob, total / node_count.unsqueeze(1)], dim=1)
        probs = F.softmax(_mask_policy(m._policy(squares, glob), legal), dim=-1)
        return probs, m.wdl_head(pooled), pooled


def _example_inputs(model: RelationalGNNEvaluator) -> Tuple[torch.Tensor, ...]:
    import chess

    from relational_fast import build_batch
    b = build_batch([chess.STARTING_FEN, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"])
    return tuple(torch.from_numpy(getattr(b, name)) for name in ONNX_INPUTS)


def export_onnx(model: RelationalGNNEvaluator, path: str, opset: int = 18) -> str:
    """Writes the network (eval mode, float32) to `path` as ONNX with dynamic batch, node and edge counts."""
    model = model.eval()
    cpu_model = RelationalGNNEvaluator(**model.config, device=torch.device("cpu"))
    cpu_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    module = _ExportModule(cpu_model.eval()).eval()
    nodes, edges, graphs, squares = (torch.export.Dim("nodes"), torch.export.Dim("edges"),
                                     torch.export.Dim("graphs"), torch.export.Dim("squares"))
    dims = {"x": {0: nodes}, "src": {0: edges}, "dst": {0: edges}, "edge_type": {0: edges}, "inv_deg": {0: edges},
            "square_idx": {0: squares}, "global_idx": {0: graphs}, "batch": {0: nodes}, "node_count": {0: graphs},
            "legal": {0: graphs}}
    tmp = path + ".tmp"
    with torch.no_grad():
        torch.onnx.export(module, _example_inputs(cpu_model), tmp, input_names=list(ONNX_INPUTS),
                          output_names=list(ONNX_OUTPUTS), dynamic_shapes=dims, dynamo=True, opset_version=opset,
                          external_data=False, verbose=False)
    os.replace(tmp, path)
    return path


def _state_digest(model: RelationalGNNEvaluator) -> str:
    h = hashlib.sha256(repr(sorted(model.config.items())).encode())
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:20]


def cached_onnx(model: RelationalGNNEvaluator, cache_dir: Optional[str]) -> str:
    """The ONNX file of this exact network (by a digest of its weights), exported on first use."""
    directory = cache_dir or os.path.join(tempfile.gettempdir(), "nags_onnx_cache")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"gnn_v3-{_state_digest(model)}-e{ONNX_EXPORT_VERSION}.onnx")
    if not os.path.exists(path):
        logger.info("Exporting the network to %s", path)
        export_onnx(model, path)
    return path


# Execution providers by short name, in the order they are tried.
PROVIDERS = {"tensorrt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}


def provider_list(names: Sequence[str], threads: int, fp16: bool, cache_dir: Optional[str]) -> List[Any]:
    """ONNX Runtime providers (with options) for short names such as ['tensorrt', 'cuda', 'cpu']."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    out: List[Any] = []
    for name in names:
        full = PROVIDERS.get(name.strip().lower(), name.strip())
        if full not in available:
            logger.warning("ONNX Runtime provider %s is not available here (available: %s)", full,
                           ", ".join(sorted(available)))
            continue
        if full == "TensorrtExecutionProvider":
            trt_cache = os.path.join(cache_dir or tempfile.gettempdir(), "trt_engines")
            os.makedirs(trt_cache, exist_ok=True)
            # Dynamic shapes need an optimisation profile: from one position up to MAX_BATCH (256)
            # positions of at most 97 nodes and 1500 edges each.
            lo = "x:65x32,src:1,dst:1,edge_type:1,inv_deg:1,square_idx:64,global_idx:1,batch:65,node_count:1,legal:1x4228"
            opt = ("x:1600x32,src:6400,dst:6400,edge_type:6400,inv_deg:6400,square_idx:1024,global_idx:16,"
                   "batch:1600,node_count:16,legal:16x4228")
            hi = ("x:24832x32,src:384000,dst:384000,edge_type:384000,inv_deg:384000,square_idx:16384,"
                  "global_idx:256,batch:24832,node_count:256,legal:256x4228")
            out.append((full, {"trt_fp16_enable": bool(fp16), "trt_engine_cache_enable": True,
                               "trt_engine_cache_path": trt_cache, "trt_profile_min_shapes": lo,
                               "trt_profile_opt_shapes": opt, "trt_profile_max_shapes": hi}))
        elif full == "CPUExecutionProvider":
            out.append(full)
        else:
            out.append(full)
    if not any((p[0] if isinstance(p, tuple) else p) == "CPUExecutionProvider" for p in out):
        out.append("CPUExecutionProvider")
    return out


class OnnxBackendV3:
    """The exported network run by ONNX Runtime."""

    name = "onnx"

    def __init__(self, model: RelationalGNNEvaluator, providers: Sequence[str] = ("cpu",), threads: int = 1,
                 fp16: bool = False, cache_dir: Optional[str] = None, onnx_path: Optional[str] = None, seed: int = 0):
        import onnxruntime as ort

        self.path = onnx_path or cached_onnx(model, cache_dir)
        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.path, opts, providers=provider_list(providers, threads, fp16,
                                                                                     cache_dir))
        self.providers = self.session.get_providers()
        p, lin1, lin2 = _wdl_layers(model)
        self.p = p
        self.w1 = lin1.weight.detach().cpu().numpy().T.astype(np.float32)
        self.b1 = lin1.bias.detach().cpu().numpy().astype(np.float32)
        self.w2 = lin2.weight.detach().cpu().numpy().T.astype(np.float32)
        self.b2 = lin2.bias.detach().cpu().numpy().astype(np.float32)
        self.rng = np.random.default_rng(seed)
        self.rng_lock = threading.Lock()

    @staticmethod
    def _value(wdl_logits: np.ndarray) -> np.ndarray:
        z = wdl_logits - wdl_logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        return p[..., 0] - p[..., 2]

    def run(self, b: BatchArrays, mc_samples: int = 5) -> Result:
        feed = {name: getattr(b, name) for name in ONNX_INPUTS}
        probs, wdl, pooled = self.session.run(None, feed)
        samples = max(1, mc_samples)
        if samples == 1:
            mean = self._value(wdl)
            std = np.zeros_like(mean)
        else:
            keep = 1.0 - self.p
            with self.rng_lock:
                m1 = self.rng.random((samples,) + pooled.shape) < keep
                m2 = self.rng.random((samples, pooled.shape[0], self.w1.shape[1])) < keep
            scale = np.float32(1.0 / keep) if keep > 0 else np.float32(0.0)
            h = np.maximum((pooled[None] * m1 * scale) @ self.w1 + self.b1, 0.0)
            v = self._value((h * m2 * scale) @ self.w2 + self.b2)  # [S, B]
            mean = v.mean(axis=0)
            std = v.std(axis=0)
        return (torch.from_numpy(np.ascontiguousarray(probs)), torch.from_numpy(mean.astype(np.float32)),
                torch.from_numpy(std.astype(np.float32)))


def make_backend(model: RelationalGNNEvaluator, kind: str = "torch", **options: Any):
    """'torch' (bit-identical to evaluate) or 'onnx' (ONNX Runtime; options: providers, threads, fp16, cache_dir)."""
    if kind == "torch":
        return TorchBackendV3(model, fp16=options.get("fp16", False))
    if kind == "onnx":
        return OnnxBackendV3(model, providers=options.get("providers", ("cpu",)), threads=options.get("threads", 1),
                             fp16=options.get("fp16", False), cache_dir=options.get("cache_dir"),
                             onnx_path=options.get("onnx_path"), seed=options.get("seed", 0))
    raise ValueError(f"unknown backend {kind!r} (expected 'torch' or 'onnx')")


__all__ = ["TorchBackendV3", "OnnxBackendV3", "export_onnx", "cached_onnx", "make_backend", "provider_list",
           "POLICY_V3_DIM", "ONNX_EXPORT_VERSION"]
