"""How much evaluation the export loses: a trained float network against the integer network the engine runs.

    python tools/nnue/quant_error.py --net nets/my.nnue.ckpt --data data/selfplay_1.bin
    python tools/nnue/quant_error.py --net nets/nags.nnue --data data/selfplay_1.bin

Three evaluations of the same positions, in centipawns for the side to move:

  float      the training checkpoint (.ckpt) as trained,
  rounded    the exported integer weights, computed in floating point (no clipping or rounding of the sums),
  engine     the exported network computed as the engine does (integer sums, truncating divisions).

float - rounded is the error of rounding the weights; rounded - engine is the error of the integer arithmetic;
float - engine is the total. Given an exported .nnue alone only the arithmetic part can be measured (its
float weights are gone); --simulate estimates the weight part by rounding the exported weights once more
(uniform offsets of up to half a step, which is what rounding leaves when the float weights' fractions are
uniform), layer by layer. Reported overall and by output bucket (piece count): mean and percentiles of the
absolute difference, and how often the sign of the evaluation changes.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402

QA, QB, SCALE = train.QA, train.QB, train.SCALE


def load_float(path):
    """The float model of a training checkpoint."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model = train.Nnue(ckpt["hidden"], ckpt.get("buckets", 1), ckpt.get("king_buckets", "none"),
                       ckpt.get("activation", "crelu") == "screlu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


class IntegerNet:
    """An exported network, evaluated in batches: exactly as the engine does (engine) or with the same
    weights in floating point (rounded)."""

    def __init__(self, path):
        ft_w, ft_b, out_w, out_b, info = train.read_net(path)
        kb, mirror, layout = info["king_buckets"], info["mirror"], info["layout"]
        name = "none" if kb == 1 and not mirror else ",".join(str(int(layout[r * 8 + f])) for r in range(8)
                                                               for f in range(4))
        self.king = train.KingBuckets(name, factor=False)
        self.ft_w = torch.from_numpy(np.vstack([ft_w, np.zeros((1, ft_w.shape[1]), np.int32)])).long()
        self.ft_b = torch.from_numpy(ft_b).long()
        self.out_w = torch.from_numpy(out_w).long()
        self.out_b = torch.from_numpy(out_b).long()
        self.screlu = info["screlu"]
        self.buckets = len(out_b)
        self.hidden = len(ft_b)

    def _acc(self, f, ft_w=None, ft_b=None):
        ft_w = self.ft_w if ft_w is None else ft_w
        rows = self.king(f)
        rows = torch.where(rows == self.king.rows, len(ft_w) - 1, rows)
        return (self.ft_b if ft_b is None else ft_b) + ft_w[rows].sum(dim=1)

    def evaluate(self, white, black, stm, bucket):
        """(engine, rounded) centipawns for the side to move; white/black/stm/bucket as train.tensors gives."""
        aw, ab = self._acc(white), self._acc(black)
        s = stm.bool().unsqueeze(1)
        us, them = torch.where(s, ab, aw), torch.where(s, aw, ab)
        w = self.out_w[bucket]
        h = self.hidden
        # The engine: clipped integer activations, integer sums, divisions truncated towards zero.
        a, b = us.clamp(0, QA), them.clamp(0, QA)
        if self.screlu:
            total = torch.div((a * a * w[:, :h]).sum(1) + (b * b * w[:, h:]).sum(1), QA, rounding_mode="trunc")
        else:
            total = (a * w[:, :h]).sum(1) + (b * w[:, h:]).sum(1)
        engine = torch.div((total + self.out_b[bucket]) * SCALE, QA * QB, rounding_mode="trunc")
        return engine.double(), self.float_eval(white, black, stm, bucket)

    def float_eval(self, white, black, stm, bucket, noise=None):
        """The exported weights in floating point; `noise` = (feature transformer, output layer) offsets in
        units of the integer weights, to simulate another rounding."""
        ft_w, ft_b, out_w = self.ft_w.double(), self.ft_b.double(), self.out_w.double()
        if noise is not None:
            ft_w, ft_b, out_w = ft_w + noise[0][0], ft_b + noise[0][1], out_w + noise[1]
        aw, ab = self._acc(white, ft_w, ft_b) / QA, self._acc(black, ft_w, ft_b) / QA
        s = stm.bool().unsqueeze(1)
        fa, fb = torch.where(s, ab, aw).clamp(0, 1), torch.where(s, aw, ab).clamp(0, 1)
        if self.screlu:
            fa, fb = fa * fa, fb * fb
        wd = out_w[bucket] / QB
        h = self.hidden
        out = (fa * wd[:, :h]).sum(1) + (fb * wd[:, h:]).sum(1) + self.out_b[bucket].double() / (QA * QB)
        return out * SCALE

    def rounding_noise(self, gen, layer):
        """Uniform offsets of up to half an integer step, as rounding a float weight leaves, on one layer."""
        u = lambda t: torch.rand(t.shape, generator=gen, dtype=torch.float64) - 0.5  # noqa: E731
        zero = (torch.zeros(self.ft_w.shape, dtype=torch.float64), torch.zeros(self.ft_b.shape, dtype=torch.float64))
        if layer == "feature transformer":
            ft = (u(self.ft_w), u(self.ft_b))
            ft[0][-1] = 0  # the padding row
            return ft, torch.zeros(self.out_w.shape, dtype=torch.float64)
        return zero, u(self.out_w)


def positions(paths, count, seed):
    """Up to `count` records drawn evenly at random from the files."""
    recs = [np.memmap(p, np.uint8, "r").reshape(-1, train.RECORD_BYTES) for p in paths]
    total = sum(len(r) for r in recs)
    if total == 0:
        sys.exit("no positions in the data")
    pick = np.sort(np.random.default_rng(seed).choice(total, min(count, total), replace=False))
    out, offset = [], 0
    for r in recs:
        here = pick[(pick >= offset) & (pick < offset + len(r))] - offset
        out.append(np.asarray(r[here]))
        offset += len(r)
    return np.concatenate(out)


def _summary(diff, base, other):
    d = diff.abs()
    q = torch.quantile(d, torch.tensor([0.5, 0.9, 0.99], dtype=d.dtype)).tolist() if len(d) else [0, 0, 0]
    flips = ((base > 0) & (other < 0)) | ((base < 0) & (other > 0))
    return {"positions": len(d), "mean": float(d.mean()) if len(d) else 0.0, "p50": q[0], "p90": q[1],
            "p99": q[2], "max": float(d.max()) if len(d) else 0.0, "bias": float(diff.mean()) if len(d) else 0.0,
            "sign_changes": int(flips.sum())}


def measure(net_path, data_paths, count=20000, seed=1, batch=8192, simulate=0):
    """The report as a dict (see the module docstring); with `simulate`, also that many simulated roundings
    of each layer of the exported network (an estimate of the weight rounding error when the float weights
    are not at hand)."""
    model = None
    if net_path.endswith(".ckpt"):
        model = load_float(net_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".nnue", delete=False)
        tmp.close()
        try:
            train.export(model, tmp.name)
            net = IntegerNet(tmp.name)
        finally:
            os.unlink(tmp.name)
    else:
        net = IntegerNet(net_path)
    recs = positions(data_paths, count, seed)
    groups = []  # (white, black, stm, bucket) per batch
    for start in range(0, len(recs), batch):
        rec = torch.from_numpy(recs[start : start + batch])
        feats, stm, score, result = train.decode(rec)
        groups.append(train.tensors(feats, stm, score, result, torch.arange(len(rec)), 1.0, net.buckets)[:4])
    with torch.no_grad():
        engine, rounded = (torch.cat(x) for x in zip(*(net.evaluate(*g) for g in groups)))
        flt = torch.cat([model(*g).double() * SCALE for g in groups]) if model is not None else None
        simulated = {}
        gen = torch.Generator().manual_seed(seed)
        for layer in ("feature transformer", "output layer") if simulate else ():
            trials = []
            for _ in range(simulate):
                noise = net.rounding_noise(gen, layer)
                moved = torch.cat([net.float_eval(*g, noise) for g in groups])
                trials.append(_summary(moved - rounded, rounded, moved))
            simulated[layer] = trials
    buckets = torch.cat([g[3] for g in groups])
    pairs = {"arithmetic (rounded - engine)": (rounded, engine)}
    if flt is not None:
        pairs = {"total (float - engine)": (flt, engine), "weights (float - rounded)": (flt, rounded), **pairs}
    report = {"network": net_path, "positions": len(engine), "data": [os.path.basename(p) for p in data_paths],
              "seed": seed, "eval_cp": _summary(engine, engine * 0, engine * 0)}
    for name, (x, y) in pairs.items():
        report[name] = {"overall": _summary(x - y, x, y),
                        "by_output_bucket": [_summary((x - y)[buckets == i], x[buckets == i], y[buckets == i])
                                             for i in range(net.buckets)]}
    if simulate:
        report["simulated rounding"] = simulated
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="a training checkpoint (.ckpt) or an exported network (.nnue)")
    ap.add_argument("--data", nargs="+", required=True, help="32-byte record files (.bin) to draw positions from")
    ap.add_argument("--positions", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--simulate", type=int, default=0, metavar="N",
                    help="also N simulated roundings of each layer (uniform offsets of up to half a step): an "
                         "estimate of the weight rounding error for an exported network without its checkpoint")
    ap.add_argument("--json", help="also write the report here")
    args = ap.parse_args(argv)
    report = measure(args.net, train.expand_paths(args.data), args.positions, args.seed, simulate=args.simulate)
    ev = report["eval_cp"]
    print(f"{report['network']}: {report['positions']} positions; |eval| mean {ev['mean']:.0f} cp, "
          f"p50 {ev['p50']:.0f}, p99 {ev['p99']:.0f}")
    for name, part in report.items():
        if not isinstance(part, dict) or "overall" not in part:
            continue
        o = part["overall"]
        print(f"{name}: mean {o['mean']:.2f} cp, p50 {o['p50']:.2f}, p90 {o['p90']:.2f}, p99 {o['p99']:.2f}, "
              f"max {o['max']:.2f}, bias {o['bias']:+.2f}, sign changes {o['sign_changes']}")
        cells = [f"{i}: {b['mean']:.2f}" + (f"/{b['p99']:.1f}" if b["positions"] else "")
                 for i, b in enumerate(part["by_output_bucket"]) if b["positions"]]
        print("  by output bucket (mean/p99): " + ", ".join(cells))
    for layer, trials in report.get("simulated rounding", {}).items():
        each = lambda key, fmt: ", ".join(format(t[key], fmt) for t in trials)  # noqa: E731
        print(f"simulated rounding of the {layer} ({len(trials)} trials): mean {each('mean', '.2f')} cp; "
              f"p99 {each('p99', '.1f')}; bias {each('bias', '+.2f')}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
