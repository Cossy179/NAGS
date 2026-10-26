"""tools/nnue/train.py --export-bias-correction: each exported output bias corrected for the rounding of its
bucket's output weights, on positions not trained on; nothing else in the file or the checkpoint changes."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import quant_error  # noqa: E402
import train  # noqa: E402
from test_nnue import _dataset  # noqa: E402

COMMON = ["--hidden", "16", "--batch", "64", "--threads", "1", "--device", "cpu", "--buckets", "8",
          "--king-buckets", "4", "--activation", "screlu", "--epochs", "3", "--val", "0.25"]


def test_the_correction_takes_the_mean_rounding_shift_off(tmp_path):
    _dataset(tmp_path / "d.txt", n=800)
    pack.main([str(tmp_path / "d.txt")])
    data = str(tmp_path / "d.bin")
    plain, fixed = tmp_path / "plain.nnue", tmp_path / "fixed.nnue"
    train.main(["--data", data, "--out", str(plain)] + COMMON)
    train.main(["--data", data, "--out", str(fixed), "--export-bias-correction", "200"] + COMMON)
    a, b = train.read_net(str(plain)), train.read_net(str(fixed))
    for i in range(3):  # feature transformer and output weights: the same
        assert np.array_equal(a[i], b[i])
    assert not np.array_equal(a[3], b[3])  # the output biases moved
    assert (Path(str(plain) + ".ckpt").read_bytes() != b"" and
            torch.load(str(plain) + ".ckpt", weights_only=False)["model"]["out.bias"].equal(
                torch.load(str(fixed) + ".ckpt", weights_only=False)["model"]["out.bias"]))  # training unchanged
    assert json.loads(Path(str(fixed) + ".json").read_text())["export"]["bias_correction_positions"] == 200
    # On the positions it was measured on, the rounded weights' mean shift per bucket is gone: the float
    # network and the exported one (in floating point, so without the engine's truncation) agree on average.
    model = train.Nnue(16, 8, "4", True)
    model.load_state_dict(torch.load(str(fixed) + ".ckpt", weights_only=False)["model"])
    source = train.BinData([data], 0.25, torch.device("cpu"))
    w, bl, s, k = train.sample_positions(source, 200, 8, torch.device("cpu"))
    with torch.no_grad():
        flt = model(w, bl, s, k).double() * train.SCALE
    for net in (plain, fixed):
        rounded = quant_error.IntegerNet(str(net)).float_eval(w, bl, s, k)
        shift = [float((flt - rounded)[k == i].mean()) for i in range(8) if (k == i).any()]
        if net == plain:
            before = shift
        else:
            after = shift
    # What is left is the rounding of the bias itself (1 / (QA * QB) output units, 0.025 cp).
    assert all(abs(x) < 0.03 for x in after) and max(abs(x) for x in before) > 10 * max(abs(x) for x in after)
