"""tools/nnue/train.py --sync-every (fewer GPU synchronisations, the same network) and --profile."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import train  # noqa: E402
from test_nnue import _dataset  # noqa: E402

COMMON = ["--hidden", "8", "--batch", "32", "--threads", "1", "--device", "cpu", "--epochs", "2", "--buckets", "8",
          "--king-buckets", "4", "--activation", "screlu"]


def test_reading_the_loss_less_often_trains_the_same_network(tmp_path):
    _dataset(tmp_path / "d.txt")
    pack.main([str(tmp_path / "d.txt")])
    data = str(tmp_path / "d.bin")
    train.main(["--data", data, "--out", str(tmp_path / "every.nnue")] + COMMON)
    train.main(["--data", data, "--out", str(tmp_path / "seven.nnue"), "--sync-every", "7"] + COMMON)
    assert (tmp_path / "every.nnue").read_bytes() == (tmp_path / "seven.nnue").read_bytes()


def test_profile_measures_and_writes_nothing(tmp_path, capsys):
    _dataset(tmp_path / "d.txt")
    pack.main([str(tmp_path / "d.txt")])
    out = tmp_path / "p.nnue"
    train.main(["--data", str(tmp_path / "d.bin"), "--out", str(out), "--profile", "6", "--sync-every", "3"] + COMMON)
    text = capsys.readouterr().out
    assert "median ms per batch" in text and "reading it every 3" in text
    assert not out.exists() and not Path(str(out) + ".ckpt").exists() and not Path(str(out) + ".json").exists()


def test_eval_set_reports_by_bucket_and_changes_nothing(tmp_path):
    import json

    import numpy as np
    import torch

    _dataset(tmp_path / "d.txt")
    _dataset(tmp_path / "e.txt", n=150, seed=3)
    pack.main([str(tmp_path / "d.txt"), str(tmp_path / "e.txt")])
    data, ev = str(tmp_path / "d.bin"), str(tmp_path / "e.bin")
    train.main(["--data", data, "--out", str(tmp_path / "plain.nnue")] + COMMON)
    train.main(["--data", data, "--out", str(tmp_path / "eval.nnue"), "--eval-set", ev] + COMMON)
    assert (tmp_path / "plain.nnue").read_bytes() == (tmp_path / "eval.nnue").read_bytes()  # training unchanged
    m = json.loads((tmp_path / "eval.nnue.json").read_text())
    rep = m["history"][-1]["eval"]["e.bin"]
    assert m["eval_sets"][0]["sha256"] == train.file_sha256(ev, cache=False)
    assert rep["positions"] == 150 == sum(rep["output_bucket_positions"]) == sum(rep["king_bucket_positions"])
    assert len(rep["by_output_bucket"]) == 8 and len(rep["by_king_bucket"]) == 4
    # The overall loss is the mean squared error of the trained network on those records.
    model = train.Nnue(8, 8, "4", True)
    train.init_from(model, str(tmp_path / "eval.nnue.ckpt"))
    rec = torch.from_numpy(np.fromfile(ev, np.uint8).reshape(-1, 32))
    feats, stm, score, result = train.decode(rec)
    w, b, s, k, t = train.tensors(feats, stm, score, result, torch.arange(len(rec)), 0.75, 8)
    with torch.no_grad():
        mse = float(((torch.sigmoid(model(w, b, s, k)) - t) ** 2).mean())
    assert abs(rep["loss"] - mse) < 1e-6
    weighted = sum(l * c for l, c in zip(rep["by_output_bucket"], rep["output_bucket_positions"]) if l is not None)
    assert abs(weighted / 150 - rep["loss"]) < 1e-9
