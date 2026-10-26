"""tools/nnue/train.py: the best epoch kept (--keep-best) and early stopping (--early-stop), chosen by the
validation loss or an evaluation set's (--select-by)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import train  # noqa: E402
from test_nnue import _dataset  # noqa: E402

COMMON = ["--hidden", "8", "--batch", "64", "--threads", "1", "--device", "cpu", "--buckets", "8", "--val", "0.2"]


def _data(tmp_path, n=600, seed=0):
    _dataset(tmp_path / "d.txt", n=n, seed=seed)
    pack.main([str(tmp_path / "d.txt")])
    return str(tmp_path / "d.bin")


def _flipped(src, dst):
    """The same positions with the opposite scores and results: a set the training makes worse."""
    rec = np.fromfile(src, np.uint8).reshape(-1, 32).copy()
    score = rec[:, 24:26].copy().view("<i2").ravel()
    rec[:, 24:26] = (-score).astype("<i2").view(np.uint8).reshape(-1, 2)
    rec[:, 26] = 2 - rec[:, 26]
    rec.tofile(dst)
    return str(dst)


def test_the_best_epoch_is_kept_and_is_that_epoch_s_network(tmp_path):
    data = _data(tmp_path)
    out = tmp_path / "n.nnue"
    train.main(["--data", data, "--out", str(out), "--epochs", "5", "--keep-best"] + COMMON)
    m = json.loads(Path(str(out) + ".json").read_text())
    vals = [h["val"] for h in m["history"]]
    assert m["best"]["epoch"] == 1 + int(np.argmin(vals)) and m["best"]["loss"] == min(vals)
    assert m["best"]["file"] == "n.best.nnue" and m["best"]["select_by"] == "val"
    # The kept file is the network of that epoch: the same run stopped there writes it.
    part = tmp_path / "p.nnue"
    train.main(["--data", data, "--out", str(part), "--epochs", "5", "--stop-after", str(m["best"]["epoch"])]
               + COMMON)
    assert (tmp_path / "n.best.nnue").read_bytes() == part.read_bytes()


def test_early_stop_by_an_evaluation_set(tmp_path, capsys):
    data = _data(tmp_path)
    bad = _flipped(data, tmp_path / "flipped.bin")
    out = tmp_path / "n.nnue"
    args = ["--data", data, "--out", str(out), "--epochs", "8", "--eval-set", bad, "--select-by", "eval:flipped.bin",
            "--early-stop", "2", "--keep-best", "--eval-overlap-limit", "1"] + COMMON  # drawn 0-score records repeat
    train.main(args)
    m = json.loads(Path(str(out) + ".json").read_text())
    losses = [h["eval"]["flipped.bin"]["loss"] for h in m["history"]]
    assert m["epochs_done"] == m["best"]["epoch"] + 2 < 8  # stopped two epochs after the best
    assert losses[m["best"]["epoch"] - 1] == min(losses) and "early stop after epoch" in capsys.readouterr().out
    # Resuming a run that stopped early trains no further.
    train.main(args + ["--resume"])
    assert json.loads(Path(str(out) + ".json").read_text())["epochs_done"] == m["epochs_done"]


def test_without_the_options_nothing_changes(tmp_path):
    data = _data(tmp_path)
    train.main(["--data", data, "--out", str(tmp_path / "a.nnue"), "--epochs", "2"] + COMMON)
    train.main(["--data", data, "--out", str(tmp_path / "b.nnue"), "--epochs", "2", "--keep-best"] + COMMON)
    assert (tmp_path / "a.nnue").read_bytes() == (tmp_path / "b.nnue").read_bytes()  # training is unchanged
    assert not (tmp_path / "a.best.nnue").exists()
    with pytest.raises(SystemExit, match="select-by"):
        train.main(["--data", data, "--out", str(tmp_path / "c.nnue"), "--epochs", "1", "--select-by", "eval:x.bin",
                    "--keep-best"] + COMMON)
