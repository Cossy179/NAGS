"""tools/nnue/train.py: a resumed run keeps its evaluation setup (the evaluation sets by content, --select-by,
--early-stop, --keep-best), and an earlier run's files are never overwritten without --resume."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import train  # noqa: E402
from test_train_select import _data, _flipped  # noqa: E402

# The synthetic positions repeat, so any evaluation set shares them with the training data: no overlap limit.
COMMON = ["--hidden", "8", "--batch", "64", "--threads", "1", "--device", "cpu", "--buckets", "8", "--val", "0.2",
          "--eval-overlap-limit", "1"]


def _setup(tmp_path):
    data = _data(tmp_path)
    ev = _flipped(data, tmp_path / "ev.bin")
    out = str(tmp_path / "n.nnue")
    base = ["--data", data, "--out", out, "--epochs", "4", "--eval-set", ev, "--select-by", "eval:ev.bin"] + COMMON
    train.main(base + ["--stop-after", "2"])
    return data, ev, out, base


def _files(out):
    return {p: Path(p).read_bytes() for p in (out, out + ".ckpt", out + ".json") if Path(p).exists()}


def test_a_resume_keeps_its_evaluation_setup(tmp_path):
    data, ev, out, base = _setup(tmp_path)
    before = _files(out)
    other = _flipped(data, tmp_path / "ev2.bin")
    for argv, message in [
        (base + ["--keep-best"], "--keep-best was False, now True"),
        (base + ["--early-stop", "2"], "--early-stop was 0, now 2"),
        ([a if a != "eval:ev.bin" else "val" for a in base], "--select-by was 'eval:ev.bin', now 'val'"),
        (["--data", data, "--out", out, "--epochs", "4"] + COMMON, "--eval-set ev.bin was evaluated, now it is not"),
        (base + ["--eval-set", other], "--eval-set ev2.bin was not evaluated before"),
    ]:
        with pytest.raises(SystemExit, match=re.escape(message)):
            train.main(argv + ["--resume"])
    # Another file of the same name: the history's results would mix two sets under one name.
    rec = np.fromfile(ev, np.uint8).reshape(-1, 32)
    rec[:, 26] = 1  # every result a draw
    rec.tofile(ev)
    with pytest.raises(SystemExit, match="--eval-set ev.bin has other content"):
        train.main(base + ["--resume"])
    assert _files(out) == before  # the refused runs wrote nothing


def test_the_same_setup_resumes_to_the_uninterrupted_run(tmp_path):
    data, ev, out, base = _setup(tmp_path)
    train.main(base + ["--resume"])
    whole = str(tmp_path / "whole.nnue")
    train.main([a if a != out else whole for a in base])
    assert Path(out).read_bytes() == Path(whole).read_bytes()
    m = json.loads(Path(out + ".json").read_text())
    assert m["selection"] == {"select_by": "eval:ev.bin", "early_stop": 0, "keep_best": False}
    assert m["eval_sets"][0]["sha256"] == train.file_sha256(ev, cache=False)
    ckpt = torch.load(out + ".ckpt", weights_only=False)
    assert ckpt["evaluation"]["eval_sets"] == [{"name": "ev.bin", "sha256": m["eval_sets"][0]["sha256"]}]


def test_older_checkpoints_resume_only_without_evaluation_results(tmp_path):
    data = _data(tmp_path)
    out = str(tmp_path / "n.nnue")
    plain = ["--data", data, "--out", out, "--epochs", "3"] + COMMON
    train.main(plain + ["--stop-after", "1"])
    ckpt = torch.load(out + ".ckpt", weights_only=False)
    del ckpt["evaluation"]  # as written before the evaluation setup was recorded
    torch.save(ckpt, out + ".ckpt")
    train.main(plain + ["--resume", "--stop-after", "1"])  # none recorded, none used: resumes
    with pytest.raises(SystemExit, match="--keep-best was False, now True"):
        train.main(plain + ["--resume", "--keep-best"])
    # With evaluation-set results in its history it cannot be checked, so it is not resumed.
    ckpt = torch.load(out + ".ckpt", weights_only=False)
    del ckpt["evaluation"]
    ckpt["history"][0]["eval"] = {"ev.bin": {"loss": 0.1}}
    torch.save(ckpt, out + ".ckpt")
    with pytest.raises(SystemExit, match="not which files they came from"):
        train.main(plain + ["--resume"])


def test_two_evaluation_sets_of_the_same_name_are_refused(tmp_path):
    data = _data(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = _flipped(data, tmp_path / "a" / "ev.bin"), _flipped(data, tmp_path / "b" / "ev.bin")
    with pytest.raises(SystemExit, match="two files of the same name"):
        train.main(["--data", data, "--out", str(tmp_path / "n.nnue"), "--epochs", "1", "--eval-set", a,
                    "--eval-set", b] + COMMON)


def test_an_earlier_run_is_never_overwritten(tmp_path):
    data = _data(tmp_path)
    out = str(tmp_path / "n.nnue")
    argv = ["--data", data, "--out", out, "--epochs", "2", "--keep-best"] + COMMON
    train.main(argv)
    before = _files(out)
    best = Path(tmp_path / "n.best.nnue").read_bytes()
    with pytest.raises(SystemExit, match="already exist.*--resume"):
        train.main(argv)  # the same command again, without --resume
    with pytest.raises(SystemExit, match="already exist"):
        train.main(argv[:argv.index("--epochs") + 1] + ["3"] + argv[argv.index("--epochs") + 2:])
    assert _files(out) == before and Path(tmp_path / "n.best.nnue").read_bytes() == best
    # --resume continues from the checkpoint (here: already finished, nothing to do).
    train.main(argv + ["--resume"])
    assert _files(out)[out] == before[out]
    # Files without their checkpoint cannot be resumed.
    Path(out + ".ckpt").unlink()
    with pytest.raises(SystemExit, match="without the checkpoint"):
        train.main(argv + ["--resume"])
    # Any one of the run's files is enough to refuse: here only the best network is left.
    for p in (out, out + ".json"):
        Path(p).unlink()
    with pytest.raises(SystemExit, match="n.best.nnue already exist"):
        train.main(argv)
    # --profile writes nothing, so it may name an existing run.
    train.main(argv + ["--profile", "2"])
    assert Path(tmp_path / "n.best.nnue").read_bytes() == best and not Path(out).exists()


def test_resume_starts_a_run_that_has_no_files_yet(tmp_path):
    data = _data(tmp_path)
    out = str(tmp_path / "n.nnue")
    train.main(["--data", data, "--out", out, "--epochs", "1", "--resume"] + COMMON)  # how run_round calls it
    assert Path(out).exists() and Path(out + ".ckpt").exists()
