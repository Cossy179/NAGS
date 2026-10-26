"""tools/nnue/screen.py: candidates ranked on a fixed evaluation set at a small scale, the best trained in
full, every run resumable."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import screen  # noqa: E402
from test_nnue import FENS  # noqa: E402


def _learnable(path, n):
    """Each of the five positions always with the same score and result, so training can learn them."""
    scores, results = (-250, -100, 0, 120, 300), ("0.0", "0.0", "0.5", "1.0", "1.0")
    with open(path, "w") as fh:
        for i in range(n):
            k = i % len(FENS)
            sign = 1 if FENS[k].split()[1] == "w" else -1  # scores and results are White's
            fh.write(f"{FENS[k]} | {sign * scores[k]} | {results[k] if sign > 0 else results[-1 - k]}\n")


def test_screening_two_stages(tmp_path, capsys):
    for name, n in (("small", 300), ("full", 900), ("evalset", 200)):
        _learnable(tmp_path / f"{name}.txt", n)
        pack.main([str(tmp_path / f"{name}.txt")])
    design = ["--hidden", "8", "--batch", "64", "--threads", "1", "--device", "cpu", "--buckets", "8"]
    cands = {"lr-1e-3": design + ["--lr", "1e-3"], "lr-3e-3": design + ["--lr", "3e-3"],
             "lr-1e-7": design + ["--lr", "1e-7"]}  # hardly learns: last at stage 0
    (tmp_path / "c.json").write_text(json.dumps(cands))
    argv = ["--candidates", str(tmp_path / "c.json"), "--out", str(tmp_path / "s"), "--eval-set",
            str(tmp_path / "evalset.bin"), "--stage0-data", str(tmp_path / "small.bin"), "--stage0-epochs", "2",
            "--stage1-data", str(tmp_path / "full.bin"), "--stage1-epochs", "2", "--keep", "2",
            "--train-arg=--eval-overlap-limit", "--train-arg=1"]  # the synthetic sets share their five FENs
    assert screen.main(argv) == 0
    res = json.loads((tmp_path / "s" / "results.json").read_text())
    s0 = res["stage0"]
    assert res["ranking_stage0"] == sorted(s0, key=lambda n: s0[n]["eval_loss"])
    assert res["ranking_stage0"][-1] == "lr-1e-7" and set(res["stage1"]) == set(res["ranking_stage0"][:2])
    for n in res["stage1"]:
        assert (tmp_path / "s" / "stage1" / f"{n}.best.nnue").exists()
    out = capsys.readouterr().out
    assert "need a match at equal time" in out and "EvalFile=" in out
    # Run again: every run is finished, nothing is trained, the results stand.
    before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "s" / "stage1").glob("*.nnue")}
    assert screen.main(argv) == 0
    assert {p.name: p.stat().st_mtime_ns for p in (tmp_path / "s" / "stage1").glob("*.nnue")} == before
    assert json.loads((tmp_path / "s" / "results.json").read_text())["ranking_stage0"] == res["ranking_stage0"]
