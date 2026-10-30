"""tools/spsa.py: the SPSA schedule, convergence on a known optimum (a fake match), resuming and refusing a
changed run, and a real tuning build when one is built."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import spsa  # noqa: E402


def test_the_schedule_ends_at_c_end_and_r_end():
    n, c_end, r_end = 1000, 5.0, 0.002
    c0, a0 = spsa.schedule(0, n, c_end, r_end)
    c_last, a_last = spsa.schedule(n - 1, n, c_end, r_end)
    assert c0 > c_last and a0 > a_last
    assert math.isclose(c_last, c_end, rel_tol=1e-6) and math.isclose(a_last / c_last ** 2, r_end, rel_tol=1e-6)


def _fake_match(target):
    """+1 for the side nearer the target values (summed over the parameters), -1 for the other, 0 if equal."""
    def play(state, plus, minus, seed, concurrency):
        dp = sum(abs(plus[k] - target[k]) for k in target)
        dm = sum(abs(minus[k] - target[k]) for k in target)
        return (dp < dm) - (dp > dm)
    return play


def _state(n=400, r_end=0.05, **theta):
    # A larger learning rate than fishtest's 0.002: the fake match gives +-1 per iteration, a real one up to
    # +-2 per pair over many more iterations.
    params = {k: {"theta": float(v), "default": v, "min": 0, "max": 300, "c_end": 6.0, "r_end": r_end}
              for k, v in theta.items()}
    return {"params": params, "iterations": n, "pairs": 2, "seed": 1, "done": 0}


def test_spsa_moves_towards_the_optimum():
    state = _state(n=600, A=80, B=200)
    spsa.iterate(state, _fake_match({"A": 140, "B": 120}), 1, save=lambda: None, log=lambda *a, **k: None)
    assert state["done"] == 600 and state["games"] == 600 * 4
    assert abs(state["params"]["A"]["theta"] - 140) < abs(80 - 140) / 3
    assert abs(state["params"]["B"]["theta"] - 120) < abs(200 - 120) / 3


def test_a_run_resumes_and_refuses_other_settings(tmp_path, monkeypatch, capsys):
    engine = Path(__file__)  # any file: the engine is only hashed and its options faked
    monkeypatch.setattr(spsa, "engine_params", lambda cmd: {"A": {"default": 80, "min": 0, "max": 300},
                                                           "B": {"default": 200, "min": 0, "max": 300}})
    calls = []

    def play(state, plus, minus, seed, concurrency):
        calls.append(seed)
        if len(calls) == 5:
            raise KeyboardInterrupt
        return _fake_match({"A": 140, "B": 120})(state, plus, minus, seed, concurrency)

    monkeypatch.setattr(spsa, "play_pairs", play)
    state_file = tmp_path / "s.json"
    argv = ["--engine", str(engine), "--params", "A", "B", "--iterations", "20", "--pairs", "1", "--state",
            str(state_file)]
    assert spsa.main(argv) == 4  # interrupted during iteration 5
    assert json.loads(state_file.read_text())["done"] == 4
    assert spsa.main(argv) == 0
    out = capsys.readouterr().out
    assert "Continuing after iteration 4 of 20" in out and "Tuned after 40 games" in out and "--option1 A=" in out
    assert calls[5:] == list(range(1 + 4, 1 + 20))  # the interrupted iteration played again, then the rest
    assert spsa.main(argv[:-4] + ["--iterations", "30", "--pairs", "1", "--state", str(state_file)]) == 3
    assert spsa.main(["--engine", str(engine), "--params", "A", "C", "--iterations", "20", "--state",
                      str(tmp_path / "t.json")]) == 3  # C is not an option of the engine


def _tune_build():
    for c in (ROOT / "build-tune" / "nags_enhanced", ROOT / "build-tune" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return str(c)
    return None


@pytest.mark.skipif(_tune_build() is None, reason="no NAGS_TUNE build in build-tune/")
def test_a_real_tuning_build(tmp_path):
    params = spsa.engine_params([_tune_build()])
    assert params["RfpMargin"] == {"default": 80, "min": 30, "max": 200}
    state_file = tmp_path / "s.json"
    assert spsa.main(["--engine", _tune_build(), "--params", "RfpMargin", "FutilityMargin", "--iterations", "2",
                      "--pairs", "1", "--tc", "depth=2", "--concurrency", "1", "--state", str(state_file)]) == 0
    state = json.loads(state_file.read_text())
    assert state["done"] == 2 and state["games"] == 4 and len(state["trace"]) == 2
