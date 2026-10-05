"""The NAGS hybrid with a (fake, instant) GNN service: the MCTS arm runs, the
time limit holds, and an MCTS proposal goes through alpha-beta verification.
Skipped when the nags engine is not built."""

import json
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _engine():
    for c in (ROOT / "build" / "nags", ROOT / "build" / "Release" / "nags.exe"):
        if c.exists():
            return c
    return None


class _FakeGnn(socketserver.StreamRequestHandler):
    """Answers like rpc_server.py: value 0 and nearly all prior on a2a3/a7a6
    (a quiet, slightly passive move, so MCTS proposes it and alpha-beta has
    to judge it)."""

    def handle(self):
        for line in self.rfile:
            req = json.loads(line)
            results = []
            for moves in req["moves"]:
                favoured = [m for m in moves if m in ("a2a3", "a7a6")]
                pri = [0.9 if m in favoured else 0.1 / max(1, len(moves) - len(favoured)) for m in moves]
                if not favoured:
                    pri = [1.0 / len(moves)] * len(moves)
                results.append({"value": 0.0, "uncertainty": 0.05, "move_priors": pri})
            self.wfile.write((json.dumps({"results": results, "policy_dim": 4672}) + "\n").encode())
            self.wfile.flush()


@pytest.fixture
def fake_gnn():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _FakeGnn)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _run(port, go, min_time=None):
    cmds = f"setoption name NNPort value {port}\nsetoption name UseMetaLearner value false\n"
    if min_time is not None:
        cmds += f"setoption name MctsMinTime value {min_time}\n"
    cmds += f"position startpos\n{go}\n"
    start = time.monotonic()
    out = subprocess.run([str(_engine())], input=cmds, capture_output=True, text=True, timeout=60).stdout
    return out, time.monotonic() - start


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_mcts_arm_runs_and_respects_time(fake_gnn):
    # MctsMinTime 0 forces the MCTS arm on even for very short moves.
    for movetime in (50, 400, 1200):
        out, elapsed = _run(fake_gnn, f"go movetime {movetime}", min_time=0 if movetime < 1200 else None)
        summary = [l for l in out.splitlines() if l.startswith("info string nags ")]
        assert summary and "evaluator network" in summary[-1], out
        sims = int(summary[-1].split("mcts_sims ")[1].split()[0])
        assert sims > 0
        assert "bestmove" in out
        assert elapsed < movetime / 1000 + 1.5, f"movetime {movetime} took {elapsed:.1f}s"


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_no_mcts_with_little_time(fake_gnn):
    out, elapsed = _run(fake_gnn, "go movetime 300")  # default MctsMinTime 1000
    assert "too little time: MCTS off" in out and "mcts_sims 0" in out
    assert elapsed < 1.5


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_mcts_proposal_is_verified(fake_gnn):
    out, _ = _run(fake_gnn, "go movetime 1500")
    summary = [l for l in out.splitlines() if l.startswith("info string nags ")][-1]
    # MCTS concentrates on a2a3; unless alpha-beta already chose it, the
    # proposal must have gone through verification.
    assert "mcts_top a2a3" in summary, summary
    chose = summary.split(" chose ")[1].split()[0]
    if chose != "a2a3":
        assert "rejected" in summary or "no time to verify" in summary, summary
    else:
        assert "verified" in summary or "(alpha-beta)" in summary, summary


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_no_service_means_plain_alpha_beta():
    cmds = "setoption name NNPort value 1\nsetoption name UseMetaLearner value false\nposition startpos\ngo depth 6\n"
    out = subprocess.run([str(_engine())], input=cmds, capture_output=True, text=True, timeout=60).stdout
    assert "mcts_sims 0" in out and "MCTS off" in out and "bestmove" in out
