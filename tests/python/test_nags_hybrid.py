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
    cmds = f"setoption name UseNN value true\nsetoption name NNPort value {port}\nsetoption name UseMetaLearner value false\n"
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
        # At movetime 50 the 50 ms move overhead leaves a 1 ms search: the
        # MCTS thread may not finish a simulation before alpha-beta stops
        # (it failed about 1 run in 5 on a loaded machine).
        assert sims > 0 or movetime < 400, summary[-1]
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
    cmds = ("setoption name UseNN value true\nsetoption name NNPort value 1\nsetoption name UseMetaLearner value false\n"
            "position startpos\ngo depth 6\n")
    out = subprocess.run([str(_engine())], input=cmds, capture_output=True, text=True, timeout=60).stdout
    assert "mcts_sims 0" in out and "MCTS off" in out and "bestmove" in out


class _DyingGnn(_FakeGnn):
    """Answers the first three requests of a connection, then hangs up."""

    def handle(self):
        for i, line in enumerate(self.rfile):
            if i >= 3:
                return
            req = json.loads(line)
            results = [{"value": 0.0, "uncertainty": 0.05, "move_priors": [1.0 / len(m)] * len(m)} for m in req["moves"]]
            self.wfile.write((json.dumps({"results": results}) + "\n").encode())
            self.wfile.flush()


def _serve(handler):
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _summary(out):
    lines = [l for l in out.splitlines() if l.startswith("info string nags ")]
    assert lines, out
    fields = lines[-1].split()
    return lines[-1], {k: fields[i + 1] for i, k in enumerate(fields[:-1])}


def _run_cmds(port, cmds, timeout=60):
    pre = f"setoption name UseNN value true\nsetoption name NNPort value {port}\nsetoption name UseMetaLearner value false\n"
    start = time.monotonic()
    out = subprocess.run([str(_engine())], input=pre + cmds, capture_output=True, text=True, timeout=timeout).stdout
    return out, time.monotonic() - start


@pytest.mark.skipif(_engine() is None, reason="nags not built")
@pytest.mark.parametrize("pipeline", ["false", "true"])
def test_batched_mcts(fake_gnn, pipeline):
    for movetime in (400, 1500):
        out, elapsed = _run_cmds(fake_gnn, f"setoption name NNBatch value 16\nsetoption name NNPipeline value {pipeline}\n"
                                           f"setoption name MctsMinTime value 0\nposition startpos\ngo movetime {movetime}\n")
        line, f = _summary(out)
        assert "evaluator network" in line and "bestmove" in out
        assert int(f["mcts_sims"]) > 64 and float(f["rpc_batch"]) > 2, line
        assert int(f["rpc_failures"]) == 0
        assert elapsed < movetime / 1000 + 1.5, f"movetime {movetime} took {elapsed:.1f}s"
        # With virtual loss the batch spreads over moves, yet the visits still concentrate on a2a3.
        if movetime >= 1500:
            assert "mcts_top a2a3" in line, line
            chose = line.split(" chose ")[1].split()[0]
            if chose != "a2a3":
                assert "rejected" in line or "no time to verify" in line, line


@pytest.mark.skipif(_engine() is None, reason="nags not built")
@pytest.mark.parametrize("batch", ["1", "8"])
def test_network_lost_mid_search(batch):
    server = _serve(_DyingGnn)
    try:
        out, elapsed = _run_cmds(server.server_address[1], f"setoption name NNBatch value {batch}\nsetoption name NNPipeline "
                                                           f"value true\nposition startpos\ngo movetime 1200\n")
    finally:
        server.shutdown()
        server.server_close()
    line, f = _summary(out)
    assert "bestmove" in out and elapsed < 2.7
    assert int(f["rpc_failures"]) == 1, line  # one failure, then the heuristic for the rest of the move
    assert int(f["mcts_sims"]) > 0


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_evaluation_cache(fake_gnn):
    go = "position startpos moves e2e4\ngo movetime 1200\n"
    out, _ = _run_cmds(fake_gnn, "setoption name NNBatch value 8\n" + go + go)
    summaries = [l for l in out.splitlines() if l.startswith("info string nags ")]
    assert len(summaries) == 2
    hits = [int(l.split("cache_hits ")[1].split()[0]) for l in summaries]
    requests = [int(l.split("rpc_positions ")[1].split()[0]) for l in summaries]
    # The second search of the same position finds the first one's tree in the cache.
    assert hits[1] > hits[0] and hits[1] > 50, summaries
    out, _ = _run_cmds(fake_gnn, "setoption name NNBatch value 8\nsetoption name NNCache value 0\n" + go + go)
    assert all("cache_hits 0 " in l for l in out.splitlines() if l.startswith("info string nags ")), out
    assert requests[0] > 0


class _Session:
    """An engine process driven command by command (to read one search's result before the next)."""

    def __init__(self, port, options=""):
        self.proc = subprocess.Popen([str(_engine())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.send(f"setoption name UseNN value true\nsetoption name NNPort value {port}\n"
                  f"setoption name UseMetaLearner value false\nsetoption name MctsMinTime value 0\n{options}")

    def send(self, cmds):
        self.proc.stdin.write(cmds)
        self.proc.stdin.flush()

    def go(self, position, go="go movetime 600"):
        self.send(f"{position}\n{go}\n")
        lines = []
        while True:
            line = self.proc.stdout.readline()
            assert line, "engine exited"
            lines.append(line.rstrip("\n"))
            if line.startswith("bestmove"):
                return line.split()[1], _summary("\n".join(lines))[1]

    def close(self):
        self.send("quit\n")
        self.proc.wait(timeout=10)


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_default_hybrid_options_report_off(fake_gnn):
    out, _ = _run_cmds(fake_gnn, "setoption name MctsMinTime value 0\nposition startpos\ngo movetime 300\n")
    _, f = _summary(out)
    assert (f["mcts_reused"], f["tt_values"], f["smart_stop"], f["root_order"]) == ("0", "0", "0", "0")


@pytest.mark.skipif(_engine() is None, reason="nags not built")
@pytest.mark.parametrize("batch", ["1", "8"])
def test_tree_reuse(fake_gnn, batch):
    s = _Session(fake_gnn, f"setoption name MctsReuse value true\nsetoption name NNBatch value {batch}\n")
    try:
        best, f1 = s.go("position startpos")
        assert f1["mcts_reused"] == "0" and int(f1["mcts_sims"]) > 0
        # Our move and the reply the fake network prefers (a7a6): that grandchild has a subtree.
        _, f2 = s.go(f"position startpos moves {best} a7a6")
        assert int(f2["mcts_reused"]) > 0, f2
        # The same position again (e.g. a second 'go'): the whole tree is kept.
        _, f3 = s.go(f"position startpos moves {best} a7a6")
        assert int(f3["mcts_reused"]) >= int(f2["mcts_reused"]) + int(f2["mcts_sims"]) - 1, (f2, f3)
        # A new game starts from nothing.
        s.send("ucinewgame\n")
        _, f4 = s.go(f"position startpos moves {best} a7a6")
        assert f4["mcts_reused"] == "0"
        # A position not in the tree: nothing to reuse, and no error.
        _, f5 = s.go("position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        assert f5["mcts_reused"] == "0"
    finally:
        s.close()


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_smart_stop(fake_gnn):
    # The fake network puts 0.9 of the prior on a2a3, so its lead soon exceeds the rest of the budget.
    out, _ = _run_cmds(fake_gnn, "setoption name MctsMinTime value 0\nposition startpos\ngo movetime 1500\n")
    _, plain = _summary(out)
    out, _ = _run_cmds(fake_gnn, "setoption name MctsMinTime value 0\nsetoption name MctsSmartStop value true\n"
                                 "position startpos\ngo movetime 1500\n")
    line, smart = _summary(out)
    assert plain["smart_stop"] == "0" and smart["smart_stop"] == "1", line
    assert int(smart["mcts_sims"]) < int(plain["mcts_sims"]) and "mcts_top a2a3" in line


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_root_order_and_tt_values(fake_gnn):
    # NNRootOrder needs only the root evaluation, so it also works with MCTS off.
    out, _ = _run_cmds(fake_gnn, "setoption name NNRootOrder value true\nsetoption name MctsMinTime value 600000\n"
                                 "position startpos\ngo movetime 300\n")
    line, f = _summary(out)
    assert f["root_order"] == "1" and f["mcts_sims"] == "0" and "bestmove" in out, line
    out, _ = _run_cmds(fake_gnn, "setoption name MctsMinTime value 0\nsetoption name MctsTTValue value 1\n"
                                 "position startpos\ngo movetime 800\n")
    line, f = _summary(out)
    assert int(f["tt_values"]) > 0 and "bestmove" in out, line


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_puct_and_share_options(fake_gnn):
    opts = ("setoption name MctsMinTime value 0\nsetoption name MctsCpuctBase value 500\nsetoption name MctsFpu value 0\n"
            "setoption name MctsPolicyTemp value 200\nsetoption name MctsAbShare value 100\n")
    out, elapsed = _run_cmds(fake_gnn, opts + "position startpos\ngo movetime 600\n")
    line, f = _summary(out)
    assert int(f["mcts_sims"]) > 0 and "bestmove" in out and elapsed < 2.1, line
    # A flatter prior (T = 2) spreads the visits: a2a3 keeps fewer of them than with T = 1.
    base, _ = _run_cmds(fake_gnn, "setoption name MctsMinTime value 0\nposition startpos\ngo movetime 600\n")
    _, fb = _summary(base)
    assert int(f["visits"]) < int(fb["visits"]), (f, fb)
