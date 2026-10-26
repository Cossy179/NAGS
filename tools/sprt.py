#!/usr/bin/env python3
"""Engine-vs-engine match runner with SPRT, for NAGS or any UCI engines.

Every opening is played twice with colours reversed (a game pair), and pairs
are scored with the pentanomial model: the pair result for engine 1 is one of
0, 0.5, 1, 1.5 or 2 points. With --sprt the match stops as soon as the
generalised sequential probability ratio test (GSPRT) decides between

    H0: engine 1 is elo0 stronger than engine 2   (usually 0)
    H1: engine 1 is elo1 stronger than engine 2   (e.g. 5)

using logistic Elo bounds. The log-likelihood ratio is computed exactly from
maximum-likelihood pentanomial distributions constrained to each hypothesis
(as in fishtest's statistics); the popular variance-based approximation is
not used because it accepts hypotheses far too early in short samples. Without --sprt a fixed number of games is played
and the Elo difference is reported with a 95% interval.

Example (candidate vs baseline, 8+0.08 s per game, 4 games in parallel):

    python tools/sprt.py --engine build/nags_enhanced --engine /tmp/base/nags_enhanced \\
        --tc 8+0.08 --openings tools/openings/nags_balanced.epd \\
        --concurrency 4 --sprt 0 5 --pgnout match.pgn

Before the first game each engine is started once (UCI handshake, every option set and accepted, ready,
a legal move from a one-ply search), and the engines, the network each one reports, the openings and the
settings are written as one "Fingerprint:" line. An engine that cannot be started, then or later, makes
the match invalid (exit 3, no decision): a failed launch is not a game. An engine that crashes, hangs,
plays an illegal move or loses on time during a game loses that game, and the "Forfeits:" line counts
these per engine. --resume refuses a log whose fingerprint names other engines or settings; a log with a
fingerprint and no status line yet (stopped in the first seconds) resumes at zero games. An earlier match's
games are never overwritten: a --pgnout file that already holds games is refused unless --resume continues
that match.

Exit status: 0 = H1 accepted (or the fixed-length match finished),
1 = H0 accepted, 2 = maximum games reached without a decision, 3 = error (no decision),
4 = interrupted (Ctrl-C) before a decision or the end: continue it with --resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn

# ----------------------------------------------------------------------------
# Statistics


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def sprt_bounds(alpha: float, beta: float) -> Tuple[float, float]:
    """(lower, upper) LLR bounds: accept H0 below lower, H1 above upper."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)  # pair points / 2


def _constrained_mle(p: Sequence[float], s: float) -> List[float]:
    """Distribution q over PAIR_SCORES maximising sum(p_i log q_i) subject to
    sum(q_i x_i) = s. Lagrange: q_i = p_i / (1 + theta (x_i - s)), with theta
    the root of f(theta) = sum(p_i (x_i - s) / (1 + theta (x_i - s))), which is
    decreasing on the interval where every denominator is positive."""
    xs = PAIR_SCORES
    lo = -1.0 / (max(xs) - s) + 1e-12
    hi = 1.0 / (s - min(xs)) - 1e-12
    theta = 0.0
    for _ in range(100):
        f = sum(pi * (x - s) / (1 + theta * (x - s)) for pi, x in zip(p, xs))
        if abs(f) < 1e-14:
            break
        if f > 0:
            lo = theta
        else:
            hi = theta
        fp = -sum(pi * (x - s) ** 2 / (1 + theta * (x - s)) ** 2 for pi, x in zip(p, xs))
        step = theta - f / fp if fp != 0 else (lo + hi) / 2
        theta = step if lo < step < hi else (lo + hi) / 2  # Newton, falling back to bisection
    q = [pi / (1 + theta * (x - s)) for pi, x in zip(p, xs)]
    total = sum(q)
    return [qi / total for qi in q]


@dataclass
class Pentanomial:
    """Counts of game-pair outcomes for engine 1: [LL, LD, DD+WL, WD, WW]."""

    counts: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0])

    def add(self, pair_points: float) -> None:
        self.counts[int(round(pair_points * 2))] += 1

    @property
    def pairs(self) -> int:
        return int(sum(self.counts))

    def _frequencies(self) -> List[float]:
        # Zero counts are replaced by a small value (as fishtest does) so the
        # constrained maximum-likelihood estimates always exist.
        counts = [c if c > 0 else 1e-3 for c in self.counts]
        n = sum(counts)
        return [c / n for c in counts]

    def mean_var(self) -> Tuple[float, float]:
        """Mean and variance of the per-pair score (pair points / 2)."""
        if self.pairs == 0:
            return 0.5, 0.0
        p = self._frequencies()
        mean = sum(pi * x for pi, x in zip(p, PAIR_SCORES))
        var = sum(pi * (x - mean) ** 2 for pi, x in zip(p, PAIR_SCORES))
        return mean, var

    def llr(self, elo0: float, elo1: float) -> float:
        """Generalised log-likelihood ratio of H1 (score s1) against H0 (score s0)."""
        n = self.pairs
        if n == 0:
            return 0.0
        p = self._frequencies()
        q0 = _constrained_mle(p, elo_to_score(elo0))
        q1 = _constrained_mle(p, elo_to_score(elo1))
        return n * sum(pi * (math.log(b) - math.log(a)) for pi, a, b in zip(p, q0, q1))

    def elo(self) -> Tuple[float, float]:
        """Elo difference and the half-width of its 95% interval."""
        n = self.pairs
        if n == 0:
            return 0.0, float("inf")
        mean, var = self.mean_var()
        err = 1.959964 * math.sqrt(var / n)
        lo, hi = score_to_elo(mean - err), score_to_elo(mean + err)
        return score_to_elo(mean), (hi - lo) / 2.0


# ----------------------------------------------------------------------------
# Time control and openings


@dataclass
class Limits:
    base: Optional[float] = None      # seconds per game
    inc: float = 0.0                  # seconds per move
    nodes: Optional[int] = None
    depth: Optional[int] = None
    movetime: Optional[float] = None  # seconds per move
    margin: float = 0.0               # seconds an engine may exceed its clock

    def describe(self) -> str:
        if self.base is not None:
            return f"{self.base:g}+{self.inc:g}"
        if self.movetime is not None:
            return f"movetime {self.movetime:g}s"
        if self.nodes is not None:
            return f"nodes {self.nodes}"
        return f"depth {self.depth}"


def describe_limits(a: Limits, b: Limits) -> str:
    """One limit, or 'x vs y' when the two sides play with different ones."""
    return a.describe() if a.describe() == b.describe() else f"{a.describe()} vs {b.describe()}"


def make_limits(tc: Optional[str], nodes: Optional[int], depth: Optional[int], movetime: Optional[float],
                margin: float) -> Optional[Limits]:
    """Limits from one set of options, or None if none is given."""
    if tc is None and nodes is None and depth is None and movetime is None:
        return None
    lim = Limits(margin=margin, nodes=nodes, depth=depth, movetime=movetime)
    if tc:
        lim.base, lim.inc = parse_tc(tc)
    return lim


def parse_tc(tc: str) -> Tuple[float, float]:
    """'8+0.08' -> (8.0, 0.08); '60' -> (60.0, 0.0). Seconds, like fastchess/cutechess."""
    base, _, inc = tc.partition("+")
    b, i = float(base), float(inc or 0.0)
    if b <= 0 or i < 0:
        raise ValueError(f"invalid time control {tc!r}")
    return b, i


@dataclass
class Opening:
    fen: str
    moves: List[chess.Move]

    def board(self) -> chess.Board:
        b = chess.Board(self.fen)
        for m in self.moves:
            b.push(m)
        return b


def load_openings(path: Optional[str], plies: int = 0) -> List[Opening]:
    """EPD/FEN lines (one position per line) or PGN games (first `plies` moves; 0 = all)."""
    if not path:
        return [Opening(chess.STARTING_FEN, [])]
    openings: List[Opening] = []
    if path.lower().endswith(".pgn"):
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                moves = list(game.mainline_moves())
                if plies > 0:
                    moves = moves[:plies]
                openings.append(Opening(game.board().fen(), moves))
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    board = chess.Board(line)  # full FEN
                except ValueError:
                    board, _ = chess.Board.from_epd(line)
                openings.append(Opening(board.fen(), []))
    if not openings:
        raise ValueError(f"no openings found in {path}")
    return openings


# ----------------------------------------------------------------------------
# Adjudication


@dataclass
class Adjudication:
    draw_movenumber: int = 0   # 0 disables draw adjudication
    draw_count: int = 8        # consecutive moves per side
    draw_score: int = 10       # centipawns
    resign_count: int = 0      # 0 disables resign adjudication
    resign_score: int = 1000   # centipawns
    max_plies: int = 400


def adjudicate(white_scores: Sequence[Optional[int]], fullmove: int, adj: Adjudication) -> Optional[Tuple[str, str]]:
    """white_scores: each engine's evaluation after its move, from White's view,
    one entry per ply. Returns (result, reason) or None."""
    if adj.resign_count > 0 and len(white_scores) >= 2 * adj.resign_count:
        recent = white_scores[-2 * adj.resign_count:]
        if all(s is not None and s >= adj.resign_score for s in recent):
            return "1-0", "adjudication: black resigns"
        if all(s is not None and s <= -adj.resign_score for s in recent):
            return "0-1", "adjudication: white resigns"
    if adj.draw_movenumber > 0 and fullmove >= adj.draw_movenumber and len(white_scores) >= 2 * adj.draw_count:
        recent = white_scores[-2 * adj.draw_count:]
        if all(s is not None and abs(s) <= adj.draw_score for s in recent):
            return "1/2-1/2", "adjudication: draw"
    return None


# ----------------------------------------------------------------------------
# Engines and games


@dataclass
class EngineSpec:
    name: str
    command: List[str]
    options: Dict[str, str] = field(default_factory=dict)
    limits: Optional["Limits"] = None  # this engine's own limits (time odds, a node-limited opponent)


def engine_command(text: str) -> List[str]:
    """An --engine value as an argument list. The path of an existing file is
    one argument as it is: Windows paths contain backslashes, which POSIX
    shell splitting would drop, and paths may contain spaces. Anything else
    is split like a command line (with Windows rules on Windows)."""
    if os.path.isfile(text):
        return [text]
    if os.name == "nt":
        return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t
                for t in shlex.split(text, posix=False)]
    return shlex.split(text)


_STATUS_LINE = re.compile(r"Games (\d+): \+(\d+) =(\d+) -(\d+) \| penta \[([\d, ]+)\]")


class InfrastructureError(RuntimeError):
    """An engine could not be started or set up. The match is invalid: this is not a game result."""


class Player:
    """One engine process; restarted after a crash or hang."""

    def __init__(self, spec: EngineSpec):
        self.spec = spec
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def ensure(self) -> chess.engine.SimpleEngine:
        """The running engine, started (or restarted after a crash) if need be. Raises InfrastructureError
        if it cannot be started or configured: before the fix of 2026-10-23 a Windows path the runner had
        split wrongly made every launch fail, every game was scored as a crash and the match was decided."""
        if self.engine is None:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(self.spec.command, timeout=30)
            except (OSError, chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError) as e:
                raise InfrastructureError(f"{self.spec.name}: cannot start {self.spec.command} "
                                          f"({type(e).__name__}: {e})") from e
            unknown = [k for k in self.spec.options if k not in engine.options]
            if unknown:
                engine.quit()
                raise InfrastructureError(f"{self.spec.name}: unknown UCI option(s) {unknown}")
            try:
                if self.spec.options:
                    engine.configure({k: v for k, v in self.spec.options.items()})
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError, OSError) as e:
                engine.close()
                raise InfrastructureError(f"{self.spec.name}: cannot set its options ({type(e).__name__}: {e})") from e
            self.engine = engine
        return self.engine

    def kill(self) -> None:
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None

    def quit(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                self.kill()
            self.engine = None


@dataclass
class GameResult:
    result: str
    reason: str
    game: chess.pgn.Game
    forfeit: Optional[Tuple[str, str]] = None  # (engine name, "crash" | "time" | "illegal") when one lost so


# What the engine said while its options were set that means it did not take them (NAGS: "EvalFile: cannot
# open ...", "unknown option ..."; other engines' wording varies). A network actually loaded is reported by
# NAGS as "NNUE evaluation: <name>".
_OPTION_FAILURE = re.compile(r"\b(error|cannot|can't|could not|unknown option|invalid|failed|not found|no such)\b",
                             re.IGNORECASE)


def file_sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def command_sha256(command: Sequence[str]) -> Optional[str]:
    """SHA-256 of the files an engine command runs: the binary, or an interpreter and its script (combined)."""
    digests = [file_sha256(c) for c in command if os.path.isfile(c)]
    return digests[0] if len(digests) == 1 else (
        hashlib.sha256("".join(d or "" for d in digests).encode()).hexdigest() if digests else None)


def preflight(spec: EngineSpec, timeout: float = 60.0) -> Dict[str, Any]:
    """Starts the engine once, before any game: the UCI handshake, every option known and set without a
    complaint, isready answered, and a legal move from a one-ply search of the start position. Raises
    InfrastructureError otherwise. Returns the engine's fingerprint: its command, the SHA-256 of the files it
    names (the binary, or an interpreter and its script), its id name, the options set, what it said while
    they were set (for NAGS the network it loaded), the network file's SHA-256 when an EvalFile is given,
    and its static evaluation of the start position (NAGS names the evaluation in use: an embedded network,
    a file or the hand-written one; the embedded network itself is pinned by the binary's SHA-256)."""
    try:
        proc = subprocess.Popen(spec.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except OSError as e:
        raise InfrastructureError(f"{spec.name}: cannot start {spec.command} ({type(e).__name__}: {e})") from e
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def read() -> None:
        for line in proc.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()

    def send(cmd: str) -> None:
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except OSError as e:
            raise InfrastructureError(f"{spec.name}: exited during the preflight (writing {cmd!r})") from e

    def until(token: str, what: str) -> List[str]:
        got: List[str] = []
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                raise InfrastructureError(f"{spec.name}: no {token!r} within {timeout:.0f} s ({what})") from None
            if line is None:
                raise InfrastructureError(f"{spec.name}: exited during the preflight ({what}); it said: "
                                          + " | ".join(got[-5:]))
            got.append(line)
            if line == token or line.startswith(token + " "):
                return got

    try:
        send("uci")
        hello = until("uciok", "UCI handshake")
        name = next((l[len("id name "):] for l in hello if l.startswith("id name ")), None)
        known = {m.group(1).strip().lower() for l in hello
                 if (m := re.match(r"option name (.+?) type ", l))}
        unknown = [k for k in spec.options if k.lower() not in known]
        if unknown:
            raise InfrastructureError(f"{spec.name}: unknown UCI option(s) {unknown}")
        for k, v in spec.options.items():
            send(f"setoption name {k} value {v}" if v != "" else f"setoption name {k}")
        send("isready")
        said = [l[len("info string "):] for l in until("readyok", "setting options")[:-1]
                if l.startswith("info string ")]
        complaints = [s for s in said if _OPTION_FAILURE.search(s) and not s.startswith("NNUE evaluation:")]
        if complaints:
            raise InfrastructureError(f"{spec.name}: did not take its options: " + " | ".join(complaints))
        send("ucinewgame")
        send("position startpos")
        send("eval")  # NAGS: "info string eval <cp> (side to move, <which evaluation>)"; others ignore it or trace
        send("isready")
        evaluation = next((l[len("info string "):] for l in until("readyok", "new game")
                           if l.startswith("info string eval ")), None)
        send("go depth 1")
        best = until("bestmove", "a one-ply search")[-1].split()
        try:
            legal = len(best) > 1 and chess.Move.from_uci(best[1]) in chess.Board().legal_moves
        except ValueError:
            legal = False
        if not legal:
            raise InfrastructureError(f"{spec.name}: a one-ply search from the start position returned "
                                      f"{' '.join(best)!r}, not a legal move")
        send("quit")
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    network = next((s[len("NNUE evaluation: "):] for s in said if s.startswith("NNUE evaluation: ")), None)
    digest = command_sha256(spec.command)
    eval_file = next((v for k, v in spec.options.items() if k.lower() == "evalfile"), None)
    return {"name": spec.name, "command": spec.command, "sha256": digest,
            "id": name, "options": spec.options, "said": said, "network": network, "evaluation": evaluation,
            "eval_file_sha256": file_sha256(eval_file) if eval_file and os.path.isfile(eval_file) else None,
            "limits": spec.limits.describe() if spec.limits else None}


def play_game(white: Player, black: Player, opening: Opening, limits: Limits, adj: Adjudication,
              round_label: str = "?") -> GameResult:
    board = opening.board()
    start_board = chess.Board(opening.fen)
    players = {chess.WHITE: white, chess.BLACK: black}
    # Each side plays with its engine's own limits if it has them, else the match's.
    side_limits = {c: p.spec.limits or limits for c, p in players.items()}
    clocks = {c: side_limits[c].base for c in players}
    white_scores: List[Optional[int]] = []
    played: List[chess.Move] = []
    result, reason = "*", ""
    forfeit: Optional[Tuple[str, str]] = None
    game_key = object()  # a new key makes python-chess send ucinewgame to both engines

    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            result, reason = board.result(claim_draw=True), outcome.termination.name.lower() if outcome else "game over"
            break
        if len(played) >= adj.max_plies:
            result, reason = "1/2-1/2", "adjudication: max plies"
            break

        mover = board.turn
        player = players[mover]
        lim = side_limits[mover]
        loss = "0-1" if mover == chess.WHITE else "1-0"
        if lim.base is not None:
            # An opponent without a clock (fixed nodes, depth or move time) is shown our own.
            other = clocks[not mover] if clocks[not mover] is not None else clocks[mover]
            other_inc = side_limits[not mover].inc if clocks[not mover] is not None else lim.inc
            wc, bc = (clocks[mover], other) if mover == chess.WHITE else (other, clocks[mover])
            wi, bi = (lim.inc, other_inc) if mover == chess.WHITE else (other_inc, lim.inc)
            limit = chess.engine.Limit(white_clock=wc, black_clock=bc, white_inc=wi, black_inc=bi)
        else:
            limit = chess.engine.Limit(nodes=lim.nodes, depth=lim.depth, time=lim.movetime)
        engine = player.ensure()  # a launch that fails is an InfrastructureError: the match is invalid
        try:
            start = time.monotonic()
            play = engine.play(board, limit, info=chess.engine.INFO_SCORE, game=game_key)
            elapsed = time.monotonic() - start
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError, OSError) as e:
            player.kill()
            result, reason = loss, f"{player.spec.name} crashed or hung ({type(e).__name__})"
            forfeit = (player.spec.name, "crash")
            break

        if lim.base is not None:
            clocks[mover] -= elapsed
            if clocks[mover] < -lim.margin:
                result, reason = loss, f"{player.spec.name} lost on time"
                forfeit = (player.spec.name, "time")
                break
            clocks[mover] += lim.inc

        move = play.move
        if move is None or move not in board.legal_moves:
            result, reason = loss, f"{player.spec.name} played an illegal move ({move})"
            forfeit = (player.spec.name, "illegal")
            break

        score = play.info.get("score")
        white_scores.append(score.white().score(mate_score=32000) if score is not None else None)
        board.push(move)
        played.append(move)

        decision = adjudicate(white_scores, board.fullmove_number, adj)
        if decision:
            result, reason = decision
            break

    game = chess.pgn.Game()
    game.setup(start_board)
    node = game
    for m in opening.moves + played:
        node = node.add_variation(m)
    game.headers.update({"Event": "NAGS SPRT", "Round": round_label, "White": white.spec.name,
                         "Black": black.spec.name, "Result": result, "Termination": reason,
                         "TimeControl": describe_limits(side_limits[chess.WHITE], side_limits[chess.BLACK]),
                         "PlyCount": str(len(opening.moves) + len(played))})
    return GameResult(result, reason, game, forfeit)


def points_for(result: str, as_white: bool) -> float:
    white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
    return white if as_white else 1.0 - white


# ----------------------------------------------------------------------------
# Match


class Match:
    def __init__(self, specs: Tuple[EngineSpec, EngineSpec], openings: List[Opening], limits: Limits,
                 adj: Adjudication, max_pairs: int, concurrency: int, sprt: Optional[Tuple[float, float]],
                 alpha: float, beta: float, pgnout: Optional[str], seed: int, shuffle: bool):
        self.specs = specs
        self.openings = list(openings)
        if shuffle:
            random.Random(seed).shuffle(self.openings)
        self.limits = limits
        self.adj = adj
        self.max_pairs = max_pairs
        self.concurrency = max(1, concurrency)
        self.sprt = sprt
        self.bounds = sprt_bounds(alpha, beta)
        self.pgnout = pgnout
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.next_pair = 0
        self.penta = Pentanomial()
        self.wdl = [0, 0, 0]  # engine 1 wins, draws, losses
        self.reasons: Dict[str, int] = {}
        self.forfeits: Dict[Tuple[str, str], int] = {}  # (engine, kind) -> games lost so
        self.decision: Optional[str] = None
        self.errors: List[str] = []
        self.resumed = False  # continues an earlier run of this match (its PGN is added to)

    def _claim_pair(self) -> Optional[int]:
        with self.lock:
            if self.stop.is_set() or self.next_pair >= self.max_pairs:
                return None
            idx = self.next_pair
            self.next_pair += 1
            return idx

    def _record(self, idx: int, games: List[GameResult], engine1_white: List[bool]) -> None:
        with self.lock:
            if self.decision is not None:
                return  # pairs finishing after the SPRT stopped are not counted
            pair_points = 0.0
            for res, e1w in zip(games, engine1_white):
                p = points_for(res.result, e1w)
                pair_points += p
                self.wdl[0 if p == 1.0 else 1 if p == 0.5 else 2] += 1
                self.reasons[res.reason] = self.reasons.get(res.reason, 0) + 1
                if res.forfeit:
                    self.forfeits[res.forfeit] = self.forfeits.get(res.forfeit, 0) + 1
            self.penta.add(pair_points)
            if self.pgnout:
                with open(self.pgnout, "a") as f:
                    for res in games:
                        print(res.game, file=f, end="\n\n")
            if self.sprt and self.decision is None:
                llr = self.penta.llr(*self.sprt)
                if llr >= self.bounds[1]:
                    self.decision = "H1"
                elif llr <= self.bounds[0]:
                    self.decision = "H0"
                if self.decision:
                    self.stop.set()

    def _worker(self) -> None:
        p1, p2 = Player(self.specs[0]), Player(self.specs[1])
        try:
            while True:
                idx = self._claim_pair()
                if idx is None:
                    break
                opening = self.openings[idx % len(self.openings)]
                g1 = play_game(p1, p2, opening, self.limits, self.adj, f"{idx + 1}.1")
                if self.stop.is_set() and self.decision is not None:
                    break  # decided while this pair was in progress: discard the half pair
                g2 = play_game(p2, p1, opening, self.limits, self.adj, f"{idx + 1}.2")
                self._record(idx, [g1, g2], [True, False])
        except Exception as e:  # an engine that cannot be started, configuration errors: stop the whole match
            with self.lock:
                self.errors.append(f"{type(e).__name__}: {e}")
            self.stop.set()
        finally:
            p1.quit()
            p2.quit()

    def status(self) -> str:
        with self.lock:
            games = sum(self.wdl)
            elo, err = self.penta.elo()
            line = (f"Games {games}: +{self.wdl[0]} ={self.wdl[1]} -{self.wdl[2]} | "
                    f"penta {[int(c) for c in self.penta.counts]} | Elo {elo:+.1f} ± {err:.1f}")
            if self.sprt:
                line += (f" | LLR {self.penta.llr(*self.sprt):+.2f} "
                         f"[{self.bounds[0]:.2f}, {self.bounds[1]:.2f}] for [{self.sprt[0]:g}, {self.sprt[1]:g}]")
            return line

    def resume(self, status_line: str) -> None:
        """Continues an interrupted match from its last status line: the counts
        are restored and play goes on with the next opening pair."""
        m = _STATUS_LINE.search(status_line)
        if not m:
            raise ValueError("no status line ('Games N: +W =D -L | penta [...]') to resume from")
        self.wdl = [int(m.group(2)), int(m.group(3)), int(m.group(4))]
        self.penta.counts = [int(x) for x in m.group(5).split(",")]
        if len(self.penta.counts) != 5 or 2 * self.penta.pairs != sum(self.wdl):
            raise ValueError("inconsistent status line")
        self.next_pair = self.penta.pairs
        self.resumed = True

    def run(self, report_interval: float = 10.0) -> int:
        # An earlier match's games are never overwritten: only a resumed match adds to its PGN.
        if self.pgnout and not self.resumed and pgn_has_games(self.pgnout):
            raise FileExistsError(f"{self.pgnout} already holds games: resume that match, or write elsewhere")
        threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(self.concurrency)]
        for t in threads:
            t.start()
        last = time.monotonic()
        interrupted = False
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(0.2)
                if time.monotonic() - last >= report_interval:
                    print(self.status(), flush=True)
                    last = time.monotonic()
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; finishing games in progress...", flush=True)
            self.stop.set()
            for t in threads:
                t.join()
        print(self.status(), flush=True)
        if self.reasons:
            print("Endings: " + ", ".join(f"{k} {v}" for k, v in sorted(self.reasons.items(), key=lambda kv: -kv[1])))
        if self.forfeits:
            kinds = {"crash": "crashed or hung", "time": "lost on time", "illegal": "illegal move"}
            print("Forfeits: " + "; ".join(
                f"{spec.name}: " + ", ".join(f"{self.forfeits[(spec.name, k)]} {kinds[k]}" for k in kinds
                                             if (spec.name, k) in self.forfeits)
                for spec in self.specs if any(n == spec.name for n, _ in self.forfeits)), flush=True)
        if self.errors:
            print("ERROR: " + "; ".join(self.errors), file=sys.stderr)
            print("Match invalid: an engine could not be started or set up; no decision", flush=True)
            return 3
        if self.sprt:
            if self.decision == "H1":
                print(f"H1 accepted: {self.specs[0].name} is stronger (Elo bounds [{self.sprt[0]:g}, {self.sprt[1]:g}])")
                return 0
            if self.decision == "H0":
                print(f"H0 accepted: {self.specs[0].name} is not stronger by {self.sprt[1]:g} Elo")
                return 1
        if interrupted:  # not the end of the match: --resume continues it (and run_round does so by itself)
            print(f"Match interrupted after {sum(self.wdl)} games, before its end: no result yet", flush=True)
            return 4
        if self.sprt:
            print("SPRT inconclusive: maximum number of games reached")
            return 2
        return 0


# ----------------------------------------------------------------------------
# Command line


def pgn_has_games(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def fingerprint_differences(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """What makes games under `new` not the same experiment as those under `old` (the game cap and the
    concurrency may change on a resume; what an engine printed besides its network name is not compared)."""
    out = []
    for i, (a, b) in enumerate(zip(old.get("engines", []), new.get("engines", [])), 1):
        for key in ("sha256", "id", "options", "network", "eval_file_sha256", "evaluation", "limits"):
            if a.get(key) != b.get(key):
                out.append(f"engine {i} {key} was {a.get(key)!r}, now {b.get(key)!r}")
    for key in ("sha256", "plies"):
        if old.get("openings", {}).get(key) != new.get("openings", {}).get(key):
            out.append(f"openings {key} was {old['openings'].get(key)!r}, now {new['openings'].get(key)!r}")
    for key, value in new.get("settings", {}).items():
        if key not in ("games", "concurrency") and old.get("settings", {}).get(key) != value:
            out.append(f"{key} was {old.get('settings', {}).get(key)!r}, now {value!r}")
    return out


def parse_options(pairs: Sequence[str]) -> Dict[str, str]:
    out = {}
    for p in pairs:
        name, sep, value = p.partition("=")
        if not sep:
            raise ValueError(f"option {p!r} must look like NAME=VALUE")
        out[name] = value
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", action="append", required=True, help="engine command (give twice: candidate first)")
    ap.add_argument("--name", action="append", help="display names (default: file names)")
    ap.add_argument("--option", action="append", default=[], help="UCI option NAME=VALUE for both engines")
    ap.add_argument("--option1", action="append", default=[], help="UCI option for engine 1 only")
    ap.add_argument("--option2", action="append", default=[], help="UCI option for engine 2 only")
    lim = ap.add_mutually_exclusive_group()
    lim.add_argument("--tc", help="time control 'base+inc' in seconds, e.g. 8+0.08")
    lim.add_argument("--nodes", type=int, help="fixed nodes per move")
    lim.add_argument("--depth", type=int, help="fixed depth per move")
    lim.add_argument("--movetime", type=float, help="fixed seconds per move")
    for k in ("1", "2"):  # one engine's own limit (time odds, a node-limited opponent), instead of the shared one
        own = ap.add_mutually_exclusive_group()
        own.add_argument(f"--tc{k}", help=f"time control of engine {k} only")
        own.add_argument(f"--nodes{k}", type=int, help=f"fixed nodes per move for engine {k} only")
        own.add_argument(f"--depth{k}", type=int, help=f"fixed depth per move for engine {k} only")
        own.add_argument(f"--movetime{k}", type=float, help=f"fixed seconds per move for engine {k} only")
    ap.add_argument("--timemargin", type=float, default=0.0, help="seconds an engine may overrun its clock")
    ap.add_argument("--openings", help="EPD/FEN (one position per line) or PGN file")
    ap.add_argument("--plies", type=int, default=0, help="opening plies to use from PGN games (0 = all)")
    ap.add_argument("--no-shuffle", action="store_true", help="play openings in file order")
    ap.add_argument("--games", type=int, default=20000, help="maximum number of games (rounded up to pairs)")
    ap.add_argument("--concurrency", type=int, default=1, help="games played in parallel")
    ap.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"), help="run an SPRT with these logistic Elo bounds")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--draw-adjudication", nargs=3, type=int, metavar=("MOVENUMBER", "COUNT", "SCORE"),
                    help="draw after MOVENUMBER if both engines report |score| <= SCORE cp for COUNT moves each")
    ap.add_argument("--resign-adjudication", nargs=2, type=int, metavar=("COUNT", "SCORE"),
                    help="decide the game if both engines agree on |score| >= SCORE cp for COUNT moves each")
    ap.add_argument("--max-plies", type=int, default=400, help="adjudicate a draw after this many plies")
    ap.add_argument("--pgnout", help="write all games to this PGN file (a file that already holds games is "
                                     "refused, except by --resume)")
    ap.add_argument("--seed", type=int, default=1, help="seed for the opening order")
    ap.add_argument("--report", type=float, default=10.0, help="seconds between status lines")
    ap.add_argument("--resume", metavar="LOG",
                    help="continue an interrupted match from the last status line in its output (same engines, "
                         "settings and --seed); games are appended to --pgnout")
    ap.add_argument("--force-resume", action="store_true",
                    help="resume even if the log's fingerprint names other engines or settings (the games of "
                         "both are then mixed in one result)")
    args = ap.parse_args(argv)

    if len(args.engine) != 2:
        ap.error("give exactly two --engine arguments (candidate first, baseline second)")
    names = args.name or [os.path.basename(engine_command(e)[0]) for e in args.engine]
    if len(names) != 2:
        ap.error("give either no --name or exactly two")
    if names[0] == names[1]:
        names = [names[0] + "-1", names[1] + "-2"]
    try:
        common = parse_options(args.option)
        limits = make_limits(args.tc, args.nodes, args.depth, args.movetime, args.timemargin)
        own = [make_limits(getattr(args, f"tc{k}"), getattr(args, f"nodes{k}"), getattr(args, f"depth{k}"),
                           getattr(args, f"movetime{k}"), args.timemargin) for k in ("1", "2")]
        if limits is None and None in own:
            ap.error("give a limit (--tc, --nodes, --depth or --movetime), or one for each engine (--tc1 ... --tc2 ...)")
        limits = limits or own[0]
        specs = (EngineSpec(names[0], engine_command(args.engine[0]), {**common, **parse_options(args.option1)}, own[0]),
                 EngineSpec(names[1], engine_command(args.engine[1]), {**common, **parse_options(args.option2)}, own[1]))
        openings = load_openings(args.openings, args.plies)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    if args.pgnout and not args.resume and pgn_has_games(args.pgnout):
        print(f"ERROR: --pgnout {args.pgnout} already holds games: continue that match with --resume LOG, or "
              "write to another file (an earlier match's games are never overwritten)", file=sys.stderr)
        return 3

    adj = Adjudication(max_plies=args.max_plies)
    if args.draw_adjudication:
        adj.draw_movenumber, adj.draw_count, adj.draw_score = args.draw_adjudication
    if args.resign_adjudication:
        adj.resign_count, adj.resign_score = args.resign_adjudication

    match = Match(specs, openings, limits, adj, max_pairs=(args.games + 1) // 2, concurrency=args.concurrency,
                  sprt=tuple(args.sprt) if args.sprt else None, alpha=args.alpha, beta=args.beta,
                  pgnout=args.pgnout, seed=args.seed, shuffle=not args.no_shuffle)
    shown = describe_limits(specs[0].limits or limits, specs[1].limits or limits)
    try:
        engines = [preflight(s) for s in specs]
    except InfrastructureError as e:
        print(f"ERROR: preflight: {e}", file=sys.stderr)
        print("Match invalid: an engine could not be started or set up; no decision", flush=True)
        return 3
    fingerprint = {
        "engines": engines,
        "openings": {"path": args.openings, "sha256": file_sha256(args.openings) if args.openings else None,
                     "count": len(openings), "plies": args.plies},
        "settings": {"limits": shown, "sprt": args.sprt, "alpha": args.alpha, "beta": args.beta, "seed": args.seed,
                     "shuffle": not args.no_shuffle, "max_plies": args.max_plies, "timemargin": args.timemargin,
                     "draw_adjudication": args.draw_adjudication, "resign_adjudication": args.resign_adjudication,
                     "games": args.games, "concurrency": args.concurrency},
        "runner_sha256": file_sha256(__file__), "python_chess": chess.__version__,
    }
    if args.resume:
        try:
            with open(args.resume) as f:
                log = f.readlines()
            # The last complete status line: a runner killed while writing one leaves it cut off.
            lines = [l for l in log if l.startswith("Games ") and _STATUS_LINE.search(l)]
            if lines or not any(l.startswith("Fingerprint: ") for l in log):
                match.resume(lines[-1] if lines else "")
            else:  # stopped before its first status line: no game counted yet, its PGN is continued
                match.resumed = True
        except (ValueError, OSError) as e:
            print(f"ERROR: --resume: {e}", file=sys.stderr)
            return 3
        earlier = [l for l in log if l.startswith("Fingerprint: ")]
        if not earlier:
            print("Note: the log has no fingerprint (an older runner): its engines and settings are not checked")
        else:
            differences = fingerprint_differences(json.loads(earlier[-1][len("Fingerprint: "):]), fingerprint)
            if differences and not args.force_resume:
                print("ERROR: --resume: the log was written with other engines or settings: " + "; ".join(differences)
                      + " (--force-resume mixes both in one result)", file=sys.stderr)
                return 3
            for d in differences:
                print(f"WARNING: resuming although {d}", flush=True)
        print(f"Resuming after {sum(match.wdl)} games", flush=True)
    print("Fingerprint: " + json.dumps(fingerprint, sort_keys=True), flush=True)
    print(f"{specs[0].name} vs {specs[1].name}: {shown}, {len(openings)} openings, "
          f"concurrency {match.concurrency}" + (f", SPRT [{args.sprt[0]:g}, {args.sprt[1]:g}]" if args.sprt else ""),
          flush=True)
    return match.run(args.report)


if __name__ == "__main__":
    sys.exit(main())
