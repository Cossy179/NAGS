#!/usr/bin/env python3
"""Teacher labels: rescore nags_datagen shards with another UCI engine, and compare label sets.

  label    relabels whole games of indexed shards (nags_datagen --first-game) with any UCI engine at a fixed
           limit (nodes, depth or move time), one engine process per --workers, and writes them as a new shard
           in --out-dir: the same records with the teacher's score (White's view, times --scale, rounded), and
           an index with the games' new checksums. A sample of the games (--fraction or --max-games, drawn by
           --select-seed) keeps the run small. Stop it at any time: the same command continues with the games
           still missing. <out>.teacher.json records the engine (name, SHA-256 of the binary, options), the
           limit and the engine time per position, which is what "equal compute" compares.
  compare  joins a base shard with one or more relabelled versions of it (by seed, game number and record) and
           reports, on the games all of them hold: each label set's fit to the game results (the sigmoid scale K
           that predicts them best and the error at that K), the scale that puts each set on the base's
           (--scale for label), the mean difference between sets once scaled, and each set's cost per position.

  python tools/nnue/rescore.py label --data data/selfplay_9101.bin --engine /usr/games/stockfish \\
      --nodes 200000 --fraction 0.05 --workers 5 --out-dir data/teacher_sf16
  python tools/nnue/rescore.py label --data data/selfplay_9101.bin --engine build/nags_enhanced \\
      --nodes 400000 --fraction 0.05 --workers 5 --out-dir data/teacher_nags
  python tools/nnue/rescore.py compare --base data/selfplay_9101.bin \\
      --labels data/teacher_sf16/selfplay_9101.bin data/teacher_nags/selfplay_9101.bin

Limits of the 32-byte records: they hold no castling rights, en-passant square or move counters. A position is
given to the teacher with castling rights wherever the king and rook stand on their original squares, no
en-passant square and a halfmove clock of 0. A position where the teacher finds a mate is dropped (--mate drop,
the default), as nags_datagen keeps no mate scores; --mate clamp keeps it at +-cap. The game results stay those of
the self-play games, which the base engine played: a teacher's fit to them is a hint, not a strength measure.
Only networks trained on each label set and played against each other decide (docs/STRENGTH_PLAN.md, item 1).
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shards  # noqa: E402

RECORD_BYTES = 32
PIECES = "PNBRQKpnbrqk"


def record_board(rec):
    """(piece placement {square: char}, side to move 'w'/'b') of one 32-byte record (bytes or uint8 array)."""
    rec = bytes(rec)
    occ = int.from_bytes(rec[:8], "little")
    placement, n = {}, 0
    for sq in range(64):
        if occ >> sq & 1:
            byte = rec[8 + n // 2]
            placement[sq] = PIECES[(byte >> 4) if n % 2 else (byte & 15)]
            n += 1
    return placement, "b" if rec[27] else "w"


def record_fen(rec):
    """A FEN for a record: castling rights inferred from kings and rooks on their original squares, no en-passant
    square, halfmove clock 0 (the records hold none of these)."""
    placement, stm = record_board(rec)
    ranks = []
    for r in range(7, -1, -1):
        row, empty = "", 0
        for f in range(8):
            c = placement.get(r * 8 + f)
            if c is None:
                empty += 1
                continue
            row += (str(empty) if empty else "") + c
            empty = 0
        ranks.append(row + (str(empty) if empty else ""))
    castle = ""
    for king, rook, ksq, rsq, flag in (("K", "R", 4, 7, "K"), ("K", "R", 4, 0, "Q"),
                                       ("k", "r", 60, 63, "k"), ("k", "r", 60, 56, "q")):
        if placement.get(ksq) == king and placement.get(rsq) == rook:
            castle += flag
    return f"{'/'.join(ranks)} {stm} {castle or '-'} - 0 1"


def record_score(rec):
    return int.from_bytes(bytes(rec[24:26]), "little", signed=True)


def with_score(rec, score):
    out = bytearray(rec)
    out[24:26] = int(score).to_bytes(2, "little", signed=True)
    return bytes(out)


def engine_sha256(cmd):
    path = shutil.which(cmd[0]) or cmd[0]
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def parse_options(items):
    out = {}
    for item in items or []:
        name, _, value = item.partition("=")
        out[name] = value
    return out


def limit_text(args):
    return (f"nodes:{args.nodes}" if args.nodes else f"depth:{args.depth}" if args.depth
            else f"movetime:{args.movetime}")


class Labeler:
    """One engine process, used by one worker thread at a time."""

    def __init__(self, cmd, options, limit):
        import chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci(cmd)
        if options:
            self.engine.configure({k: v for k, v in options.items()})
        self.limit = limit
        self.name = self.engine.id.get("name", os.path.basename(cmd[0]))

    def game(self, number, records, scale, cap, mate):
        """(labelled records, engine seconds, dropped) for one game's records (a list of 32-byte bytes)."""
        import chess
        import chess.engine
        out, seconds, dropped = [], 0.0, 0
        key = object()  # python-chess sends ucinewgame when the game object changes
        for rec in records:
            board = chess.Board(record_fen(rec))
            t0 = time.perf_counter()
            info = self.engine.analyse(board, self.limit, game=key)
            seconds += time.perf_counter() - t0
            score = info["score"].white()
            if score.is_mate():
                if mate == "drop":
                    dropped += 1
                    continue
                value = cap if score > chess.engine.Cp(0) else -cap
            else:
                value = max(-cap, min(cap, round(score.score() * scale)))
            out.append(with_score(rec, value))
        return out, seconds, dropped

    def close(self):
        self.engine.quit()


def select_games(games, fraction, max_games, seed):
    """The games to relabel, in file order: a random share or number of them (all by default)."""
    rng = np.random.default_rng(seed)
    count = len(games)
    if fraction:
        count = min(count, max(1, round(len(games) * fraction)))
    if max_games:
        count = min(count, max_games)
    pick = np.sort(rng.choice(len(games), count, replace=False)) if count < len(games) else np.arange(len(games))
    return [games[i] for i in pick]


def resume_output(out, header):
    """{game number} already in a (partly) written output, its data cut back to the indexed games. A foreign or
    damaged output is refused."""
    if not os.path.exists(shards.index_path(out)):
        if os.path.exists(out) and os.path.getsize(out):
            sys.exit(f"{out} holds data but has no index: not touched")
        with open(out, "wb"), open(shards.index_path(out), "w", newline="\n") as idx:
            idx.write(header + "\n")
        return set()
    have, games = shards.read_index(out)
    if have != header:
        sys.exit(f"{out}: made with other settings, engine or source:\n  {have}\nnot\n  {header}")
    size = sum(g.bytes for g in games)
    if os.path.getsize(out) < size:
        sys.exit(f"{out}: shorter than its index lists")
    with open(out, "r+b") as fh:
        fh.truncate(size)
        for g in games[-64:]:
            fh.seek(g.offset)
            if f"{shards.fnv1a(fh.read(g.bytes)):016x}" != g.checksum:
                sys.exit(f"{out}: game {g.number} does not match its checksum")
    with open(shards.index_path(out), "rb") as fh:  # drop an unfinished last line
        text = fh.read()
    keep = text[: text.rfind(b"\n") + 1]
    if keep != text:
        with open(shards.index_path(out), "wb") as fh:
            fh.write(keep)
    return {g.number for g in games}


def label(args):
    cmd = args.engine.split() if " " in args.engine and not os.path.exists(args.engine) else [args.engine]
    options = parse_options(args.option)
    import chess.engine
    limit = (chess.engine.Limit(nodes=args.nodes) if args.nodes else chess.engine.Limit(depth=args.depth)
             if args.depth else chess.engine.Limit(time=args.movetime / 1000))
    sha = engine_sha256(cmd)
    probe = Labeler(cmd, options, limit)
    name = probe.name
    labelers = [probe] + [Labeler(cmd, options, limit) for _ in range(args.workers - 1)]
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        for path in shards_of(args.data):
            try:
                src_header, games = shards.read_index(path)
            except (OSError, ValueError) as e:
                sys.exit(f"{path}: needs its nags_datagen shard index (--first-game): {e}")
            if sum(g.bytes for g in games) != os.path.getsize(path):
                sys.exit(f"{path}: its index does not cover the file (an unfinished shard?)")
            chosen = select_games(games, args.fraction, args.max_games, args.select_seed)
            out = os.path.join(args.out_dir, os.path.basename(path))
            if os.path.realpath(out) == os.path.realpath(path):
                sys.exit("--out-dir must not be the source's directory")
            header = (src_header + f" teacher={name.replace(' ', '_')} teacher-sha={(sha or 'unknown')[:16]}"
                      f" teacher-limit={limit_text(args)} teacher-scale={args.scale:g} teacher-cap={args.cap}"
                      f" teacher-mate={args.mate}"
                      + "".join(f" teacher-option={k}:{v}" for k, v in sorted(options.items()))
                      + f" select={args.fraction or 1:g}/{args.max_games or 0}/{args.select_seed}")
            done = resume_output(out, header)
            todo = [g for g in chosen if g.number not in done]
            print(f"{os.path.basename(path)}: {len(chosen)} of {len(games)} games chosen, {len(done)} already "
                  f"labelled, {len(todo)} to label by {name} ({limit_text(args)}, {args.workers} workers)", flush=True)
            side = out + ".teacher.json"
            stats = {"engine": name, "command": cmd, "engine_sha256": sha, "options": options,
                     "limit": limit_text(args), "scale": args.scale, "cap": args.cap, "mate": args.mate,
                     "source": os.path.basename(path), "source_header": src_header, "games": 0, "positions": 0,
                     "dropped": 0, "engine_seconds": 0.0, "wall_seconds": 0.0}
            if os.path.exists(side) and done:
                with open(side) as fh:
                    stats.update({k: v for k, v in json.load(fh).items()
                                  if k in ("games", "positions", "dropped", "engine_seconds", "wall_seconds")})
            free = list(labelers)
            lock = threading.Lock()

            def work(game):
                with lock:
                    lab = free.pop()
                try:
                    with open(path, "rb") as fh:
                        fh.seek(game.offset)
                        raw = fh.read(game.bytes)
                    recs = [raw[i : i + RECORD_BYTES] for i in range(0, len(raw), RECORD_BYTES)]
                    return game, *lab.game(game.number, recs, args.scale, args.cap, args.mate)
                finally:
                    with lock:
                        free.append(lab)

            t0, last = time.time(), time.time()
            with open(out, "ab") as data, open(shards.index_path(out), "a", newline="\n") as idx, \
                    concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                for n, fut in enumerate(concurrent.futures.as_completed([pool.submit(work, g) for g in todo]), 1):
                    game, recs, seconds, dropped = fut.result()
                    blob = b"".join(recs)
                    data.write(blob)
                    data.flush()
                    result = "0.5" if game.result == 0.5 else str(int(game.result))
                    idx.write(f"{game.number} {len(recs)} {len(blob)} {result} {shards.fnv1a(blob):016x}\n")
                    idx.flush()
                    stats["games"] += 1
                    stats["positions"] += len(recs)
                    stats["dropped"] += dropped
                    stats["engine_seconds"] += seconds
                    if time.time() - last > 30 or n == len(todo):
                        last = time.time()
                        print(f"  games {n}/{len(todo)}  positions {stats['positions']}  "
                              f"{stats['engine_seconds'] / max(1, stats['positions'] + stats['dropped']) * 1000:.1f} "
                              "ms engine time a position", flush=True)
            stats["wall_seconds"] += time.time() - t0
            labelled = stats["positions"] + stats["dropped"]
            stats["engine_seconds_per_position"] = stats["engine_seconds"] / labelled if labelled else None
            with open(side + ".tmp", "w") as fh:
                json.dump(stats, fh, indent=1)
            os.replace(side + ".tmp", side)
            print(f"wrote {out}: {stats['games']} games, {stats['positions']} positions ({stats['dropped']} dropped "
                  f"for mate scores)", flush=True)
    finally:
        for lab in labelers:
            lab.close()
    return 0


def shards_of(patterns):
    out = []
    for p in patterns:
        if any(c in p for c in "*?["):
            import glob
            matches = sorted(glob.glob(p))
            if not matches:
                sys.exit(f"no files match {p}")
            out.extend(matches)
        else:
            out.append(p)
    return out


def game_records(path):
    """(seed, {game number: uint8 [n, 32] records}) of an indexed shard."""
    header, games = shards.read_index(path)
    raw = np.fromfile(path, np.uint8)
    out = {}
    for g in games:
        out[g.number] = raw[g.offset : g.offset + g.bytes].reshape(-1, RECORD_BYTES)
    return shards.header_fields(header).get("seed"), out


def white_scores(recs):
    return recs[:, 24:26].copy().view("<i2").reshape(-1).astype(np.float64)


def fit_scale(scores, results):
    """(K, mean squared error) of sigmoid(score / K) against the results, K minimising the error."""
    def err(k):
        return float(np.mean((1 / (1 + np.exp(-scores / k)) - results) ** 2))
    grid = np.exp(np.linspace(np.log(50), np.log(3000), 120))
    k = float(grid[int(np.argmin([err(x) for x in grid]))])
    lo, hi = k / 1.05, k * 1.05
    for _ in range(40):  # golden-section refinement
        a, b = hi - (hi - lo) / 1.618, lo + (hi - lo) / 1.618
        lo, hi = (lo, b) if err(a) < err(b) else (a, hi)
    k = (lo + hi) / 2
    return k, err(k)


def compare(args):
    names = args.names or ["base"] + [os.path.basename(os.path.dirname(os.path.abspath(p))) or p for p in args.labels]
    if len(names) != len(args.labels) + 1:
        sys.exit("--names: one for the base and one for each --labels file")
    base_seed, base = game_records(args.base)
    sets = []
    for path in args.labels:
        seed, games = game_records(path)
        if seed != base_seed:
            sys.exit(f"{path}: seed {seed}, the base's is {base_seed}")
        sets.append(games)
    common = sorted(set(base).intersection(*[set(s) for s in sets]))
    if not common:
        sys.exit("no game is in every file")
    columns = [[] for _ in range(len(sets) + 1)]
    results, pieces, mismatched = [], [], 0
    strip = lambda r: np.delete(r, [24, 25], axis=1)  # noqa: E731
    for number in common:
        rows = [base[number]] + [s[number] for s in sets]
        if all(len(x) == len(rows[0]) for x in rows):
            picks = [np.arange(len(rows[0]))] * len(rows)
        else:  # a teacher dropped positions (mate scores): match the records by content, in order
            picks, kept = [], None
            keys = [[bytes(r) for r in strip(x)] for x in rows]
            for j, k in enumerate(keys):
                pos, i = [], 0
                for key in keys[0]:
                    while i < len(k) and k[i] != key and key in k[i:]:
                        i += 1
                    pos.append(i if i < len(k) and k[i] == key else -1)
                    i += pos[-1] >= 0
                picks.append(np.array(pos))
            kept = np.all([p >= 0 for p in picks], axis=0)
            picks = [p[kept] for p in picks]
            mismatched += int((~kept).sum())
        for j, (x, idx) in enumerate(zip(rows, picks)):
            if not np.array_equal(strip(x[idx]), strip(rows[0][picks[0]])):
                sys.exit(f"game {number}: the label sets hold different positions")
            columns[j].append(white_scores(x[idx]))
        pick = base[number][picks[0]]
        results.append(pick[:, 26].astype(np.float64) / 2)
        pieces.append(np.unpackbits(pick[:, :8], axis=1).sum(axis=1))
    scores = [np.concatenate(c) for c in columns]
    res, pc = np.concatenate(results), np.concatenate(pieces)
    report = {"games": len(common), "positions": int(len(res)), "left_out": mismatched, "sets": []}
    k_base = None
    for j, (name, s) in enumerate(zip(names, scores)):
        k, e = fit_scale(s, res)
        k_base = k if j == 0 else k_base
        cost = None
        if j:
            side = args.labels[j - 1] + ".teacher.json"
            if os.path.exists(side):
                with open(side) as fh:
                    cost = json.load(fh).get("engine_seconds_per_position")
        scaled = s * (k_base / k)
        phases = {}
        for label_, lo, hi in (("<=10 pieces", 0, 10), ("11-20", 11, 20), ("21-32", 21, 32)):
            m = (pc >= lo) & (pc <= hi)
            if m.any():
                phases[label_] = {"positions": int(m.sum()), "fit_error": float(
                    np.mean((1 / (1 + np.exp(-s[m] / k)) - res[m]) ** 2))}
        report["sets"].append({
            "name": name, "fit_scale": k, "fit_error": e,
            "error_at_400": float(np.mean((1 / (1 + np.exp(-s / 400)) - res) ** 2)),
            "scale_to_base": k_base / k, "mean_abs_diff_to_base_cp": float(np.mean(np.abs(scaled - scores[0]))),
            "correlation_with_base": float(np.corrcoef(s, scores[0])[0, 1]) if len(s) > 1 else None,
            "engine_seconds_per_position": cost, "by_pieces": phases})
    print(f"{report['games']} games, {report['positions']} positions in every set"
          + (f" ({mismatched} left out: dropped by a teacher)" if mismatched else ""))
    print(f"{'set':<20} {'fit K':>7} {'fit error':>10} {'at K=400':>9} {'to base':>8} {'|diff| cp':>9} "
          f"{'corr':>6} {'ms/pos':>8}")
    for s in report["sets"]:
        ms = f"{s['engine_seconds_per_position'] * 1000:.1f}" if s["engine_seconds_per_position"] else "-"
        print(f"{s['name']:<20} {s['fit_scale']:7.0f} {s['fit_error']:10.6f} {s['error_at_400']:9.6f} "
              f"{s['scale_to_base']:8.3f} {s['mean_abs_diff_to_base_cp']:9.1f} {s['correlation_with_base']:6.3f} {ms:>8}")
        print("    fit error by pieces: " + ", ".join(f"{k} {v['fit_error']:.6f} ({v['positions']})"
                                                     for k, v in s["by_pieces"].items()))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label", help="relabel games of indexed shards with a UCI engine")
    lb.add_argument("--data", nargs="+", required=True, help="nags_datagen shards with their .games index")
    lb.add_argument("--engine", required=True, help="the teacher: a UCI engine command")
    lb.add_argument("--option", action="append", metavar="NAME=VALUE", help="a UCI option for the teacher")
    g = lb.add_mutually_exclusive_group(required=True)
    g.add_argument("--nodes", type=int)
    g.add_argument("--depth", type=int)
    g.add_argument("--movetime", type=int, metavar="MS", help="milliseconds a position (machine-dependent)")
    lb.add_argument("--workers", type=int, default=1, help="engine processes (one thread each, unless --option)")
    lb.add_argument("--fraction", type=float, default=0.0, help="share of each shard's games to relabel")
    lb.add_argument("--max-games", type=int, default=0, help="at most this many games of each shard")
    lb.add_argument("--select-seed", type=int, default=1, help="which games the sample takes")
    lb.add_argument("--scale", type=float, default=1.0,
                    help="multiply the teacher's centipawns by this (compare reports the factor that puts a "
                         "teacher on the base's scale)")
    lb.add_argument("--cap", type=int, default=4000, help="scores are clamped to +-cap")
    lb.add_argument("--mate", choices=["drop", "clamp"], default="drop",
                    help="a position the teacher scores as mate: dropped (as nags_datagen does) or kept at +-cap")
    lb.add_argument("--out-dir", required=True, help="where the relabelled shards go (same file names)")
    cp = sub.add_parser("compare", help="compare label sets of the same games")
    cp.add_argument("--base", required=True, help="the original shard")
    cp.add_argument("--labels", nargs="+", required=True, help="relabelled versions of it (rescore label output)")
    cp.add_argument("--names", nargs="+", help="names for the base and each label set")
    cp.add_argument("--json", help="also write the report here")
    args = ap.parse_args(argv)
    if args.cmd == "label":
        if args.workers < 1:
            sys.exit("--workers must be at least 1")
        return label(args)
    compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
