#!/usr/bin/env python3
"""Deeper labels: rescore nags_datagen shards with a UCI engine, and compare label sets.

NAGS's training labels come from NAGS only (docs/STRENGTH_PLAN.md: the engine stays its own, for any tournament).
The teacher for training data is nags_enhanced with more nodes; another engine may be given to `label` and
`compare` to measure how its scores differ, but such labels are not trained on.

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

  python tools/nnue/rescore.py label --data data/selfplay_9101.bin --engine build/nags_enhanced \\
      --nodes 40000 --fraction 0.1 --workers 5 --out-dir data/deep40k
  python tools/nnue/rescore.py compare --base data/selfplay_9101.bin --labels data/deep40k/selfplay_9101.bin

Exact positions only. A plain 32-byte record holds the pieces and the side to move but not the castling rights,
the en-passant square or the move counters, so it does not say which position it was, and none of that is
guessed. `label` reads shards written with `nags_datagen --record-state on` (header `state=1`), whose records
carry the rest of the position in their reserved bytes, and refuses any other. For shards without it, relabel
during play instead: `nags_datagen --relabel-nodes N --relabel-select random:1` with the shard's seed and game
range plays the same games and labels every position with the deeper search (docs/NNUE.md).

Accounting survives a stop: each game's positions, mate drops, engine seconds and nodes are appended to
<out>.teacher.games before the game's index line, and <out>.teacher.json is computed from that file for the
games the index holds, so a resumed run counts all of its work once. The header fingerprints what the labels
depend on: the source shard's content (SHA-256), the engine binary, and every option whose value is a file (an
EvalFile replaced at the same path is a different teacher), so a resume with any of them changed is refused.

A position where the teacher finds a mate is dropped (--mate drop, the default), as nags_datagen keeps no mate
scores; --mate clamp keeps it at +-cap. The game results stay those of the self-play games: a label set's fit to
them is a hint, not a strength measure. Only networks trained on each label set and played against each other
decide (docs/STRENGTH_PLAN.md, item 1).
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


def record_state(rec):
    """(castling, en passant, halfmove clock, fullmove number) as FEN fields from a record's reserved bytes
    (nags_datagen --record-state on), or None for a record written without them."""
    rec = bytes(rec)
    full = rec[30] | rec[31] << 8
    if full == 0:
        return None
    c = rec[28]
    castle = "".join(ch for bit, ch in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if c & bit) or "-"
    ep = "-" if not c & 0x80 else "abcdefgh"[(c >> 4) & 7] + ("6" if rec[27] == 0 else "3")
    return castle, ep, rec[29], full


def with_state(rec, castling="", ep_file=None, halfmove=0, fullmove=1):
    """A record with its state bytes set (castling as 'KQkq' letters, ep_file 0-7 or None)."""
    out = bytearray(rec)
    out[28] = sum(bit for bit, ch in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if ch in castling) | (
        0x80 | ep_file << 4 if ep_file is not None else 0)
    out[29] = min(255, halfmove)
    out[30:32] = max(1, min(65535, fullmove)).to_bytes(2, "little")
    return bytes(out)


def record_board_fen(rec):
    """The placement and side to move of a record, as the first two FEN fields."""
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
    return f"{'/'.join(ranks)} {stm}"


def record_fen(rec):
    """The record's exact position as a FEN. A record without state (written without --record-state) raises
    ValueError: its castling rights, en-passant square and move counters are unknown, and are not guessed."""
    state = record_state(rec)
    if state is None:
        raise ValueError("a record without its state (castling, en passant, move counters): written without "
                         "nags_datagen --record-state on")
    return f"{record_board_fen(rec)} {state[0]} {state[1]} {state[2]} {state[3]}"


def record_score(rec):
    return int.from_bytes(bytes(rec[24:26]), "little", signed=True)


def with_score(rec, score):
    out = bytearray(rec)
    out[24:26] = int(score).to_bytes(2, "little", signed=True)
    return bytes(out)


def file_sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def engine_sha256(cmd):
    return file_sha256(shutil.which(cmd[0]) or cmd[0])


def option_files(options):
    """{option: SHA-256} of the options whose value is a file (an EvalFile, a book): what the engine reads."""
    return {k: file_sha256(v) for k, v in sorted(options.items()) if v and os.path.isfile(v)}


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
        """(labelled records, engine seconds, dropped, nodes) for one game's records (a list of 32-byte bytes)."""
        import chess
        import chess.engine
        out, seconds, dropped, nodes = [], 0.0, 0, 0
        key = object()  # python-chess sends ucinewgame when the game object changes
        for rec in records:
            board = chess.Board(record_fen(rec))
            t0 = time.perf_counter()
            info = self.engine.analyse(board, self.limit, game=key)
            seconds += time.perf_counter() - t0
            nodes += int(info.get("nodes", 0))
            score = info["score"].white()
            if score.is_mate():
                if mate == "drop":
                    dropped += 1
                    continue
                value = cap if score > chess.engine.Cp(0) else -cap
            else:
                value = max(-cap, min(cap, round(score.score() * scale)))
            out.append(with_score(rec, value))
        return out, seconds, dropped, nodes

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


def accounting_path(out):
    return out + ".teacher.games"


def read_accounting(out, keep):
    """{game: (positions, dropped, engine seconds, nodes)} from <out>.teacher.games for the games in `keep` (the
    index's), rewriting the file without the others (games cut by a stop, to be labelled again)."""
    path, rows = accounting_path(out), {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 5 and line.endswith("\n"):
                    g = int(parts[0])
                    if g in keep:
                        rows[g] = (int(parts[1]), int(parts[2]), float(parts[3]), int(parts[4]))
    with open(path + ".tmp", "w", newline="\n") as fh:
        for g, (p, d, s_, n) in rows.items():
            fh.write(f"{g} {p} {d} {s_:.6f} {n}\n")
    os.replace(path + ".tmp", path)
    return rows


def resume_output(out, header):
    """{game number} already in a (partly) written output, its data cut back to the indexed games. A foreign or
    damaged output is refused."""
    if not os.path.exists(shards.index_path(out)):
        if os.path.exists(out) and os.path.getsize(out):
            sys.exit(f"{out} holds data but has no index: not touched")
        with open(out, "wb"), open(shards.index_path(out), "w", newline="\n") as idx:
            idx.write(header + "\n")
        if os.path.exists(accounting_path(out)):
            os.remove(accounting_path(out))  # no game of this output exists
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
    files = option_files(options)
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
            if shards.header_fields(src_header).get("state") != "1":
                sys.exit(f"{path}: its records hold no castling rights, en-passant square or move counters "
                         "(written without nags_datagen --record-state on), so the positions are not known exactly "
                         "and are not guessed. Relabel during play instead: nags_datagen --relabel-nodes N "
                         "--relabel-select random:1 with this shard's seed and game range")
            source_sha = file_sha256(path)
            chosen = select_games(games, args.fraction, args.max_games, args.select_seed)
            out = os.path.join(args.out_dir, os.path.basename(path))
            if os.path.realpath(out) == os.path.realpath(path):
                sys.exit("--out-dir must not be the source's directory")
            header = (src_header + f" teacher={name.replace(' ', '_')} teacher-sha={(sha or 'unknown')[:16]}"
                      f" teacher-limit={limit_text(args)} teacher-scale={args.scale:g} teacher-cap={args.cap}"
                      f" teacher-mate={args.mate} source-sha={(source_sha or 'unknown')[:16]}"
                      + "".join(f" teacher-option={k}:{v.replace(' ', '_')}" for k, v in sorted(options.items()))
                      + "".join(f" teacher-file={k}:{(h or 'unknown')[:16]}" for k, h in files.items())
                      + f" select={args.fraction or 1:g}/{args.max_games or 0}/{args.select_seed}")
            done = resume_output(out, header)
            accounting = read_accounting(out, done)
            todo = [g for g in chosen if g.number not in done]
            print(f"{os.path.basename(path)}: {len(chosen)} of {len(games)} games chosen, {len(done)} already "
                  f"labelled, {len(todo)} to label by {name} ({limit_text(args)}, {args.workers} workers)", flush=True)
            side = out + ".teacher.json"
            sittings = []
            if os.path.exists(side) and done:
                try:
                    with open(side) as fh:
                        sittings = list(json.load(fh).get("sittings", []))
                except (OSError, ValueError):
                    pass
            sitting = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_seconds": 0.0, "games": 0}
            sittings.append(sitting)
            t0 = time.time()

            def save():
                sitting["wall_seconds"] = round(time.time() - t0, 1)
                rows = list(accounting.values())
                labelled = sum(r[0] + r[1] for r in rows)
                stats = {"engine": name, "command": cmd, "engine_sha256": sha, "options": options,
                         "option_files_sha256": files, "limit": limit_text(args), "scale": args.scale,
                         "cap": args.cap, "mate": args.mate, "source": os.path.basename(path),
                         "source_sha256": source_sha, "source_header": src_header, "header": header,
                         "games": len(rows), "positions": sum(r[0] for r in rows), "dropped": sum(r[1] for r in rows),
                         "engine_seconds": sum(r[2] for r in rows), "nodes": sum(r[3] for r in rows),
                         "games_without_accounting": len(done - set(accounting)),
                         "wall_seconds": round(sum(x["wall_seconds"] for x in sittings), 1), "sittings": sittings,
                         "engine_seconds_per_position": sum(r[2] for r in rows) / labelled if labelled else None,
                         "nodes_per_position": sum(r[3] for r in rows) / labelled if labelled else None}
                with open(side + ".tmp", "w") as fh:
                    json.dump(stats, fh, indent=1)
                os.replace(side + ".tmp", side)
                return stats

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

            last = time.time()
            # Per game: its bytes, then its accounting line, then its index line. A game in the index therefore
            # always has its accounting, and a stop between the writes leaves a game that is labelled again.
            with open(out, "ab") as data, open(shards.index_path(out), "a", newline="\n") as idx, \
                    open(accounting_path(out), "a", newline="\n") as acc, \
                    concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                for n, fut in enumerate(concurrent.futures.as_completed([pool.submit(work, g) for g in todo]), 1):
                    game, recs, seconds, dropped, nodes = fut.result()
                    blob = b"".join(recs)
                    data.write(blob)
                    data.flush()
                    acc.write(f"{game.number} {len(recs)} {dropped} {seconds:.6f} {nodes}\n")
                    acc.flush()
                    result = "0.5" if game.result == 0.5 else str(int(game.result))
                    idx.write(f"{game.number} {len(recs)} {len(blob)} {result} {shards.fnv1a(blob):016x}\n")
                    idx.flush()
                    accounting[game.number] = (len(recs), dropped, seconds, nodes)
                    sitting["games"] += 1
                    if time.time() - last > 30 or n == len(todo):
                        last = time.time()
                        stats = save()
                        print(f"  games {n}/{len(todo)}  positions {stats['positions']}  "
                              f"{(stats['engine_seconds_per_position'] or 0) * 1000:.1f} ms engine time a position",
                              flush=True)
            stats = save()
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
