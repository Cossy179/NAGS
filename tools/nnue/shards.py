"""nags_datagen shard indexes (<out>.games, written with --first-game): reading and checking them.

    python tools/nnue/shards.py check data/selfplay_*.bin

The index's first line holds the settings, the network and the search fingerprint; every further line is one
finished game: number, records, bytes, result and the FNV-1a 64 checksum of its bytes, in the order the games
are in the data file. `check` verifies each shard against its data (sizes and checksums) and the shards
against each other: the same header apart from the seed (settings, network, search) and no game of a seed
twice. Shards that pass were made by one build and one network and hold every game once, wherever they were
generated.

Version 3 (written by tools/nnue/mix.py and tools/nnue/tb_data.py, never by nags_datagen) adds two fields to every
line, so that a file that is not one engine's self-play still says what its units are:

    <unit> <records> <bytes> <result> <checksum> <kind> <origin>

kind g is a whole game (origin <seed>:<game>@<file>), p a run of independent positions (any record boundary
inside it is a safe place to split), and b a block of consecutive records of a file that had no index (it may hold
parts of games that continue in its neighbouring blocks: no boundary next to a block is known to be safe).
"""

import argparse
import os
import re
import sys
from collections import namedtuple

Game = namedtuple("Game", "number records bytes result checksum offset kind origin", defaults=("g", None))
_LINE = re.compile(r"(\d+) (\d+) (\d+) (0|0\.5|1) ([0-9a-f]{16})")
_LINE3 = re.compile(r"(\d+) (\d+) (\d+) (0|0\.5|1) ([0-9a-f]{16}) ([gpb]) (\S+)")
HEADER = "# nags_datagen games v2 "
HEADER3 = "# nags_datagen games v3 "
KINDS = {"g": "game", "p": "independent positions", "b": "block of a file without an index"}


def fnv1a(data, h=0xCBF29CE484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv1a_many(blobs, group=4096):
    """fnv1a of each bytes object, vectorised across them (the same values; a pure Python loop over the bytes of
    a 100M-position mixture would take hours)."""
    import numpy as np
    out = [0] * len(blobs)
    order = sorted(range(len(blobs)), key=lambda i: len(blobs[i]))
    prime = np.uint64(0x100000001B3)
    for g in range(0, len(order), group):
        ids = order[g : g + group]
        lengths = np.array([len(blobs[i]) for i in ids])
        width = int(lengths.max()) if len(ids) else 0
        buf = np.zeros((len(ids), width), np.uint64)
        for row, i in enumerate(ids):
            buf[row, : len(blobs[i])] = np.frombuffer(blobs[i], np.uint8)
        h = np.full(len(ids), 0xCBF29CE484222325, np.uint64)
        with np.errstate(over="ignore"):
            for col in range(width):
                live = lengths > col
                h = np.where(live, (h ^ buf[:, col]) * prime, h)
        for row, i in enumerate(ids):
            out[i] = int(h[row])
    return out


def index_path(data_path):
    return str(data_path) + ".games"


def read_index(data_path):
    """(header, [Game ...]) of a shard's complete index lines, in file order, with each game's byte offset in
    the data file. An unfinished last line (a stop while it was written) is left out, as nags_datagen does.
    Raises ValueError for a missing or foreign header, a damaged line or a game listed twice."""
    with open(index_path(data_path), "rb") as fh:
        text = fh.read().decode("ascii", errors="replace")
    lines = [l[:-1] if l.endswith("\r") else l for l in text.split("\n")[:-1]]  # after the last newline: unfinished
    v3 = bool(lines) and lines[0].startswith(HEADER3)
    if not lines or not (v3 or lines[0].startswith(HEADER)):
        raise ValueError(f"{index_path(data_path)}: not a nags_datagen shard index (v2 or v3)")
    seed = header_fields(lines[0]).get("seed")
    games, seen, offset = [], set(), 0
    for i, line in enumerate(lines[1:], start=2):
        m = (_LINE3 if v3 else _LINE).fullmatch(line)
        if not m:
            raise ValueError(f"{index_path(data_path)}: damaged line {i}: {line!r}")
        number, records, size = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if number in seen:
            raise ValueError(f"{index_path(data_path)}: game {number} listed twice")
        seen.add(number)
        kind, origin = (m.group(6), m.group(7)) if v3 else ("g", f"{seed}:{number}")
        games.append(Game(number, records, size, float(m.group(4)), m.group(5), offset, kind, origin))
        offset += size
    return lines[0], games


def game_ids(games):
    """{(seed, game)} of the whole games among the units: their self-play identity, whatever file they are in."""
    out = set()
    for g in games:
        if g.kind == "g" and g.origin and ":" in g.origin:
            seed, _, rest = g.origin.partition(":")
            out.add((seed, int(rest.split("@")[0])))
    return out


def header_fields(header):
    """{name: value} of an index header (v2 or v3: both prefixes are the same length)."""
    return dict(f.split("=", 1) for f in header[len(HEADER):].split() if "=" in f)


def check(paths, verify="tail"):
    """Problems found in a set of shards (an empty list if none): see the module docstring. `verify`: "all"
    checks every game's checksum, "tail" the last 64 of each shard (where a stop or a crash leaves damage),
    "none" only sizes, headers and game numbers."""
    problems, headers, owner = [], {}, {}
    for p in paths:
        try:
            header, games = read_index(p)
        except (OSError, ValueError) as e:
            problems.append(str(e))
            continue
        headers[p] = header
        indexed = sum(g.bytes for g in games)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        if size != indexed:
            problems.append(f"{p}: {size} bytes, its index lists {indexed} (run its nags_datagen command again to "
                            "finish or repair it)")
        if verify != "none" and size >= indexed:
            with open(p, "rb") as fh:
                for g in games if verify == "all" else games[-64:]:
                    fh.seek(g.offset)
                    if f"{fnv1a(fh.read(g.bytes)):016x}" != g.checksum:
                        problems.append(f"{p}: game {g.number} does not match its checksum")
        for key in game_ids(games):
            if key in owner:
                problems.append(f"game {key[1]} of seed {key[0]} is in both {owner[key]} and {p}")
            owner.setdefault(key, p)
    distinct = {}
    for p, h in headers.items():
        f = header_fields(h)
        f.pop("seed", None)  # several seeds may be combined; everything else must agree
        distinct.setdefault(tuple(sorted(f.items())), []).append(p)
    if len(distinct) > 1:
        parts = ["; ".join(f"{k}={v}" for k, v in key) + ": " + ", ".join(ps) for key, ps in distinct.items()]
        problems.append("the shards come from different settings, networks or builds:\n  " + "\n  ".join(parts))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="verify shards and their indexes")
    c.add_argument("data", nargs="+", help="shard data files (their .games index next to them)")
    c.add_argument("--checksums", choices=["all", "tail", "none"], default="tail",
                   help="games whose checksums are checked: all, the last 64 of each shard (default), or none")
    args = ap.parse_args(argv)
    problems = check(args.data, verify=args.checksums)
    for p in problems:
        print(p)
    if not problems:
        games = sum(len(read_index(p)[1]) for p in args.data)
        print(f"{len(args.data)} shard(s), {games} games: consistent")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
