"""nags_datagen shard indexes (<out>.games, written with --first-game): reading and checking them.

    python tools/nnue/shards.py check data/selfplay_*.bin

The index's first line holds the settings, the network and the search fingerprint; every further line is one
finished game: number, records, bytes, result and the FNV-1a 64 checksum of its bytes, in the order the games
are in the data file. `check` verifies each shard against its data (sizes and checksums) and the shards
against each other: the same header apart from the seed (settings, network, search) and no game of a seed
twice. Shards that pass were made by one build and one network and hold every game once, wherever they were
generated.
"""

import argparse
import os
import re
import sys
from collections import namedtuple

Game = namedtuple("Game", "number records bytes result checksum offset")
_LINE = re.compile(r"(\d+) (\d+) (\d+) (0|0\.5|1) ([0-9a-f]{16})")
HEADER = "# nags_datagen games v2 "


def fnv1a(data, h=0xCBF29CE484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def index_path(data_path):
    return str(data_path) + ".games"


def read_index(data_path):
    """(header, [Game ...]) of a shard's complete index lines, in file order, with each game's byte offset in
    the data file. An unfinished last line (a stop while it was written) is left out, as nags_datagen does.
    Raises ValueError for a missing or foreign header, a damaged line or a game listed twice."""
    with open(index_path(data_path), "rb") as fh:
        text = fh.read().decode("ascii", errors="replace")
    lines = [l[:-1] if l.endswith("\r") else l for l in text.split("\n")[:-1]]  # after the last newline: unfinished
    if not lines or not lines[0].startswith(HEADER):
        raise ValueError(f"{index_path(data_path)}: not a nags_datagen shard index (v2)")
    games, seen, offset = [], set(), 0
    for i, line in enumerate(lines[1:], start=2):
        m = _LINE.fullmatch(line)
        if not m:
            raise ValueError(f"{index_path(data_path)}: damaged line {i}: {line!r}")
        number, records, size = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if number in seen:
            raise ValueError(f"{index_path(data_path)}: game {number} listed twice")
        seen.add(number)
        games.append(Game(number, records, size, float(m.group(4)), m.group(5), offset))
        offset += size
    return lines[0], games


def header_fields(header):
    """{name: value} of an index header."""
    return dict(f.split("=", 1) for f in header[len(HEADER):].split())


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
        seed = header_fields(header).get("seed")
        for g in games:
            if (seed, g.number) in owner:
                problems.append(f"game {g.number} of seed {seed} is in both {owner[seed, g.number]} and {p}")
            owner.setdefault((seed, g.number), p)
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
