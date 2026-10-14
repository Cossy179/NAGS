"""Deterministic NNUE networks in every file format the engine reads.

The byte layout is written out here independently of tools/nnue/train.py so
that the formats stay pinned even if the trainer changes: a network trained
with the trainer of commit bb1e03c (format versions 1-3) must keep loading
and evaluating exactly as it did then. The weights come from PCG64's raw
output, whose stream numpy guarantees to stay the same across versions.
"""

import struct

import numpy as np

HIDDEN = 256

# Layout of king buckets "4" in train.py at bb1e03c: 32 squares a1-d8 (rank by
# rank, files a-d) as seen after mirroring.
KB4 = [0, 0, 1, 1] + [2] * 4 + [3] * 24
KB16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11] + [12, 12, 13, 13] * 2 + [14, 14, 15, 15] * 2

# name: (version, output buckets, king layout over the 32 left-half squares or
# None, screlu, mirror)
DESIGNS = {
    "v1": (1, 1, None, False, False),
    "v2_b8": (2, 8, None, False, False),
    "v3_screlu_b8": (3, 8, None, True, False),
    "v3_mirror": (3, 1, [0] * 32, False, True),
    "v3_kb4_screlu_b8": (3, 8, KB4, True, True),
    "v3_kb16_screlu_b4": (3, 4, KB16, True, True),
}


def _layout64(half):
    layout = np.zeros(64, np.uint8)
    for sq in range(64):
        f, r = sq % 8, sq // 8
        layout[sq] = half[r * 4 + (f if f < 4 else 7 - f)]
    return layout


def _ints(raw, lo, hi):
    return (raw % np.uint64(hi - lo + 1)).astype(np.int64) + lo


def network_bytes(name, seed=2026):
    version, buckets, half, screlu, mirror = DESIGNS[name]
    kb = 1 if half is None else max(half) + 1
    raw = np.random.PCG64(seed + sum(map(ord, name))).random_raw
    ft_w = _ints(raw(kb * 768 * HIDDEN), -100, 100)
    ft_b = _ints(raw(HIDDEN), -20, 120)
    out_w = _ints(raw(buckets * 2 * HIDDEN), -100, 100)
    out_b = _ints(raw(buckets), -50000, 50000)
    head = b"NAGSNNUE"
    if version == 1:
        head += struct.pack("<II", 1, HIDDEN)
    elif version == 2:
        head += struct.pack("<III", 2, HIDDEN, buckets)
    else:
        flags = (1 if screlu else 0) | (2 if mirror else 0)
        layout = np.zeros(64, np.uint8) if half is None else _layout64(half)
        head += struct.pack("<IIIII", 3, HIDDEN, buckets, kb, flags) + layout.tobytes()
    return (head + ft_w.astype("<i2").tobytes() + ft_b.astype("<i2").tobytes() + out_w.astype("<i2").tobytes()
            + out_b.astype("<i4").tobytes())


# Positions: a FEN and moves played from it (so the engine evaluates through
# its incremental accumulators, including king moves across buckets and the
# mirror line, castling, captures, en passant and promotions).
POSITIONS = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", []),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", ["e2e4", "d7d5", "e4d5", "d8d5", "g1f3", "c8g4"]),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3", ["g1e2", "g8f6", "e1g1", "f8c5"]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", ["e1c1", "e8g8", "c1b1", "g8h8"]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", ["g1h1", "g8h8", "h1g1"]),
    ("8/8/1k6/8/8/8/6K1/7Q b - - 0 1", ["b6c5", "g2f3", "c5d4", "f3e2", "d4e5"]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", ["e2e4", "h5h8", "b4b1"]),  # f4e3 e.p. would be illegal
    ("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1", ["d7d5", "e5d6", "e8d7", "e1e2"]),
    ("8/P6k/8/8/8/8/6K1/8 w - - 0 1", ["a7a8n", "h7g6", "a8b6"]),
    ("4k3/1P6/8/8/8/8/6p1/4K3 b - - 0 1", ["g2g1r", "e1d2", "g1g8", "b7b8q"]),
]
