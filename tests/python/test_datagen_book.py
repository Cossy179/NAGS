"""nags_datagen --book: games start from the book's positions (drawn by each game's random stream), the shard
header names the book, and a book the board cannot read is refused."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import rescore  # noqa: E402
import shards  # noqa: E402
from test_datagen_relabel import _datagen  # noqa: E402

pytestmark = pytest.mark.skipif(_datagen() is None, reason="nags_datagen not built")

START = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"


def _run(out, *args, check=True):
    return subprocess.run([str(_datagen()), "--out", str(out), "--nodes", "400", "--seed", "8", "--threads", "1",
                           *map(str, args)], capture_output=True, text=True, timeout=600, check=check)


def test_games_start_from_the_book(tmp_path):
    book = tmp_path / "book.epd"
    book.write_text(START + ' id "e4";\n\n')
    out = tmp_path / "b.bin"
    _run(out, "--book", book, "--random-plies", 0, "--first-game", 0, "--games", 3)
    header, games = shards.read_index(str(out))
    assert " book=" in header and header.endswith(":1")
    raw = np.fromfile(out, np.uint8).reshape(-1, 32)
    # No random plies and one start position: every game is the same game, and starts there.
    assert len({g.checksum for g in games}) == 1
    first = rescore.record_fen(raw[0])
    assert first.split()[:2] == START.split()[:2]


def test_without_a_book_nothing_changes(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    _run(a, "--games", 4)
    book = tmp_path / "book.epd"
    book.write_text(START + "\n")
    _run(b, "--games", 4, "--book", book)
    assert a.read_bytes() != b.read_bytes()
    _run(b.with_suffix(".2.bin"), "--games", 4)
    assert a.read_bytes() == b.with_suffix(".2.bin").read_bytes()


@pytest.mark.parametrize("text,message", [("8/8/8/8/8/8/8/8 w - -\n", "not a legal position"),
                                          ("rnbqkbnr/pppppppp w\n", "not a FEN"), ("\n", "no positions")])
def test_bad_books_are_refused(tmp_path, text, message):
    book = tmp_path / "bad.epd"
    book.write_text(text)
    r = _run(tmp_path / "x.bin", "--book", book, "--games", 1, check=False)
    assert r.returncode == 2 and message in r.stderr
