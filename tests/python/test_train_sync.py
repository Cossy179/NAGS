"""tools/nnue/train.py --sync-every (fewer GPU synchronisations, the same network) and --profile."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import train  # noqa: E402
from test_nnue import _dataset  # noqa: E402

COMMON = ["--hidden", "8", "--batch", "32", "--threads", "1", "--device", "cpu", "--epochs", "2", "--buckets", "8",
          "--king-buckets", "4", "--activation", "screlu"]


def test_reading_the_loss_less_often_trains_the_same_network(tmp_path):
    _dataset(tmp_path / "d.txt")
    pack.main([str(tmp_path / "d.txt")])
    data = str(tmp_path / "d.bin")
    train.main(["--data", data, "--out", str(tmp_path / "every.nnue")] + COMMON)
    train.main(["--data", data, "--out", str(tmp_path / "seven.nnue"), "--sync-every", "7"] + COMMON)
    assert (tmp_path / "every.nnue").read_bytes() == (tmp_path / "seven.nnue").read_bytes()


def test_profile_measures_and_writes_nothing(tmp_path, capsys):
    _dataset(tmp_path / "d.txt")
    pack.main([str(tmp_path / "d.txt")])
    out = tmp_path / "p.nnue"
    train.main(["--data", str(tmp_path / "d.bin"), "--out", str(out), "--profile", "6", "--sync-every", "3"] + COMMON)
    text = capsys.readouterr().out
    assert "median ms per batch" in text and "reading it every 3" in text
    assert not out.exists() and not Path(str(out) + ".ckpt").exists() and not Path(str(out) + ".json").exists()
