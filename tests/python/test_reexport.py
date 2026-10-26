"""tools/nnue/reexport.py: a checkpoint exported again with corrected output biases, checked against the
network it belongs to."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import reexport  # noqa: E402
import train  # noqa: E402
from test_nnue import _dataset  # noqa: E402


def test_reexport_corrects_only_the_output_biases(tmp_path, capsys):
    _dataset(tmp_path / "d.txt", n=800)
    pack.main([str(tmp_path / "d.txt")])
    data, net = str(tmp_path / "d.bin"), tmp_path / "n.nnue"
    train.main(["--data", data, "--out", str(net), "--hidden", "16", "--batch", "64", "--threads", "1", "--device",
                "cpu", "--buckets", "8", "--king-buckets", "4", "--activation", "screlu", "--epochs", "2"])
    out = tmp_path / "corrected.nnue"
    assert reexport.main(["--ckpt", str(net) + ".ckpt", "--data", data, "--out", str(out), "--check", str(net),
                          "--positions", "200"]) == 0
    text = capsys.readouterr().out
    assert "byte for byte" in text and "corrected export on 200 other positions" in text
    a, b = train.read_net(str(net)), train.read_net(str(out))
    assert all(np.array_equal(a[i], b[i]) for i in range(3)) and not np.array_equal(a[3], b[3])
    # Another network's checkpoint is refused before anything is written.
    other = tmp_path / "other.nnue"
    other.write_bytes(net.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(SystemExit, match="not that network's checkpoint"):
        reexport.main(["--ckpt", str(net) + ".ckpt", "--data", data, "--out", str(tmp_path / "x.nnue"), "--check",
                       str(other)])
    assert not (tmp_path / "x.nnue").exists()
