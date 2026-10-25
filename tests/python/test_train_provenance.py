"""tools/nnue/train.py: data fingerprints, the training manifest, and weights-only warm starts (--init-from),
kept apart from exact resumes (--resume)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import train  # noqa: E402
from test_nnue import _dataset  # noqa: E402

COMMON = ["--hidden", "8", "--batch", "64", "--threads", "1", "--device", "cpu"]
DESIGNS = [  # (train.py arguments, format version)
    (["--buckets", "1"], 1),
    (["--buckets", "8"], 2),
    (["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"], 3),
    (["--buckets", "4", "--king-buckets", "mirror"], 3),
    (["--buckets", "8", "--king-buckets", "16", "--activation", "screlu"], 3),
]


def _bin_data(tmp_path, name="d", seed=0):
    text = tmp_path / f"{name}.txt"
    _dataset(text, seed=seed)
    pack.main([str(text)])
    return tmp_path / f"{name}.bin"


@pytest.mark.parametrize("design,version", DESIGNS)
def test_init_from_an_exported_network_is_exact(tmp_path, design, version):
    data = _bin_data(tmp_path)
    src = tmp_path / "src.nnue"
    train.main(["--data", str(data), "--out", str(src), "--epochs", "2"] + COMMON + design)
    manifest = json.loads((tmp_path / "src.nnue.json").read_text())
    assert manifest["architecture"]["format_version"] == version
    # Read back into a fresh model of the same design and exported before any training: the same bytes.
    args = dict(zip(design[::2], design[1::2]))
    model = train.Nnue(8, int(args["--buckets"]), args.get("--king-buckets", "none"),
                       args.get("--activation") == "screlu")
    train.init_from(model, str(src))
    copy = tmp_path / "copy.nnue"
    train.export(model, str(copy))
    assert copy.read_bytes() == src.read_bytes()
    # And from the training checkpoint.
    model2 = train.Nnue(8, int(args["--buckets"]), args.get("--king-buckets", "none"),
                        args.get("--activation") == "screlu")
    train.init_from(model2, str(src) + ".ckpt")
    train.export(model2, str(copy))
    assert copy.read_bytes() == src.read_bytes()


def test_init_from_refuses_another_architecture(tmp_path):
    data = _bin_data(tmp_path)
    kb4 = tmp_path / "kb4.nnue"
    train.main(["--data", str(data), "--out", str(kb4), "--epochs", "1", "--buckets", "8", "--king-buckets", "4",
                "--activation", "screlu"] + COMMON)
    for other in (["--buckets", "8"], ["--buckets", "8", "--king-buckets", "8", "--activation", "screlu"],
                  ["--buckets", "4", "--king-buckets", "4", "--activation", "screlu"],
                  ["--buckets", "8", "--king-buckets", "4"]):
        with pytest.raises(SystemExit, match="architecture|network"):
            train.main(["--data", str(data), "--out", str(tmp_path / "x.nnue"), "--epochs", "1",
                        "--init-from", str(kb4)] + COMMON + other)
        with pytest.raises(SystemExit):
            train.main(["--data", str(data), "--out", str(tmp_path / "y.nnue"), "--epochs", "1",
                        "--init-from", str(kb4) + ".ckpt"] + COMMON + other)


def test_warm_start_is_a_new_run_from_the_weights(tmp_path):
    data = _bin_data(tmp_path)
    design = ["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"]
    src = tmp_path / "src.nnue"
    train.main(["--data", str(data), "--out", str(src), "--epochs", "6"] + COMMON + design)
    warm, cold = tmp_path / "warm.nnue", tmp_path / "cold.nnue"
    train.main(["--data", str(data), "--out", str(warm), "--epochs", "2", "--init-from", str(src)] + COMMON + design)
    train.main(["--data", str(data), "--out", str(cold), "--epochs", "2"] + COMMON + design)
    w = json.loads((tmp_path / "warm.nnue.json").read_text())
    c = json.loads((tmp_path / "cold.nnue.json").read_text())
    assert w["init_from"]["sha256"] == train.file_sha256(str(src), cache=False) and c["init_from"] is None
    # Fresh optimiser and schedule: epoch 1 of 2, not epoch 7 of an old run; it starts where src ended.
    assert [h["epoch"] for h in w["history"]] == [1, 2]
    assert w["history"][0]["train"] < c["history"][0]["train"]
    ckpt = torch.load(str(warm) + ".ckpt", weights_only=False)
    assert ckpt["epoch"] == 2 and ckpt["init_from"]["path"] == str(src)
    # A resumed warm start keeps the weights it started from, and refuses to be told other ones.
    part = tmp_path / "part.nnue"
    train.main(["--data", str(data), "--out", str(part), "--epochs", "3", "--stop-after", "1",
                "--init-from", str(src)] + COMMON + design)
    train.main(["--data", str(data), "--out", str(part), "--epochs", "3", "--resume"] + COMMON + design)
    assert json.loads((tmp_path / "part.nnue.json").read_text())["init_from"]["sha256"] == w["init_from"]["sha256"]
    with pytest.raises(SystemExit, match="started from"):
        train.main(["--data", str(data), "--out", str(part), "--epochs", "3", "--resume", "--init-from",
                    str(warm)] + COMMON + design)


def test_resume_refuses_other_data_with_the_same_count(tmp_path):
    data = _bin_data(tmp_path)
    out = tmp_path / "n.nnue"
    args = ["--data", str(data), "--out", str(out), "--epochs", "3"] + COMMON + ["--buckets", "8"]
    train.main(args + ["--stop-after", "1"])
    raw = bytearray(data.read_bytes())
    raw[24] ^= 1  # one record's score: same size, same number of positions
    data.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="other data"):
        train.main(args + ["--resume"])
    raw[24] ^= 1
    data.write_bytes(bytes(raw))
    train.main(args + ["--resume"])  # the original data resumes


def test_fingerprint_cache_follows_the_file(tmp_path):
    data = _bin_data(tmp_path)
    first = train.data_fingerprint([str(data)])
    side = Path(str(data) + ".sha256")
    assert side.exists() and first["files"][0]["sha256"] in side.read_text()
    assert train.data_fingerprint([str(data)], cache=False) == first
    raw = bytearray(data.read_bytes())
    raw[0] ^= 1
    data.write_bytes(bytes(raw))
    st = data.stat()
    os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # a new modification time
    assert train.data_fingerprint([str(data)])["sha256"] != first["sha256"]
    # The order of the files is part of what was trained on.
    other = _bin_data(tmp_path, "e", seed=1)
    assert (train.data_fingerprint([str(data), str(other)])["sha256"]
            != train.data_fingerprint([str(other), str(data)])["sha256"])


def test_manifest_describes_the_network(tmp_path):
    data = _bin_data(tmp_path)
    out = tmp_path / "n.nnue"
    train.main(["--data", str(data), "--out", str(out), "--epochs", "2", "--buckets", "8", "--king-buckets", "4",
                "--activation", "screlu"] + COMMON)
    m = json.loads((tmp_path / "n.nnue.json").read_text())
    assert m["network_sha256"] == train.file_sha256(str(out), cache=False)
    assert m["architecture"] == {**m["architecture"], "format_version": 3, "hidden": 8, "buckets": 8,
                                 "king_buckets": 4, "mirror": True, "activation": "screlu"}
    assert m["data"]["sha256"] == train.data_fingerprint([str(data)])["sha256"]
    assert m["data"]["positions"] == m["data"]["train"] + m["data"]["validation"] == 400
    assert m["epochs_done"] == 2 and len(m["history"]) == 2 and m["settings"]["lr"] == 1e-3
    assert m["code"]["train_py_sha256"] == train.file_sha256(train.__file__, cache=False)
    assert np.isfinite(m["history"][-1]["val"])


@pytest.mark.skipif(not (ROOT / "nets" / "nags.nnue").exists(), reason="no built-in network")
def test_the_built_in_network_round_trips_exactly(tmp_path):
    """A warm start from nets/nags.nnue begins exactly at the network the engine plays with."""
    src = ROOT / "nets" / "nags.nnue"
    _, ft_b, _, out_b, info = train.read_net(str(src))
    layout = {1: "none", 4: "4", 8: "8", 16: "16"}.get(info["king_buckets"])
    if info["king_buckets"] == 1 and info["mirror"]:
        layout = "mirror"
    model = train.Nnue(len(ft_b), len(out_b), layout, info["screlu"])
    train.init_from(model, str(src))
    copy = tmp_path / "copy.nnue"
    train.export(model, str(copy))
    assert copy.read_bytes() == src.read_bytes()


def test_run_round_design_names():
    import run_round

    assert set(run_round.DESIGN_IDS.values()) == set(run_round.DESIGNS)  # every design has an explicit name
    assert run_round.design_id("current") == "crelu-b8" and run_round.design_id("kb4") == "kb4-screlu-b8"
    if (ROOT / "nets" / "nags.nnue").exists():
        assert run_round.design_of_network(ROOT / "nets" / "nags.nnue") == "kb4-screlu-b8"  # network 8


@pytest.mark.skipif(not (ROOT / "nets" / "nags.nnue").exists(), reason="no built-in network")
def test_run_round_trains_explicit_designs_into_the_historical_files(tmp_path, capsys):
    import run_round

    _dataset(tmp_path / "selfplay_1.txt")
    run_round.main(["--data-dir", str(tmp_path), "--skip-datagen", "--skip-test", "--epochs", "1",
                    "--designs", "crelu-b8", "kb4-screlu-b8", "current", "--device", "cpu",
                    "--warm-start", "builtin"])
    out = capsys.readouterr().out
    assert "built-in network: kb4-screlu-b8" in out and "crelu-b8: not the built-in network's design" in out
    cold = tmp_path / "net_current_400_e1.nnue"  # crelu-b8 and current are one design, trained once
    warm = tmp_path / "net_kb4_400_e1_warm.nnue"
    assert cold.exists() and warm.exists() and not (tmp_path / "net_crelu-b8_400_e1.nnue").exists()
    assert json.loads(Path(str(warm) + ".json").read_text())["init_from"]["path"].endswith("nags.nnue")
    assert json.loads(Path(str(cold) + ".json").read_text())["init_from"] is None
