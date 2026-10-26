# Results from the training PC

Results of the tasks in `docs/PC_TASKS.md`, run on the Windows PC (AMD Ryzen 5 3500X, 6 cores / 6 threads,
RTX 3060 12 GB, 16 GB RAM, Visual Studio 2019 Build Tools, Python 3.13) from branch `pc/results`, which
starts at `integ/round3` `7464810`. `$T` = 6, `$C` = 5. Commands ran from `C:\Users\Alex\Documents\code\NAGS-round3`;
`$PY` is `C:\Users\Alex\Documents\code\NAGS-main\.venv\Scripts\python.exe`.

The user chose to run sections 0-5 first and report before section 6, to drop the planned kb8 training for
now, and to keep their rule that GPU training never overlaps data generation or a match (stricter than
rule 5).

## 0. Setup and checks (2026-10-26)

Checkout: `git worktree add ..\NAGS-round3 origin/integ/round3`, then `git switch -c pc/results`. The first
attempt started downloading the 1 GB Git LFS file `AJ-CORR-PGN-000.pgn` into the worktree, which no task
needs; it was stopped and the worktree recreated with `GIT_LFS_SKIP_SMUDGE=1` (1 s instead of minutes).
Build: `cmake -B build`, `cmake --build build --config Release --parallel` (32 s).

| Check | Result | Expected |
|---|---|---|
| `ctest --test-dir build --build-config Release` | `100% tests passed, 0 tests failed out of 49` | 49/49 |
| `nags_basic bench` | `Nodes searched  : 1417097` | 1417097 |
| `nags_fast bench` | `Nodes searched  : 1340613` | 1340613 |
| `nags_enhanced bench` | `Nodes searched  : 1528850` | 1528850 |
| `nags bench` | `Nodes searched  : 1528850` | 1528850 |
| `pip install -r requirements.txt -r requirements-onnx.txt` | the CUDA torch kept; torch_geometric 2.8.1 and onnxruntime 1.31.1 added | keeps CUDA torch |
| torch | `2.11.0+cu128 True NVIDIA GeForce RTX 3060` | `True` and the GPU |
| `pytest -q -rs` | `359 passed, 1 skipped, 794 warnings in 383.44s (0:06:23)`; skipped: `tests\python\test_funnel.py:148: sends SIGINT to a POSIX process group (Ctrl-C)` | all pass, the funnel SIGINT test skipped |

All as expected. The two Windows-only failures seen on `main` (`test_sprt.py::test_engine_paths_stay_whole`,
`test_ladder.py::test_anchor_and_rate`) pass on `integ/round3`. Time: about 30 minutes, mostly pytest and the
LFS detour.

## 1. Cross-machine check of resumable self-play (2026-10-26)

As in `docs/PC_TASKS.md` with `$T` = 6: `nags_datagen --out runs\xm\pc.bin --nodes 5000 --seed 77 --first-game 0
--games 400 --threads 6` started with `Start-Process`, killed with `Stop-Process -Force` after 15 s (the file
had 971,616 bytes), then the same command again.

```
nags_datagen: games 0-399, 323 already written, 77 to play
nags_datagen: evaluation NNUE embedded
1 shard(s), 400 games: consistent
games 400 identical to the VM: True first differing game: []
header equal: True
# nags_datagen games v2 seed=77 nodes=5000 random-plies=8 max-opening-score=400 hash=16 repetition=first format=bin network=embedded network-fnv=c7c60e3d6a52e155 search=1528850
```

Kill point N = 323. Time: 19 s in all. The MSVC build plays the VM's (GCC) games exactly, across a hard kill
and a resume at 6 threads, so data from this PC and from the VM are the same games for the same seed.

Sections 3 and 4 name `runs\xm\ref.bin`, which no step creates (the repo holds only its index,
`docs/data/ref_seed77_n5000_g400.games`). Since `runs\xm\pc.bin` holds the same 400 games (checked above),
sections 3 and 4 use `runs\xm\pc.bin` instead.
