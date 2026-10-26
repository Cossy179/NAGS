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

## 2. Network 8's training checkpoint (2026-10-26)

Searched `D:\NAGS` and `C:\Users\Alex\Documents\code` (the round's data directory is `D:\NAGS\data`):

```
FullName                                       Length LastWriteTime
D:\NAGS\data\net_kb4_201667886_e20.nnue.ckpt 11858755 15/10/2026 21:23:18
```

The `.nnue` next to it, `D:\NAGS\data\net_kb4_201667886_e20.nnue` (1,581,692 bytes), is byte-identical to
`nets\nags.nnue` (SHA-256 equal). Section 4's `--check` confirms the checkpoint itself.

## 3. The real quantization error of network 8 (2026-10-26)

```
$PY tools\nnue\quant_error.py --net D:\NAGS\data\net_kb4_201667886_e20.nnue.ckpt --data runs\xm\pc.bin --positions 40000 --json runs\quant_net8.json
D:\NAGS\data\net_kb4_201667886_e20.nnue.ckpt: 37195 positions; |eval| mean 400 cp, p50 214, p99 1838
total (float - engine): mean 9.40 cp, p50 7.48, p90 20.50, p99 32.21, max 47.13, bias +5.76, sign changes 530
  by output bucket (mean/p99): 0: 7.89/24.9, 1: 7.33/23.2, 2: 6.69/22.8, 3: 8.35/26.0, 4: 6.95/22.8, 5: 7.35/24.1, 6: 6.72/23.4, 7: 21.01/38.6
weights (float - rounded): mean 9.40 cp, p50 7.45, p90 20.53, p99 32.14, max 47.27, bias +5.77, sign changes 569
  by output bucket (mean/p99): 0: 8.02/25.5, 1: 7.35/23.5, 2: 6.61/22.7, 3: 8.33/26.3, 4: 6.92/22.7, 5: 7.33/23.9, 6: 6.71/23.5, 7: 21.07/38.7
arithmetic (rounded - engine): mean 0.51 cp, p50 0.51, p90 0.91, p99 1.00, max 1.02, bias -0.01, sign changes 0
  by output bucket (mean/p99): 0: 0.52/1.0, 1: 0.51/1.0, 2: 0.52/1.0, 3: 0.51/1.0, 4: 0.51/1.0, 5: 0.51/1.0, 6: 0.51/1.0, 7: 0.51/1.0
```

The 400 reference games (37,195 positions, all there are; network 8 never trained on seed 77). Time: 6 s.
Network 8 loses 9.4 cp on average to quantization, with a systematic +5.8 cp bias; nearly all of it is the
rounding of the weights (integer arithmetic adds 0.51 cp). Output bucket 7 (the most pieces: openings) is
much the worst, 21.0 cp mean against 6.7-8.4 for the others. For comparison, the VM measured 17.0 cp mean,
bias +6.9 cp and arithmetic 0.51 cp on a fine-tuned derivative of network 8.

## 4. Bias-corrected network 8 and its SPRTs (2026-10-26)

```
$PY tools\nnue\reexport.py --ckpt D:\NAGS\data\net_kb4_201667886_e20.nnue.ckpt --data runs\xm\pc.bin --positions 20000 --out nets\candidates\net8_corrected.nnue --check nets\nags.nnue
the plain export of net_kb4_201667886_e20.nnue.ckpt is nets\nags.nnue, byte for byte
wrote nets\candidates\net8_corrected.nnue: output biases corrected on 18597 positions
  plain     export on 18598 other positions: float - engine 9.39 cp mean, p50 7.47, bias +5.75, sign changes 271
  corrected export on 18598 other positions: float - engine 6.04 cp mean, p50 4.91, bias +0.00, sign changes 119
```

Time: 7 s. The checkpoint is network 8's (byte-for-byte check). The engine loads the corrected file
(`info string NNUE evaluation: nets/candidates/net8_corrected.nnue`); its evaluation of the start position
is 62 cp against network 8's 41 (the start position is in output bucket 7, whose quantization error is the
largest, 21.0 cp in section 3).

Launching: the template's `Start-Process`, run from the Claude app's shell, still leaves the match inside
the app's Windows job (checked earlier with `IsProcessInJob`: such processes report `True`), so an app
update would end it. The matches were therefore started from a `.cmd` file opened through `explorer.exe`,
which puts them outside the job (`False`).
