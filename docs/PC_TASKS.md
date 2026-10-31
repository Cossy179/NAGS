# Tasks for the training PC

A handoff from the cloud VM session (2026-10-26), for a Claude session on the Windows PC. `docs/NNUE.md`
records that PC as an AMD Ryzen 5 3500X (6 cores, 6 threads), an RTX 3060 12 GB, 16 GB RAM, MSVC (VS 2019
Build Tools) and Python 3.13 with CUDA PyTorch. Check the thread count, since the user mentioned 5; every
command below takes it from `$T` (see "Threads"). Everything here uses branch **`integ/round3`**, which combines all the
work of the last two rounds. CI is green on it on Linux, macOS and Windows; ctest 49/49 and pytest 359
passed with none skipped on the VM. These are the tasks that need this PC's GPU, its data or its
checkpoints. Work through them in order: each section says how long it takes, what to run, and what to
record.

| # | Task | Time | Needs |
|---|---|---|---|
| 0 | Setup, build, tests, bench check | 30 min | - |
| 1 | Cross-machine check of resumable self-play (MSVC vs the VM's GCC) | 5 min | 0 |
| 2 | Find network 8's training checkpoint | minutes | - |
| 3 | Network 8's real quantization error | 10 min | 2 |
| 4 | Bias-corrected network 8: re-export, then SPRTs against network 8 | hours | 2 |
| 5 | GPU training overhead (`--profile`, `--sync-every`) | 20 min | 0 |
| 6 | Warm start vs from scratch: new data, three training runs, SPRTs | 1-2 days | 0 |
| 7 | Deeper labels experiment (optional) | 1-2 days | 6a |
| 8 | Strength at 4 threads (optional) | 6 h | 0 |
| 9 | The fused feature transformer on the GPU (`--ft-impl fused`) | 30 min | 0 |
| 10 | Teacher labels against NAGS's own deeper labels, equal compute | 2-3 days | 6a |
| 11 | Quantization-aware fine-tune (arm D of section 6; optional: H0 on the VM) | 1 h + SPRTs | 6a |
| 12 | Data mixtures and targeted data | 2-3 days | 6, 7-piece books |
| 13 | v3 against v4 on identical data | 3 h + SPRTs | 6a |
| 14 | Endgame accuracy on 5-piece tablebases (optional) | 1 h | 0 |

Sections 9-14 belong to the strength programme in `docs/STRENGTH_PLAN.md` (branch `claude/upbeat-volta-95lzjz`,
which carries `integ/round3`, `pc/results` and `exp/spsa-tune`), and its decision rules are written there. Use
a checkout of that branch (`git worktree add ..\NAGS-programme origin/claude/upbeat-volta-95lzjz`); its benches
are `integ/round3`'s.

Background: `docs/NNUE.md` (training, shards, quantization, the new options), `docs/TESTING.md` (SPRT,
funnel, tuning, strength log), `docs/COMPATIBILITY.md` (what must not change). The VM's reports:
https://claude.ai/artifact/UgPVqJ7i1y3QBnH3shEUow and https://claude.ai/artifact/P6AU31rz428PeYkgdCAZgp.

## Rules for this machine (read before running anything)

1. **Do not disturb a training run that is in progress.** First check for running `python` (train.py,
   run_round.py), `nags_datagen.exe` and `nags_enhanced.exe` processes
   (`Get-Process python, nags_datagen, nags_enhanced -ErrorAction SilentlyContinue`) and ask the user before
   continuing if any exists. Never write into the data directory of a round that is still going.
2. **Never overwrite or delete** `nets\nags.nnue`, existing `selfplay_*.bin` files or their `.games`
   indexes, existing `*.ckpt` checkpoints, or `round_*.txt` logs. Every output below goes to a new file
   or directory: `nets\candidates\`, `runs\`, and the new data directory `R9` (section 6).
3. **Use a separate checkout**, so the checkout that a running or earlier round uses keeps its binaries:
   `git worktree add ..\NAGS-round3 origin/integ/round3`. Build and run everything from there.
4. **Start long runs detached from the Claude app.** On this PC a run that was a child of a packaged app
   ended when the app updated itself (`docs/NNUE.md`). Use `Start-Process` (template below), write a log,
   and check the log. Every long job here can be resumed with the same command (shards, `train.py --resume`,
   `sprt.py --resume`).
5. **Run one CPU-heavy job at a time.** `nags_datagen --threads $T` and an SPRT with
   `--concurrency $C` each fill the CPU; running both slows the matches and makes them noisy. GPU training can run alongside data
   generation, but not alongside an SPRT.
6. **Only an SPRT decides strength.** Record every number as it was printed: games, W/D/L, Elo ± error,
   LLR, the decision. Do not replace `nets\nags.nnue`, merge to `main` or push to `main`: report, and the
   user decides.
7. **Results go to `docs/PC_RESULTS.md`** on a new branch `pc/results` (in the new checkout:
   `git switch -c pc/results`). Commit and push that branch only (`git push -u origin pc/results`),
   updating it after each section, so the results survive if the session ends.
8. **Write nothing into an earlier round's data directory.** `train.py` caches each data file's SHA-256 as
   `<file>.sha256` next to it, so add `--no-data-cache` whenever it reads data from another round
   (section 5).

**Threads.** Use every thread the CPU has:
```powershell
$T = [Environment]::ProcessorCount   # logical processors: 6 on a Ryzen 5 3500X
$C = $T - 1                          # games played at once in a match
```
- Self-play (`nags_datagen`) uses `--threads $T`. Measured on this PC: 1,465 positions/s with 6 threads,
  1,301 with 5, and no gain from more threads than the CPU has.
- Matches (`sprt.py`) use `--concurrency $C`, that is $T − 1. Each game has one engine thinking at a time, so that
  fills all but one thread; the last one stays for the runner and the clocks. Using every thread there
  makes engines steal time from each other (`docs/TESTING.md`).
- Training runs on the GPU; its CPU use is small once the data is in GPU memory.
- Only two places use fewer threads, on purpose: the 4-thread match (section 8) plays one game at a time
  so the engine has its 4 threads, and `bench` is single-threaded by definition.

The double-quoted command string of the launch template below expands `$T` and `$C`; in a `.cmd` file, write the numbers themselves.

Detached launch template (PowerShell). Adjust the paths; `$PY` is the venv's python:
```powershell
$ROOT = "C:\path\to\NAGS-round3"; $PY = "C:\path\to\NAGS\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "$ROOT\runs" | Out-Null
Start-Process cmd.exe -WindowStyle Minimized -ArgumentList '/c', "cd /d $ROOT && `"$PY`" tools\sprt.py ARGS >> runs\NAME.log 2>&1"
```
Use `>>` (append) so that a resumed run (`--resume runs\NAME.log`, same arguments) adds to the same log.

## 0. Setup and checks (about 30 minutes)

```powershell
git fetch origin
git worktree add ..\NAGS-round3 origin/integ/round3
cd ..\NAGS-round3
cmake -B build
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
foreach ($e in "nags_basic","nags_fast","nags_enhanced","nags") { & ".\build\Release\$e.exe" bench | Select-String "Nodes searched" }
& $PY -m pip install -r requirements.txt -r requirements-onnx.txt   # into the existing CUDA venv; keeps its CUDA torch
& $PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& $PY -m pytest -q -rs
```
**Expected:** ctest 49/49. Benches **1417097, 1340613, 1528850, 1528850** for nags_basic, nags_fast,
nags_enhanced and nags. These are `integ/round3`'s; `main` gives 1509779 for nags_enhanced. pytest:
everything passes; the SIGINT funnel test is skipped on Windows by design. CUDA prints `True` and the GPU.

**Record:** the four benches, the ctest and pytest summary lines, the torch version and device. **If a
bench differs, stop and report it**, because section 1 and every shard depend on it.

## 1. Cross-machine check of resumable self-play (about 5 minutes)

The VM (GCC, Linux) generated these 400 games; its index is in the repo,
`docs/data/ref_seed77_n5000_g400.games`. The MSVC build should play the same games, with the same
checksum per game, at any thread count: only the order in the file depends on the threads. Generate them
with every thread, kill the run part-way, resume it, then compare game by game:
```powershell
New-Item -ItemType Directory -Force runs\xm | Out-Null
$p = Start-Process .\build\Release\nags_datagen.exe -ArgumentList "--out runs\xm\pc.bin --nodes 5000 --seed 77 --first-game 0 --games 400 --threads $T" -PassThru
Start-Sleep 15; Stop-Process -Id $p.Id -Force       # a hard kill mid-run
.\build\Release\nags_datagen.exe --out runs\xm\pc.bin --nodes 5000 --seed 77 --first-game 0 --games 400 --threads $T
& $PY tools\nnue\shards.py check runs\xm\pc.bin --checksums all
& $PY -c "import sys; sys.path.insert(0,'tools/nnue'); import shards; ref=[l.split() for l in open('docs/data/ref_seed77_n5000_g400.games').read().splitlines()[1:]]; r={int(x[0]):(x[1],x[2],x[4]) for x in ref}; h,g=shards.read_index('runs/xm/pc.bin'); p={x.number:(str(x.records),str(x.bytes),x.checksum) for x in g}; bad=sorted(k for k in r if r[k]!=p.get(k)); print('games', len(p), 'identical to the VM:', not bad, 'first differing game:', bad[:1]); print('header equal:', h==open('docs/data/ref_seed77_n5000_g400.games').readline().rstrip())"
```
**Expected:** the resume prints `N already written, 400-N to play`; `shards.py check` prints
`1 shard(s), 400 games: consistent`; the comparison prints
`games 400 identical to the VM: True first differing game: []` and `header equal: True`. The header is:
```
# nags_datagen games v2 seed=77 nodes=5000 random-plies=8 max-opening-score=400 hash=16 repetition=first format=bin network=embedded network-fnv=c7c60e3d6a52e155 search=1528850
```
**Record:** the kill point (`N`), the time, and both comparison lines. If a game differs, rerun with
`--threads 1` into another file: with one thread the file is in game order, and the VM's single-thread
file had SHA-256 `5DAE2E861C56CA106CA768811DB80B25DAE5C3B2E0654DC698B1DB2B48AE7674` (1,190,240 bytes).
Report the first differing game's index line from both. Don't go on to section 6 until it's understood:
data from this PC and from the VM must be the same games for the same seed.

## 2. Find network 8's training checkpoint (minutes)

Network 8 (`nets\nags.nnue`, 1,581,692 bytes) was trained by `run_round.py`, design `kb4`, 20 epochs, on
201.7M positions (seeds 9001–9024), so its checkpoint is a `net_kb4_*_e20.nnue.ckpt` in that round's data
directory (`data\` or, for example, `D:\NAGS\data`).
```powershell
Get-ChildItem -Recurse -Filter "net_kb4_*e20.nnue.ckpt" C:\, D:\ -ErrorAction SilentlyContinue | Select FullName, Length, LastWriteTime
```
Section 4's `--check` proves which one it is: the checkpoint's plain export must equal `nets\nags.nnue`
byte for byte. **Record:** the path, and the matching `.nnue` next to it. If none matches, say so, and
skip sections 3 and 4.

## 3. The real quantization error of network 8 (about 10 minutes)

The VM could only estimate it. Use positions network 8 did not train on, for example the reference games
from section 1:
```powershell
& $PY tools\nnue\quant_error.py --net <NET8_CKPT> --data runs\xm\ref.bin --positions 40000 --json runs\quant_net8.json
```
**Record:** the three lines (total, weights, arithmetic: mean, p50, p90, p99, max, bias, sign changes)
and the per-bucket line. For comparison, on a network-8-derived fine-tuned checkpoint the VM measured a
total of 17.0 cp mean, bias +6.9 cp, and integer arithmetic 0.51 cp.

## 4. Bias-corrected network 8 and its SPRTs (hours; the cheapest possible gain)

On the VM, the bias-corrected export of a checkpoint beat that checkpoint's plain export: **+13.8 ± 10.7**,
SPRT [0, 10] at 3+0.03, H1 after 1664 games, and it passed its non-regression check at 10+0.1: SPRT
[−5, 0], H1 after 4486 games, +3.2 ± 6.1. That was one checkpoint (a fine-tuned derivative). Network 8
itself can be corrected without training:
```powershell
New-Item -ItemType Directory -Force nets\candidates | Out-Null
& $PY tools\nnue\reexport.py --ckpt <NET8_CKPT> --data runs\xm\ref.bin --positions 20000 --out nets\candidates\net8_corrected.nnue --check nets\nags.nnue
```
**Expected:** `the plain export ... is nets\nags.nnue, byte for byte`, then the float-vs-engine error of
the plain and the corrected export on 20,000 other positions. **Record:** both lines.

Then the match. Both sides are `integ/round3`'s `nags_enhanced.exe`; the base plays its embedded network 8.
Run it detached (template above):
```
tools\sprt.py --engine build\Release\nags_enhanced.exe --engine build\Release\nags_enhanced.exe --name net8-corrected --name net8 --option1 EvalFile=nets\candidates\net8_corrected.nnue --tc 3+0.03 --sprt 0 10 --games 8000 --concurrency $C --openings tools\openings\nags_balanced.epd --resign-adjudication 4 1000 --pgnout runs\net8c_stc.pgn
```
Log: `runs\net8c_stc.log`. It starts with a `Fingerprint:` line. The preflight refuses to start if the
`EvalFile` doesn't load. Expect a few hours; 8000 games at most.

**If H1:** the [−5, 0] check at 10+0.1, same command with `--tc 10+0.1 --sprt -5 0 --games 6000`, log
`runs\net8c_ltc.log`. **If both pass,** report it: the user decides whether `net8_corrected.nnue` becomes
`nets\nags.nnue`. If so, that is a commit with the bench change and a strength-log row in `docs/TESTING.md`,
made on the user's instruction. **If H0 or undecided at the cap,** record it; nothing changes.
**Record:** every final status line, the decision, and the `Forfeits:` line if any.

## 5. GPU training overhead (about 20 minutes)

`--sync-every N` reads the loss back from the GPU every N batches instead of every batch. The trained
network is byte-identical (tested). Measure what it saves on the 3060, using the data network 8 was
trained on (`<DATA>\selfplay_*.bin`, read only):
```powershell
$D = "<DATA>\selfplay_*.bin"
& $PY tools\nnue\train.py --data $D --out runs\prof_a.nnue --buckets 8 --king-buckets 4 --activation screlu --device cuda --no-data-cache --profile 50
& $PY tools\nnue\train.py --data $D --out runs\prof_b.nnue --buckets 8 --king-buckets 4 --activation screlu --device cuda --no-data-cache --profile 50 --sync-every 100
```
`--profile` writes nothing. Its output gives time per phase (data, forward, backward, step, loss read) and
end to end with the loss read every batch versus every 100. Then one real epoch each way (writes
`runs\ep1_*.nnue`; about 6 minutes each):
```powershell
& $PY tools\nnue\train.py --data $D --out runs\ep1_sync1.nnue --buckets 8 --king-buckets 4 --activation screlu --device cuda --no-data-cache --epochs 1
& $PY tools\nnue\train.py --data $D --out runs\ep1_sync100.nnue --buckets 8 --king-buckets 4 --activation screlu --device cuda --no-data-cache --epochs 1 --sync-every 100
(Get-FileHash runs\ep1_sync1.nnue).Hash -eq (Get-FileHash runs\ep1_sync100.nnue).Hash
```
**Expected:** the last line prints `True` (same network). **Record:** both profiles, the epoch seconds
(`runs\ep1_*.nnue.json`, `history[0].seconds`), and the comparison. If `--sync-every 100` saves 3% or
more of the epoch time, report that it should become run_round's default for GPU training; the default is
still 1.

## 6. Warm start against training from scratch (about 1–2 days: the main GPU experiment)

Question: does starting from network 8's weights (`--init-from`) on new data give a network at least as
strong as training from scratch, in less GPU time?

**6a. New data (about 20 hours with 6 threads, 22 with 5; resumable).** Make a new directory `R9` on the data drive (for
example `D:\NAGS\r9`). In it, generate eight shards of 150,000 games with seeds never used for training
(9101–9108), about 104M positions, plus a fixed evaluation set from seed 9999:
```
.\build\Release\nags_datagen.exe --out R9\selfplay_9101.bin --seed 9101 --first-game 0 --games 150000 --threads $T
... the same for 9102 to 9108, one after the other ...
.\build\Release\nags_datagen.exe --out R9\evalset_9999.bin --seed 9999 --first-game 0 --games 3000 --threads $T
```
Run them as one detached `.cmd` that loops over the seeds. After a stop, run the same `.cmd` again: each
shard resumes from its index, and finished ones report `150000 already written, 0 to play`. Then:
```powershell
& $PY tools\nnue\shards.py check R9\selfplay_910*.bin --checksums all
```
**Expected:** `8 shard(s), 1200000 games: consistent`. The evaluation set must not be named
`selfplay*`, so nothing trains on it. **Record:** positions per second, total positions, the check.

**6b. Three training runs on the same data (about an hour each on the 3060).** All three use network 8's
design, the evaluation set, and the best epoch kept:
```powershell
$COMMON = "--data R9\selfplay_910*.bin --buckets 8 --king-buckets 4 --activation screlu --device cuda --eval-set R9\evalset_9999.bin --select-by eval:evalset_9999.bin --keep-best"
# A: from scratch, 20 epochs
& $PY tools\nnue\train.py $COMMON.Split(' ') --out R9\A_scratch_e20.nnue --epochs 20
# B: from network 8's weights, 20 epochs (fresh optimiser and schedule)
& $PY tools\nnue\train.py $COMMON.Split(' ') --out R9\B_warm_e20.nnue --epochs 20 --init-from nets\nags.nnue
# C: from network 8's weights, 10 epochs (half the GPU time)
& $PY tools\nnue\train.py $COMMON.Split(' ') --out R9\C_warm_e10.nnue --epochs 10 --init-from nets\nags.nnue
```
Each can be resumed with `--resume` and the same arguments. The training/validation split is by complete
games, since these shards have indexes; the log says `split by games`. **Record from each `<out>.json`:**
the best epoch and its evaluation-set loss, the loss by piece count, the seconds per epoch and in total,
and the validation loss.

**6c. Matches (hours each, one at a time).** Each network at its best epoch (`R9\X.best.nnue`) against
network 8, then the warm runs against scratch:
```
tools\sprt.py --engine build\Release\nags_enhanced.exe --engine build\Release\nags_enhanced.exe --name A --name net8 --option1 EvalFile=R9\A_scratch_e20.best.nnue --tc 3+0.03 --sprt 0 10 --games 8000 --concurrency $C --openings tools\openings\nags_balanced.epd --resign-adjudication 4 1000
(the same for B and C)
tools\sprt.py ... --name B --name A --option1 EvalFile=R9\B_warm_e20.best.nnue --option2 EvalFile=R9\A_scratch_e20.best.nnue --sprt -5 5 --games 8000 ...
tools\sprt.py ... --name C --name A --option1 EvalFile=R9\C_warm_e10.best.nnue --option2 EvalFile=R9\A_scratch_e20.best.nnue --sprt -5 5 --games 8000 ...
```
**Acceptance:** warm starts become the recommended way to train only if B or C is not worse than A
(C vs A H1 or undecided in [−5, 5], with C at half the GPU time), and the network that beats network 8
does so by SPRT. **Record:** all final status lines and decisions, and GPU minutes per run. A network that
beats network 8 is a candidate for `nets\nags.nnue`, which is the user's decision (rule 6).

## 7. Deeper labels for informative positions (optional, after 6; about 1–2 days of CPU)

The VM measured (100 games): positions where the static eval and the 5000-node score differ by 200 cp or
more (`disagree:200`) are 16.8% of positions. Relabelling them with 40,000 nodes changes their labels by
210.7 cp on average. A random 15.6% (`random:0.168`, equal cost) changes by 66.6 cp. Whether those labels
train a better network is untested. Use the same seeds as three of the section 6 shards, so the games are
identical and only the labels differ:
```
.\build\Release\nags_datagen.exe --out R9\relabel_dis\selfplay_9101.bin --seed 9101 --first-game 0 --games 150000 --threads $T --relabel-nodes 40000 --relabel-select disagree:200
.\build\Release\nags_datagen.exe --out R9\relabel_rnd\selfplay_9101.bin --seed 9101 --first-game 0 --games 150000 --threads $T --relabel-nodes 40000 --relabel-select random:<FRACTION>
(seeds 9101–9103 for both)
```
Take `<FRACTION>` from the end line of the `disagree` run (`relabelled X of Y positions (Z%)`), so that
the two arms cost the same. Then train three networks identically from scratch (design and options as A,
`--epochs 20`) on: the plain shards 9101–9103 from section 6, the `disagree` ones and the `random` ones.
Compare their evaluation-set losses, then SPRT `disagree` against `random` ([0, 10], 3+0.03).
**Record:** the relabel fractions and mean changes, the time per shard, the losses, and the SPRT.

## 8. Multithreaded strength at 4 threads (optional, about 6 hours)

The VM measured that Threads=2 against 1 gives +107.5 ± 14.0, and that double the time gives +165.4 ± 14.9.
So a second thread is worth about 65% of a time doubling. The 6-core PC can measure 4 threads:
```
tools\sprt.py --engine build\Release\nags_enhanced.exe --engine build\Release\nags_enhanced.exe --name threads4 --name threads1 --option1 Threads=4 --tc 3+0.03 --concurrency 1 --games 1000 --openings tools\openings\nags_balanced.epd --resign-adjudication 4 1000
tools\sprt.py ... --name time4x --name time1x --tc1 12+0.12 --tc2 3+0.03 --concurrency $C --games 1000 ...
```
**Record:** both results, and the nodes per second at 1 and 4 threads (`go movetime 4000` from a few
positions).

## 9. The fused feature transformer on the GPU (about 30 minutes)

`--ft-impl fused` computes the same network as the default path with about a third of the gathered rows
(docs/STRENGTH_PLAN.md, item 9). Profile both on the data of section 5 (read only):
```powershell
$D = "<DATA>\selfplay_*.bin"
foreach ($impl in "embedding","fused") { & $PY tools\nnue\train.py --data $D --out runs\prof_$impl.nnue --buckets 8 --king-buckets 4 --activation screlu --device cuda --no-data-cache --profile 50 --ft-impl $impl }
& $PY tools\nnue\train.py --data $D --out runs\prof_qat.nnue --buckets 8 --king-buckets 4 --activation screlu --device cuda --no-data-cache --profile 50 --qat
& $PY tools\nnue\train.py --data $D --out runs\ep1_fused.nnue --buckets 8 --king-buckets 4 --activation screlu --device cuda --no-data-cache --epochs 1 --ft-impl fused
```
**Record:** the three profile lines (ms per phase, positions/s), and the one-epoch seconds and validation loss
of `runs\ep1_fused.nnue.json` next to section 5's `runs\ep1_sync1.nnue.json` (389-393 s, val 0.008673).
**Decision (pre-registered):** at least 1.2x per epoch and a validation loss within 0.5% of 0.008673 make
`fused` the default for later runs (a code change, made on the user's instruction).

## 10. Teacher labels against NAGS's own deeper labels (2-3 days of CPU; after 6a)

Question: do a stronger engine's scores train a better network than NAGS's own deeper search at the same
engine time? Use the official Stockfish build the calibrations used (`$SF`); every engine process has one
thread.

**10a. Equal time and the scale (about 30 minutes).** Label the same 20 games both ways, then compare:
```powershell
& $PY tools\nnue\rescore.py label --data R9\selfplay_9101.bin --engine $SF --nodes 100000 --max-games 20 --workers $C --out-dir runs\t_sf
& $PY tools\nnue\rescore.py label --data R9\selfplay_9101.bin --engine build\Release\nags_enhanced.exe --nodes 150000 --max-games 20 --workers $C --out-dir runs\t_nags
& $PY tools\nnue\rescore.py compare --base R9\selfplay_9101.bin --labels runs\t_sf\selfplay_9101.bin runs\t_nags\selfplay_9101.bin --names base sf nags
```
`engine_seconds_per_position` in each `.teacher.json` gives the cost: change NAGS's `--nodes` (into a new
`--out-dir`; an output made with other settings is refused) until the two agree within 10%. On the VM, with 4
processes at once, Stockfish 16 took 374 ms a position at 200k nodes and NAGS 110 ms at 200k (3.4 times as
fast per node), so start NAGS at about 3.4 times Stockfish's nodes. The `to base`
column of `sf` is the factor that puts Stockfish's centipawns on NAGS's scale (about 1.6 on the VM).
Record both node counts, both times and the compare table.

**10b. Labels** for 10% of the games of shards 9101-9108, about 10M positions (one shard at a time; the same
command resumes):
```powershell
foreach ($s in 9101..9108) { & $PY tools\nnue\rescore.py label --data R9\selfplay_$s.bin --engine $SF --nodes <SF_NODES> --scale <FACTOR> --fraction 0.1 --workers $C --out-dir R9\teacher_sf }
foreach ($s in 9101..9108) { & $PY tools\nnue\rescore.py label --data R9\selfplay_$s.bin --engine build\Release\nags_enhanced.exe --nodes <NAGS_NODES> --fraction 0.1 --workers $C --out-dir R9\teacher_nags }
```
The same `--fraction` and `--select-seed` (default 1) choose the same games for both.

**10c. Two fine-tunes** of network 8 with identical settings:
```powershell
$FT = "--buckets 8 --king-buckets 4 --activation screlu --device cuda --eval-set R9\evalset_9999.bin --select-by eval:evalset_9999.bin --keep-best --init-from nets\nags.nnue --lr 1e-4 --epochs 10"
& $PY tools\nnue\train.py --data "R9\teacher_sf\selfplay_*.bin" $FT.Split(' ') --out R9\T_sf.nnue
& $PY tools\nnue\train.py --data "R9\teacher_nags\selfplay_*.bin" $FT.Split(' ') --out R9\T_nags.nnue
```
(The evaluation set's labels are NAGS's 5000-node ones, so its loss favours T_nags; it picks each run's best
epoch, the match decides.)

**10d. SPRT** [0, 10] at 3+0.03, `T_sf.best.nnue` against `T_nags.best.nnue`; if H1, `T_sf.best.nnue` against
network 8. **Decision (pre-registered):** teacher labels are worth expanding only if T_sf beats T_nags.
**Record:** the label runs' times, the eval-set losses, every SPRT line.

## 11. Quantization-aware fine-tune (optional; about 1 hour of GPU, then SPRTs; after 6a)

On the VM a QAT fine-tune lost its SPRT against the same fine-tune without QAT (H0, −10.9 ± 14.3, 922 games;
`docs/STRENGTH_PLAN.md`, item 2), so run this only alongside section 6 when the GPU is free: it tests QAT at
scale, where the outcome could differ.

Arm D of section 6: B's command with `--qat`, everything else the same:
```powershell
& $PY tools\nnue\train.py $COMMON.Split(' ') --out R9\D_warm_qat_e20.nnue --epochs 20 --init-from nets\nags.nnue --qat
foreach ($x in "B_warm_e20","D_warm_qat_e20") {
  & $PY tools\nnue\quant_error.py --net R9\$x.best.nnue --data R9\evalset_9999.bin --positions 100000
  & $PY tools\nnue\quant_error.py --net R9\$x.nnue.ckpt --data R9\evalset_9999.bin --positions 100000
}
```
The first line of each pair is the kept network as the engine runs it: its `loss against the labels`, engine
column, is what compares B and D. The second is the last epoch's float-against-export error (for D close to the
integer arithmetic alone, by design). Then SPRT [0, 10] at 3+0.03, D against B; if H1,
[−5, 0] at 10+0.1. **Decision:** as `docs/STRENGTH_PLAN.md` item 2. **Record:** both quant_error reports, the
epoch times (QAT costs about 5% on the CPU), every SPRT line.

## 12. Data mixtures and targeted data (2-3 days; after 6)

Targeted sources first (each a few hours of CPU):
```powershell
& $PY tools\nnue\mix.py report "R9\selfplay_910*.bin" --json runs\mix_r9.json
& $PY tools\nnue\mix.py report "<DATA>\selfplay_*.bin" --json runs\mix_r8.json
& $PY tools\nnue\mix.py book "R9\selfplay_910*.bin" --select imbalanced --count 5000 --max-score 300 --out books\imbalanced.epd
& $PY tools\make_openings.py --engine build\Release\nags_enhanced.exe --pgn AJ-CORR-PGN-000.pgn --count 5000 --min-plies 8 --max-plies 16 --out books\corr.epd
.\build\Release\nags_datagen.exe --out R9T\imb_9201.bin --seed 9201 --first-game 0 --games 60000 --threads $T --book books\imbalanced.epd --random-plies 2 --max-opening-score 600
.\build\Release\nags_datagen.exe --out R9T\corr_9202.bin --seed 9202 --first-game 0 --games 60000 --threads $T --book books\corr.epd --random-plies 2
& $PY tools\nnue\tb_data.py generate --syzygy <TB345> --material KRvKB KRvKN KBvKP KNvKP KRPvKR KBPvKB KNPvKB KRPvKB KPvKP KRvKBP --count 100000 --engine build\Release\nags_enhanced.exe --nodes 5000 --workers $T --out R9T\tb_1.bin
```
(`AJ-CORR-PGN-000.pgn` needs `git lfs pull`.) Then the mixtures, about 100M positions each, and the same
training for each (network 8's design, from scratch, 20 epochs, `--eval-set R9\evalset_9999.bin --keep-best`):
```powershell
& $PY tools\nnue\mix.py build --source "R9\selfplay_910*.bin=1" --positions 100000000 --max-copies 4 --out MIX\m0.bin
& $PY tools\nnue\mix.py build --source "R9\selfplay_910*.bin=0.5" --source "<DATA>\selfplay_*.bin=0.5" --positions 100000000 --max-copies 4 --out MIX\m1.bin
& $PY tools\nnue\mix.py build --source "R9\selfplay_910*.bin=0.45" --source "<DATA>\selfplay_*.bin=0.45" --source "R9T\imb_*.bin=0.04" --source "R9T\corr_*.bin=0.04" --source "R9T\tb_*.bin=0.02" --positions 100000000 --max-copies 4 --out MIX\m2.bin
```
SPRTs [0, 10] at 3+0.03: m1 against m0, then m2 against the better. **Record:** each `.mix.json`, the
eval-set losses by piece count, `tb_data.py measure` (section 14) for each network, and the SPRT lines.

## 13. v3 against v4 on identical data (about 3 hours of GPU, then SPRTs; after 6a)

```powershell
& $PY tools\nnue\train.py --data "R9\selfplay_910*.bin" --buckets 8 --king-buckets 4 --activation screlu --device cuda --epochs 20 --out R9\v3_e20.nnue
& $PY tools\nnue\train_v4.py --data "R9\selfplay_910*.bin" --buckets 8 --king-buckets 4 --activation pairwise --device cuda --epochs 20 --out R9\v4_e20.nnue
.\build\Release\nags_enhanced.exe   # then: setoption name EvalFile value R9\v4_e20.nnue, speedtest, bench
```
(Section 6's A is the same v3 run with an eval set: use `A_scratch_e20.nnue` instead of training v3 again.)
SPRT [0, 10] v4 against v3 at 20,000 nodes a move (`--nodes 20000`), then at 3+0.03; if both pass, [−5, 0]
at 10+0.1. **Record:** both validation losses, the `speedtest` evaluate and update lines and `go nodes
1500000` nodes/s with each network, every SPRT line.

## 14. Endgame accuracy on 5-piece tablebases (optional, about 1 hour)

With the 3-4-5 Syzygy tables (`<TB345>`, about 1 GB; WDL files suffice):
```powershell
& $PY tools\nnue\tb_data.py measure --syzygy <TB345> --material KRvKB KRvKN KNNvK KBvKP KNvKP KRPvKR KBPvKB KNPvKB KRPvKB KRvKBP KQvKRP KPvKPP --count 400 --engine build\Release\nags_enhanced.exe --nodes 10000 --workers $C --json runs\tb_net8.json
```
**Record:** the per-material lines. On the VM, network 8 says 75.0% of 3-4 piece results right, with KRvKB at 8%,
KRvKN 18%, KNNvK 1% (`docs/STRENGTH_PLAN.md`, item 7).

## Reporting back

Create `docs/PC_RESULTS.md` on branch `pc/results` (from `integ/round3`) with one section per task above.
For each: the commands as run (with the real paths), the printed results (copied, not paraphrased),
anything that failed or differed from "Expected" and what was done about it, and the time taken. Push
after each section. When an SPRT finishes, also add its row in the strength-log format of
`docs/TESTING.md`, for example:
`| what | SPRT [0, 10], 3+0.03, <concurrency> at a time on the Windows PC | H1 after N games (+W =D -L) | +x ± y | bench |`.
