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
SPRT [0, 10] at 3+0.03, H1 after 1664 games. That was one checkpoint (a fine-tuned derivative), and its
[−5, 0] check at 10+0.1 was still running at handoff (+2.9 ± 6.5, LLR +2.40 after 3770 games). Network 8
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

## Reporting back

Create `docs/PC_RESULTS.md` on branch `pc/results` (from `integ/round3`) with one section per task above.
For each: the commands as run (with the real paths), the printed results (copied, not paraphrased),
anything that failed or differed from "Expected" and what was done about it, and the time taken. Push
after each section. When an SPRT finishes, also add its row in the strength-log format of
`docs/TESTING.md`, for example:
`| what | SPRT [0, 10], 3+0.03, <concurrency> at a time on the Windows PC | H1 after N games (+W =D -L) | +x ± y | bench |`.
