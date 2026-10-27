# Tasks for the training PC

A handoff from the cloud VM sessions (2026-10-26), for a Claude session on the Windows PC. `docs/NNUE.md`
records that PC as an AMD Ryzen 5 3500X (6 cores, 6 threads), an RTX 3060 12 GB, 16 GB RAM, MSVC (VS 2019
Build Tools) and Python 3.13 with CUDA PyTorch. Check the thread count, since the user mentioned 5; every
command below takes it from `$T`. Everything here uses branch **`integ/round3`**, which combines all the
work of the last rounds. On the VM ctest passed 49/49 and pytest 373 passed with none skipped. These are
the tasks that need this PC's GPU, its data or its checkpoints. Work through them in order: each section
says how long it takes, what to run, and what to record.

**Status (2026-10-27).** Sections 0-5 are done, on `integ/round3` `7464810` in the worktree
`C:\Users\Alex\Documents\code\NAGS-round3` (branch `pc/results`; results in `docs/PC_RESULTS.md` there).
Next is section 6, after updating that worktree to this version (**"Before section 6"** below). Sections
1-5 need not be run again.

The times are machine time with nothing else running, from rates measured on this PC (`docs/NNUE.md`):
self-play 1,465 positions/s with 6 threads (network 7; network 8, which `integ/round3` plays with, has not
been timed here), training `kb4` 365 s an epoch on 201.7M positions, matches at 3+0.03 with 5 at a time
37-42 games a minute (42 in section 4: 1,770 games in 42 minutes). A match is quoted at its game cap: an SPRT can stop sooner, but close networks
often run to the cap. Steps that wait for a person or a session to start them add calendar time.

| # | Task | Machine time | Needs | Status |
|---|---|---|---|---|
| 0 | Setup, build, tests, bench check | 30-60 min | - | done: as expected |
| 1 | Cross-machine check of resumable self-play (MSVC vs the VM's GCC) | 5 min | 0 | done: identical to the VM, game for game |
| 2 | Find network 8's training checkpoint | minutes | - | done: `D:\NAGS\data\net_kb4_201667886_e20.nnue.ckpt` |
| 3 | Network 8's real quantization error | 10 min | 1, 2 | done: 9.4 cp mean, bias +5.8 |
| 4 | Bias-corrected network 8: re-export, then SPRTs against network 8 | up to 4 h, then up to 10 h | 1, 2 | done: H0, −2.9 ± 10.1, not adopted |
| 5 | GPU training overhead (`--profile`, `--sync-every`), three times each | 1.5 h | 0 | done: 0.9%, the default stays 1 |
| 6 | Warm start vs from scratch: new data, three training runs, five SPRTs | about 2 days (up to 47 h) | 0 | **next** |
| 7 | Deeper labels at equal compute (optional) | about 2 days | 6a | |
| 8 | Strength at 4 threads (optional) | 5 h | 0 | |

Background: `docs/NNUE.md` (training, shards, quantization, the options used here), `docs/TESTING.md`
(SPRT, funnel, tuning, strength log), `docs/COMPATIBILITY.md` (what must not change). The VM's reports:
https://claude.ai/artifact/UgPVqJ7i1y3QBnH3shEUow and https://claude.ai/artifact/P6AU31rz428PeYkgdCAZgp.

## Rules for this machine (read before running anything)

1. **Do not disturb a training run that is in progress.** First check for running `python` (train.py,
   run_round.py), `nags_datagen.exe` and `nags_enhanced.exe` processes
   (`Get-Process python, nags_datagen, nags_enhanced -ErrorAction SilentlyContinue`) and ask the user before
   continuing if any exists. Never write into the data directory of a round that is still going.
2. **Never overwrite or delete** `nets\nags.nnue`, existing `selfplay_*.bin` files or their `.games`
   indexes, existing `*.ckpt` checkpoints, or `round_*.txt` logs. Every output below goes to a new file
   or directory: `nets\candidates\`, `runs\`, and the new data directory `$R9` (section 6). The tools
   enforce this: `train.py` refuses an existing `--out` (or its `.ckpt`, `.json`, `.best.nnue`) unless
   `--resume` continues that run; `sprt.py` refuses a `--pgnout` that holds games unless `--resume`
   continues that match; `reexport.py --out` and `quant_error.py --json` refuse existing files;
   `nags_datagen` continues a shard only under identical settings; `Start-Detached` refuses a job name
   that exists. If a tool refuses, do not delete its earlier output: choose another name, or resume.
3. **Use a separate checkout**, so the checkout that a running or earlier round uses keeps its binaries:
   `git worktree add ..\NAGS-round3 origin/integ/round3`. Build and run everything from there.
4. **Start long runs detached from the Claude app**, with `Start-Detached` or `Start-Match` (below). On
   this PC a run that was a child of a packaged app ended when the app updated itself (`docs/NNUE.md`), and
   a process started from the app's shell with `Start-Process` stays inside the app's Windows job object
   (`IsProcessInJob` reports `True`), so `Resume-Detached` opens the `.cmd` through `explorer.exe`, which
   starts it outside (`False`; `docs/PC_RESULTS.md`, section 4).
   Each job is a `runs\NAME.cmd` whose output goes to `runs\NAME.log`; after any stop,
   `Resume-Detached NAME` runs the same `.cmd` again and every command in it continues where it stopped
   (shards from their indexes, `train.py --resume`, `sprt.py --resume`).
5. **Run one CPU-heavy job at a time.** `nags_datagen --threads $T` and an SPRT with
   `--concurrency $C` each fill the CPU; running both slows the matches and makes them noisy. GPU training
   can run alongside data generation, but not alongside an SPRT, and not during a timing measurement
   (section 5). The user's own rule here is stricter, and it applies: GPU training overlaps neither data
   generation nor a match.
6. **Only an SPRT decides strength, and only as declared before its first game.** Record every number as
   it was printed: games, W/D/L, Elo ± error, LLR, the decision. A match that reaches its cap without a
   decision is inconclusive, never a pass. Do not replace `nets\nags.nnue`, merge to `main` or push to
   `main`: report, and the user decides.
7. **Results go to `docs/PC_RESULTS.md`** on a new branch `pc/results` (in the new checkout:
   `git switch -c pc/results`). Commit and push that branch only (`git push -u origin pc/results`),
   updating it after each section, so the results survive if the session ends. `runs\` and
   `nets\candidates\` are not committed (`.gitignore`).
8. **Write nothing into an earlier round's data directory.** `train.py` caches each data file's SHA-256 as
   `<file>.sha256` next to it, so add `--no-data-cache` whenever it reads data from another round
   (section 5).

## Paths, threads and detached jobs: `runs\pc_env.ps1`

Every block below starts by loading `runs\pc_env.ps1` (section 0 creates it from
`docs\pc_env.example.ps1`):

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
```

It defines, with this machine's real paths (it stops if `$ROOT` or `$PY` does not exist):

- `$ROOT` the worktree, `$PY` the CUDA venv's `python.exe`, `$DATA8` network 8's round directory (read
  only), `$R9` the new directory of section 6; `$ENG` and `$DG` the worktree's `nags_enhanced.exe` and
  `nags_datagen.exe`, `$NET8` its `nets\nags.nnue`, `$SHARDS` the eight shard paths of section 6, each
  named (PowerShell passes a wildcard to a program unexpanded, and `shards.py` does not expand one).
- `$T`, every logical processor, for self-play (`nags_datagen --threads $T`; measured here: 1,465
  positions/s with 6 threads, 1,301 with 5, and no gain from more threads than the CPU has), and
  `$C = $T - 1` for matches (`sprt.py --concurrency $C`): each game has one engine thinking at a time, so
  that fills all but one thread, which stays for the runner and the clocks. Only two places use fewer, on
  purpose: the 4-thread match (section 8) plays one game at a time so the engine has its 4 threads, and
  `bench` is single-threaded by definition.
- `ConvertTo-CmdLine ARRAY`: one command line for a `.cmd` file, every argument quoted (paths with spaces
  stay whole). Commands are built as PowerShell arrays, never by splitting a string.
- `Start-Detached NAME LINES`: writes `runs\NAME.cmd` (the lines, run one after the other in the
  worktree, output appended to `runs\NAME.log`) and starts it outside the Claude app (rule 4). An
  existing NAME is refused. `Resume-Detached NAME` starts the same `.cmd` again.
- `Start-Match NAME ARGS`: `Start-Detached` for `tools\sprt.py ARGS --pgnout runs\NAME.pgn --resume
  runs\NAME.log`. Started again, the match continues from its log; a finished SPRT reports its decision
  without playing on.

Check a job with `Get-Content runs\NAME.log -Tail 5`.

## 0. Setup and checks (30-60 minutes)

```powershell
$env:GIT_LFS_SKIP_SMUDGE = 1   # no task needs the 1 GB LFS file AJ-CORR-PGN-000.pgn
git fetch origin
git worktree add ..\NAGS-round3 origin/integ/round3
cd ..\NAGS-round3
New-Item -ItemType Directory runs | Out-Null
Copy-Item docs\pc_env.example.ps1 runs\pc_env.ps1
```
Edit the four `EDIT` lines of `runs\pc_env.ps1`: `$ROOT` is this directory (`(Get-Location).Path`);
`$PY` the existing CUDA venv's `python.exe` (`Get-ChildItem C:\, D:\ -Recurse -Filter python.exe
-ErrorAction SilentlyContinue | Where-Object FullName -like '*\Scripts\python.exe'` lists candidates; the
right one prints `True` below); `$DATA8` network 8's round directory (where its `selfplay_*.bin` and
`round_*.txt` are; section 2 confirms it); `$R9` a directory that does not exist yet on the data drive, for
example `D:\NAGS\r9`. Then:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
cmake -B build
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
foreach ($e in "nags_basic","nags_fast","nags_enhanced","nags") { & ".\build\Release\$e.exe" bench | Select-String "Nodes searched" }
& $PY -m pip install -r requirements.txt -r requirements-onnx.txt   # into the existing CUDA venv; keeps its CUDA torch
& $PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& $PY -m pytest -q -rs
```
**Expected:** ctest 49/49. Benches **1479785, 1372714, 1584695, 1584695** for nags_basic, nags_fast,
nags_enhanced and nags. These are `integ/round3`'s; its bench gained two endgames, so other branches'
numbers are not comparable with them (before that and the stalemate change, nags_enhanced gave 1528850). pytest:
everything passes; the SIGINT funnel test is skipped on Windows by design. CUDA prints `True` and the GPU.

**Record:** the four benches, the ctest and pytest summary lines, the torch version and device, and the
four paths of `runs\pc_env.ps1`. **If a bench differs, stop and report it**, because section 1 and every
shard depend on it.

## 1. Cross-machine check of resumable self-play (about 5 minutes)

The VM (GCC, Linux) generated these 400 games; its index is in the repo,
`docs/data/ref_seed77_n5000_g400.games`. The MSVC build should play the same games, with the same
checksum per game, at any thread count: only the order in the file depends on the threads. Generate them
with every thread into `runs\xm\pc.bin` (which must not exist yet), kill the run part-way, resume it,
then compare game by game:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory runs\xm | Out-Null
$p = Start-Process -FilePath $DG -ArgumentList "--out runs\xm\pc.bin --nodes 5000 --seed 77 --first-game 0 --games 400 --threads $T" -PassThru
Start-Sleep 15; Stop-Process -Id $p.Id -Force       # a hard kill mid-run
& $DG --out runs\xm\pc.bin --nodes 5000 --seed 77 --first-game 0 --games 400 --threads $T
& $PY tools\nnue\shards.py check runs\xm\pc.bin --checksums all
& $PY -c "import sys; sys.path.insert(0,'tools/nnue'); import shards; ref=[l.split() for l in open('docs/data/ref_seed77_n5000_g400.games').read().splitlines()[1:]]; r={int(x[0]):(x[1],x[2],x[4]) for x in ref}; h,g=shards.read_index('runs/xm/pc.bin'); p={x.number:(str(x.records),str(x.bytes),x.checksum) for x in g}; bad=sorted(k for k in r if r[k]!=p.get(k)); print('games', len(p), 'identical to the VM:', not bad, 'first differing game:', bad[:1]); print('header equal:', h==open('docs/data/ref_seed77_n5000_g400.games').readline().rstrip())"
```
**Expected:** the resume prints `N already written, 400-N to play`; `shards.py check` prints
`1 shard(s), 400 games: consistent`; the comparison prints
`games 400 identical to the VM: True first differing game: []` and `header equal: True`. The header is:
```
# nags_datagen games v2 seed=77 nodes=5000 random-plies=8 max-opening-score=400 hash=16 repetition=first format=bin network=embedded network-fnv=c7c60e3d6a52e155 search=1584695
```
**Record:** the kill point (`N`), the time, and both comparison lines. If a game differs, rerun with
`--threads 1` into another file: with one thread the file is in game order, and the VM's single-thread
file had SHA-256 `972F3720689523491C5A3A5C2FADFD3A6D9C4498E1B5E7E78722559BA21A2DD0` (1,188,032 bytes). Report the first differing game's index line from
both. Don't go on to section 6 until it's understood: data from this PC and from the VM must be the same
games for the same seed.

`runs\xm\pc.bin` is also the reference data of sections 3 and 4: 400 games, **37,126 positions** that
network 8 did not train on (its data came from seeds 9001-9024).

## 2. Find network 8's training checkpoint (minutes)

Network 8 (`nets\nags.nnue`, 1,581,692 bytes) was trained by `run_round.py`, design `kb4`, 20 epochs, on
201.7M positions (seeds 9001–9024), so its checkpoint is a `net_kb4_*_e20.nnue.ckpt` in that round's data
directory (`data\` or, for example, `D:\NAGS\data`).
```powershell
Get-ChildItem -Recurse -Filter "net_kb4_*e20.nnue.ckpt" C:\, D:\ -ErrorAction SilentlyContinue | Select FullName, Length, LastWriteTime
```
Section 4's `--check` proves which one it is: the checkpoint's plain export must equal `nets\nags.nnue`
byte for byte. **Record:** the path, and the matching `.nnue` next to it; check that `$DATA8` in
`runs\pc_env.ps1` is that directory. If none matches, say so, and skip sections 3 and 4.

## 3. The real quantization error of network 8 (about 10 minutes)

The VM could only estimate it. Use all of `runs\xm\pc.bin` (section 1):
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$NET8_CKPT = 'D:\NAGS\data\net_kb4_..._e20.nnue.ckpt'   # the checkpoint section 2 found
& $PY tools\nnue\quant_error.py --net $NET8_CKPT --data runs\xm\pc.bin --positions 37126 --json runs\quant_net8.json
```
**Expected:** `37126 positions` in the first line (asking for more prints a note and uses all of them).
**Record:** the three lines (total, weights, arithmetic: mean, p50, p90, p99, max, bias, sign changes)
and the per-bucket line. For comparison, on a network-8-derived fine-tuned checkpoint the VM measured a
total of 17.0 cp mean, bias +6.9 cp, and integer arithmetic 0.51 cp.

## 4. Bias-corrected network 8 and its SPRTs (up to 4 h, then up to 10 h)

On the VM, the bias-corrected export of a checkpoint beat that checkpoint's plain export: **+13.8 ± 10.7**,
SPRT [0, 10] at 3+0.03, H1 after 1664 games, and it passed its non-regression check at 10+0.1: SPRT
[−5, 0], H1 after 4486 games, +3.2 ± 6.1. That was one checkpoint (a fine-tuned derivative). Network 8
itself can be corrected without training. `reexport.py` draws 2 × `--positions` positions at random,
corrects the output biases on one half and measures the result on the other; `runs\xm\pc.bin` holds
37,126, so 18,000 each (both halves come from the same 400 games):
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$NET8_CKPT = 'D:\NAGS\data\net_kb4_..._e20.nnue.ckpt'   # as in section 3
New-Item -ItemType Directory -Force nets\candidates | Out-Null
& $PY tools\nnue\reexport.py --ckpt $NET8_CKPT --data runs\xm\pc.bin --positions 18000 --out nets\candidates\net8_corrected.nnue --check $NET8
```
**Expected:** `the plain export ... is ...nags.nnue, byte for byte`, `corrected on 18000 positions`, then
the float-vs-engine error of the plain and the corrected export on 18,000 other positions. **Record:**
those lines.

Then the match, declared here: SPRT [0, 10] at 3+0.03, at most 8,000 games (up to about 4 hours). Both
sides are `integ/round3`'s `nags_enhanced.exe`; the base plays its embedded network 8:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$CORR = Join-Path $ROOT 'nets\candidates\net8_corrected.nnue'
$M = @('--engine', $ENG, '--engine', $ENG, '--name', 'net8-corrected', '--name', 'net8', '--option1', "EvalFile=$CORR",
       '--concurrency', $C, '--openings', $OPENINGS, '--resign-adjudication', 4, 1000)
Start-Match net8c_stc ($M + @('--tc', '3+0.03', '--sprt', 0, 10, '--games', 8000))
```
The log starts with a `Fingerprint:` line. The preflight refuses to start if the `EvalFile` doesn't load.

**If H1:** the non-regression check at 10+0.1, SPRT [−5, 0], at most 6,000 games (up to about 10 hours):
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$CORR = Join-Path $ROOT 'nets\candidates\net8_corrected.nnue'
$M = @('--engine', $ENG, '--engine', $ENG, '--name', 'net8-corrected', '--name', 'net8', '--option1', "EvalFile=$CORR",
       '--concurrency', $C, '--openings', $OPENINGS, '--resign-adjudication', 4, 1000)
Start-Match net8c_ltc ($M + @('--tc', '10+0.1', '--sprt', -5, 0, '--games', 6000))
```
**If both pass,** report it: the user decides whether `net8_corrected.nnue` becomes `nets\nags.nnue`. If so, that is a
commit with the bench change and a strength-log row in `docs/TESTING.md`, made on the user's
instruction. **If H0 or no decision at the cap,** record it; nothing changes. **Record:** every final
status line, the decision, and the `Forfeits:` line if any.

## 5. GPU training overhead, measured three times (about 1.5 hours)

`--sync-every N` reads the loss back from the GPU every N batches instead of every batch. The trained
network is byte-identical (tested). Measure what it saves on the 3060, on the data network 8 was trained
on (read only). A single timing on a GPU varies with clocks and temperature, so every measurement is
made three times, alternating the two settings, and nothing else runs meanwhile (rule 5):
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
if (-not (Test-Path -LiteralPath $DATA8)) { throw "set `$DATA8 in runs\pc_env.ps1" }
$D8 = @(Get-ChildItem -LiteralPath $DATA8 -Filter 'selfplay*.bin' | Sort-Object Name | ForEach-Object FullName)
"$($D8.Count) files: $($D8[0]) ... $($D8[-1])"
$TRAIN8 = @($PY, 'tools\nnue\train.py', '--data') + $D8 + @('--buckets', 8, '--king-buckets', 4, '--activation', 'screlu', '--device', 'cuda', '--no-data-cache')
$lines = @(foreach ($r in 1..3) { ConvertTo-CmdLine ($TRAIN8 + @('--out', 'runs\prof.nnue', '--profile', 200)) })
foreach ($r in 1..3) {
    $lines += ConvertTo-CmdLine ($TRAIN8 + @('--out', "runs\ep1_sync1_r$r.nnue", '--epochs', 1))
    $lines += ConvertTo-CmdLine ($TRAIN8 + @('--out', "runs\ep1_sync100_r$r.nnue", '--epochs', 1, '--sync-every', 100))
}
Start-Detached gpu_timing $lines
```
`$D8` must be the 201.7M positions network 8 was trained on (each run prints `... positions loaded`).
`--profile` writes nothing; each run prints the median time per phase (data, forward, backward, step,
loss read) over 200 batches, and end to end with the loss read every batch against every 100. Each
epoch run takes about 6 minutes plus 2-3 minutes of hashing and loading. When the job has finished:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
Select-String -Path runs\gpu_timing.log -Pattern '^profile:'
foreach ($r in 1..3) { foreach ($s in 'sync1', 'sync100') { "$s r$r " + (Get-Content -Raw "runs\ep1_${s}_r$r.nnue.json" | ConvertFrom-Json).history[0].seconds } }
(Get-ChildItem runs\ep1_*_r?.nnue | Get-FileHash | Select-Object -ExpandProperty Hash -Unique).Count
```
**Expected:** the last line prints `1` (all six networks the same). **Record:** the three profiles, the six
epoch times, and the hash count. **Decision:** if `--sync-every 100` is at least 3% faster than
`--sync-every 1` in each of the three pairs, report that it should become run_round's default for GPU
training; otherwise (smaller, or not in every pair) the default stays 1. A difference smaller than the
spread of one setting's three times is no difference.

## Before section 6: update the worktree of sections 0-5 (about 30 minutes)

The worktree `C:\Users\Alex\Documents\code\NAGS-round3` is on `pc/results`: `integ/round3` `7464810` and
the results commits. Merge this version of `integ/round3` into it (`docs/PC_RESULTS.md` stays; nothing
else was changed there), make `runs\pc_env.ps1`, rebuild and check:
```powershell
cd C:\Users\Alex\Documents\code\NAGS-round3
git status --short                    # only runs\ and nets\candidates\ may be listed (both ignored after the merge)
$env:GIT_LFS_SKIP_SMUDGE = 1
git fetch origin
git merge origin/integ/round3
Copy-Item docs\pc_env.example.ps1 runs\pc_env.ps1   # then edit its four EDIT lines, as in section 0
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
foreach ($e in "nags_basic","nags_fast","nags_enhanced","nags") { & ".\build\Release\$e.exe" bench | Select-String "Nodes searched" }
& $PY -m pytest -q -rs
```
**Expected:** ctest 49/49; the benches of section 0 (they changed since sections 0-5: the stalemate test
and two bench positions); pytest all passed but the SIGINT funnel test. The search changed, so check
the self-play games once more against the regenerated reference, in a new directory:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory runs\xm2 | Out-Null
& $DG --out runs\xm2\pc.bin --nodes 5000 --seed 77 --first-game 0 --games 400 --threads $T
& $PY -c "import sys; sys.path.insert(0,'tools/nnue'); import shards; ref=[l.split() for l in open('docs/data/ref_seed77_n5000_g400.games').read().splitlines()[1:]]; r={int(x[0]):(x[1],x[2],x[4]) for x in ref}; h,g=shards.read_index('runs/xm2/pc.bin'); p={x.number:(str(x.records),str(x.bytes),x.checksum) for x in g}; bad=sorted(k for k in r if r[k]!=p.get(k)); print('games', len(p), 'identical to the VM:', not bad, 'first differing game:', bad[:1]); print('header equal:', h==open('docs/data/ref_seed77_n5000_g400.games').readline().rstrip())"
```
**Expected:** `games 400 identical to the VM: True` and `header equal: True` (the header now ends
`search=1584695`). Then try the launcher once, since it was written on the VM and never run on Windows:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
Start-Detached smoke (ConvertTo-CmdLine @($PY, '-c', "import ctypes; k = ctypes.windll.kernel32; r = ctypes.c_int(); k.IsProcessInJob(k.GetCurrentProcess(), None, ctypes.byref(r)); print('smoke: in a job:', bool(r.value))"))
Start-Sleep 5; Get-Content runs\smoke.log
```
**Expected:** a window opens and closes, and `runs\smoke.log` holds `smoke: in a job: False`. If the
log is missing, or says `True`, stop and report it before starting anything long. **Record:** the merge,
the benches, the test lines, the two comparison lines and the smoke line.

## 6. Warm start against training from scratch (about 2 days: the main GPU experiment)

Question: does starting from network 8's weights (`--init-from`) on new data give a network at least as
strong as training from scratch, in less GPU time?

**6a. New data (about 20 hours at 1,465 positions/s; resumable).** In the new directory `$R9`, eight
shards of 150,000 games with seeds never used for training (9101–9108), about 104M positions (87 a game),
plus a fixed evaluation set from seed 9999 (3,000 games, about 262,000 positions), generated first:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory -Path $R9 | Out-Null    # fails if it exists: $R9 must be new
$lines = @(ConvertTo-CmdLine @($DG, '--out', "$R9\evalset_9999.bin", '--seed', 9999, '--first-game', 0, '--games', 3000, '--threads', $T))
foreach ($s in 9101..9108) {
    $lines += ConvertTo-CmdLine @($DG, '--out', "$R9\selfplay_$s.bin", '--seed', $s, '--first-game', 0, '--games', 150000, '--threads', $T)
}
Start-Detached r9_datagen $lines
```
The 1,465 positions/s were measured with network 7; these games are played with network 8, which has
king buckets and SCReLU. Read the rate from the log after an hour (`games ... positions/s` lines) and
re-estimate: 104.4M positions ÷ rate. After a stop, `Resume-Detached r9_datagen`: each shard resumes
from its index, and finished ones report `150000 already written, 0 to play`. Then:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
& $PY tools\nnue\shards.py check $SHARDS --checksums all
& $PY tools\nnue\shards.py check "$R9\evalset_9999.bin" --checksums all
```
**Expected:** `8 shard(s), 1200000 games: consistent` and `1 shard(s), 3000 games: consistent`. The
evaluation set is not named `selfplay*`, so nothing trains on it. **Record:** positions per second, total
positions, the checks, and each `searches in this run` line.

**6b. Three training runs on the same data (about 1 hour each for A and B, half for C, one after the
other).** All three use network 8's design, the evaluation set, and the best epoch kept:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$TRAIN = @($PY, 'tools\nnue\train.py', '--data') + $SHARDS + @('--buckets', 8, '--king-buckets', 4,
          '--activation', 'screlu', '--device', 'cuda', '--eval-set', "$R9\evalset_9999.bin",
          '--select-by', 'eval:evalset_9999.bin', '--keep-best', '--resume')
Start-Detached r9_train @(
    (ConvertTo-CmdLine ($TRAIN + @('--out', "$R9\A_scratch_e20.nnue", '--epochs', 20))),              # A: from scratch
    (ConvertTo-CmdLine ($TRAIN + @('--out', "$R9\B_warm_e20.nnue", '--epochs', 20, '--init-from', $NET8))),  # B: network 8's weights, fresh optimiser
    (ConvertTo-CmdLine ($TRAIN + @('--out', "$R9\C_warm_e10.nnue", '--epochs', 10, '--init-from', $NET8)))   # C: the same, half the epochs
)
```
`--resume` makes each line start its run or continue it, so `Resume-Detached r9_train` after a stop
carries on; a resume refuses other data, settings or evaluation setup. The training/validation split is
by complete games, since these shards have indexes; the log says `split by games`. **Record from each
`<out>.json`:** the best epoch and its evaluation-set loss, the loss by piece count, the seconds per
epoch and in total (the GPU time: the sum of `history[].seconds`), and the validation loss.

**6c. Matches, declared before the first one (up to about 4 hours each, one at a time).**

Before starting the first match, copy this declaration into `docs/PC_RESULTS.md`, commit and push it.
It is fixed from then on:

> *Every match: both sides `nags_enhanced.exe` of `integ/round3` with the network named, 3+0.03,
> `$C` games at a time, `tools\openings\nags_balanced.epd`, resign adjudication 4 moves at 1000 cp,
> α = β = 0.05, at most 8,000 games.*
>
> 1. *A, B and C each against network 8: SPRT [0, 10]. H1 = stronger than network 8. Only these can
>    make a network a candidate for `nets\nags.nnue` (the user's decision).*
> 2. *Warm start against scratch, non-regression: C against A and B against A, each SPRT [−5, 0]. H1 =
>    passes (not worse than A by 5 Elo, at the stated error rates). H0 = fails. 8,000 games without a
>    decision = inconclusive, which is not a pass.*
> 3. *Warm starts become the recommended way to train only if C against A passes and C's GPU time is at
>    most about half of A's (from the manifests). If only B against A passes, a warm start is no worse at
>    the same cost and saves nothing: no recommendation. If neither passes, no recommendation.*
> 4. *An inconclusive or failed match stays as it is: no more games beyond the cap, no rerun of the same
>    pair, and an Elo estimate inside the bounds is not read as a pass. A further test is declared (pair,
>    bounds, time control, cap) in `docs/PC_RESULTS.md` and pushed before its first game.*

The matches, each started when the one before has finished:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$NET = @{ A = "$R9\A_scratch_e20.best.nnue"; B = "$R9\B_warm_e20.best.nnue"; C = "$R9\C_warm_e10.best.nnue" }
$M = @('--engine', $ENG, '--engine', $ENG, '--tc', '3+0.03', '--concurrency', $C, '--openings', $OPENINGS,
       '--resign-adjudication', 4, 1000, '--games', 8000)
Start-Match r9_A_vs_net8 ($M + @('--name', 'A', '--name', 'net8', '--option1', "EvalFile=$($NET.A)", '--sprt', 0, 10))
# then, one at a time:
# Start-Match r9_B_vs_net8 ($M + @('--name', 'B', '--name', 'net8', '--option1', "EvalFile=$($NET.B)", '--sprt', 0, 10))
# Start-Match r9_C_vs_net8 ($M + @('--name', 'C', '--name', 'net8', '--option1', "EvalFile=$($NET.C)", '--sprt', 0, 10))
# Start-Match r9_C_vs_A ($M + @('--name', 'C', '--name', 'A', '--option1', "EvalFile=$($NET.C)", '--option2', "EvalFile=$($NET.A)", '--sprt', -5, 0))
# Start-Match r9_B_vs_A ($M + @('--name', 'B', '--name', 'A', '--option1', "EvalFile=$($NET.B)", '--option2', "EvalFile=$($NET.A)", '--sprt', -5, 0))
```
For each next match, run the block again with its `Start-Match` line in place of the first (the load
line and `$NET`, `$M` are needed in every new shell). Five matches at up to about 4 hours each: up to 20
hours. With 6a (about 20-24 hours) and 6b (about 3 hours),
section 6 takes up to about 47 hours of machine time, more in calendar time. **Record:** every final
status line and decision as the declaration names it (H1, H0, or inconclusive at the cap), the
`Forfeits:` line if any, and each network's GPU minutes.

## 7. Deeper labels at equal compute (optional, after 6a; about 2 days)

The VM measured, on 300 games of seed 9201 (about 26,700 positions, 4 threads): positions where the
static eval and the 5000-node score differ by 200 cp or more (`disagree:200`) are 16.2% of positions
written (4,313, and 107 more chosen but dropped for a mate score). Relabelling them with 40,000 nodes
changes their labels by 210.6 cp on average; a random 16.4% (`random:0.165`, 4,401 and 34 dropped)
changes by 65.9 cp. The deeper searches cost about as much again as the games: game moves 105-111
thread-seconds, relabelling 114-120 (`disagree`, two runs) and 116 (`random`), against 104-105 for the
same games unrelabelled; relabelling nodes 180.1M and 181.4M. So on the VM equal fractions cost the same
to within the timing noise (about 3%), but that is a measurement to repeat on the PC, not an assumption.
Whether those labels train a better network is untested, and so is whether they are worth their
compute: the same compute could play about twice the games instead. So the arms are compared at equal
**measured** compute, not at equal fractions of positions. `nags_datagen` ends every run with
`searches in this run: game moves N nodes, S thread-seconds; relabelling M nodes (x% of all), R
thread-seconds (y% of all)`; a resumed shard prints one such line per sitting, so add them up.

**7a. Pilot: the price of each arm (about 30 minutes).** 2,000 games of a seed no other arm uses (9201),
each arm into its own directory:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory runs\relabel_pilot | Out-Null
$P = @('--seed', 9201, '--first-game', 0, '--games', 2000, '--threads', $T)
& $DG --out runs\relabel_pilot\plain.bin @P
& $DG --out runs\relabel_pilot\dis.bin @P --relabel-nodes 40000 --relabel-select disagree:200
```
From the `dis` run take the fraction of positions relabelled (`relabelled X of Y positions (Z%)`) as a
first `random:<FRACTION>` (Z/100) and run `--out runs\relabel_pilot\rnd1.bin` with
`--relabel-select random:<FRACTION>`. Disagreeing positions are not average positions, so their deeper
searches need not cost the same: compare the two runs' relabelling thread-seconds and nodes (nodes do not
vary with the machine's load; thread-seconds varied by about 3% between identical runs on the VM), scale
the fraction by their ratio, and run `rnd2.bin`, `rnd3.bin` ... until the random arm's relabelling
thread-seconds and nodes are both within 5% of the `dis` arm's. Also note the cost ratio of a
relabelled game to a plain one: `(S + R) / S_plain` for the same 2,000 games.

**7b. The arms (about 33 hours).** Seeds 9101–9103 for both relabelled arms, so that the games are those
of section 6's shards and only the labels differ:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$FRACTION = '0.165'   # the pilot's matched random fraction (0.165 on the VM)
New-Item -ItemType Directory "$R9\relabel_dis", "$R9\relabel_rnd" | Out-Null
$ARMS = [ordered]@{ relabel_dis = 'disagree:200'; relabel_rnd = "random:$FRACTION" }
$lines = @()
foreach ($s in 9101..9103) {
    foreach ($dir in $ARMS.Keys) {
        $lines += ConvertTo-CmdLine @($DG, '--out', "$R9\$dir\selfplay_$s.bin", '--seed', $s, '--first-game', 0,
                                      '--games', 150000, '--threads', $T, '--relabel-nodes', 40000, '--relabel-select', $ARMS[$dir])
    }
}
Start-Detached r9_relabel $lines
```
On the VM a relabelled game cost about 2.2 plain ones (the numbers above), so each relabelled shard
takes about 5.5 hours against 2.5 for a plain one: about 33 hours for the six. **The plain arm at equal
compute** is section 6's shards: as many of 9101–9108, in order, as cost what the `disagree` arm cost in
total (game and relabelling thread-seconds, from the `r9_relabel` log, against section 6a's per-shard
cost), rounded to the nearer whole shard; at the VM's ratio, 3 × 2.2 = 6.6, so seven. Record each arm's
actual total. Also train on the plain shards 9101–9103 (equal games, less compute) for reference.

**7c. Training (about 3 hours).** Four networks, each as A in 6b (from scratch, 20 epochs, the same
evaluation set, `--keep-best --resume`), outputs in `$R9\relabel_runs\`: `dis` (the `relabel_dis`
shards), `rnd` (the `relabel_rnd` shards), `plain3` (9101–9103) and `plainK` (the equal-compute plain
shards). Compare their evaluation-set losses; a loss decides nothing.

**7d. Matches, declared in `docs/PC_RESULTS.md` and pushed before the first:** `dis` against `rnd`,
SPRT [0, 10] at 3+0.03, at most 8,000 games (does choosing the positions beat choosing them at random,
at the same cost?); then `dis` against `plainK`, SPRT [0, 10] (does the relabelling beat spending the
same compute on more games?). Same match settings as 6c; no decision at the cap is inconclusive.
**Record:** the pilot lines, every shard's `relabelled ...` and `searches in this run` lines, each arm's
total thread-seconds, the losses, and the matches as declared.

## 8. Multithreaded strength at 4 threads (optional, about 5 hours)

The VM measured that Threads=2 against 1 gives +107.5 ± 14.0, and that double the time gives +165.4 ± 14.9.
So a second thread is worth about 65% of a time doubling. The 6-core PC can measure 4 threads, with two
fixed-length matches of 1,000 games (about 3.5 hours one game at a time, then about 1.5 hours):
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$M = @('--engine', $ENG, '--engine', $ENG, '--openings', $OPENINGS, '--resign-adjudication', 4, 1000, '--games', 1000)
Start-Match threads4 ($M + @('--name', 'threads4', '--name', 'threads1', '--option1', 'Threads=4', '--tc', '3+0.03', '--concurrency', 1))
# when it has finished:
# Start-Match time4x ($M + @('--name', 'time4x', '--name', 'time1x', '--tc1', '12+0.12', '--tc2', '3+0.03', '--concurrency', $C))
```
Nodes per second at 1 and 4 threads, 4 seconds from the start position, with nothing else running:
```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
foreach ($th in 1, 4) { & $PY -c "import chess, chess.engine, sys; e = chess.engine.SimpleEngine.popen_uci(sys.argv[1]); e.configure({'Threads': int(sys.argv[2])}); i = e.analyse(chess.Board(), chess.engine.Limit(time=4)); print(sys.argv[2], 'threads:', i.get('nps'), 'nodes/s'); e.quit()" $ENG $th }
```
**Record:** both results and the nodes per second.

## Reporting back

Create `docs/PC_RESULTS.md` on branch `pc/results` (from `integ/round3`) with one section per task above.
For each: the commands as run (with the real paths), the printed results (copied, not paraphrased),
anything that failed or differed from "Expected" and what was done about it, and the time taken. Push
after each section, and push each declaration (6c, 7d) before its first game. When an SPRT finishes,
also add its row in the strength-log format of `docs/TESTING.md`, for example:
`| what | SPRT [0, 10], 3+0.03, <concurrency> at a time on the Windows PC | H1 after N games (+W =D -L) | +x ± y | bench |`.
