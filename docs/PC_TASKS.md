# Tasks for the training PC (round 4)

The one handoff for the Windows PC: an AMD Ryzen 5 3500X (6 cores, 6 threads), an RTX 3060 12 GB, 16 GB RAM,
MSVC (VS 2019 Build Tools) and Python 3.13 with CUDA PyTorch (`docs/NNUE.md`). It replaces every earlier
version of this file, including the ones on `integ/round3`, `fix/round3-review` and `pc/results`.

**Run this code:** branch **`main`**, code commit **`26b2d6c`**. Later commits on `main` may change
documentation only. Section R checks this: `git diff 26b2d6c HEAD -- src CMakeLists.txt cmake tools tests nets`
must print nothing. If it prints anything, stop and ask the user before running any of it.

`main` at `26b2d6c` includes:

- the round-3 work: `integ/round3`, the PC's results and launcher (`pc/results`), and the safeguards of
  `fix/round3-review`;
- the trainer's overlap check by position, resume checks and no overwriting;
- `sprt.py`, `reexport.py` and `quant_error.py` never overwriting;
- datagen's cost line;
- the strength programme's tools (`docs/STRENGTH_PLAN.md`).

It does **not** include the stalemate change of `fix/round3-review` (`06f55df`, a stalemate with a piece left in
quiescence). Its non-regression SPRT has no result yet (the table below). `main`'s benches are therefore those
of `integ/round3`: **1417097, 1340613, 1528850, 1528850** for nags_basic, nags_fast, nags_enhanced and nags.

**Training policy.** Everything NAGS learns comes from NAGS (`docs/STRENGTH_PLAN.md`):

- every label and every training game is made by NAGS's own searches;
- the only other sources are Syzygy results (exact) and start positions from opening books;
- Stockfish is an opponent and a measuring stick only, never a teacher.

## Status and order

Machine time with nothing else running, from rates measured on this PC: self-play 1,465 positions/s with 6
threads (network 7), training `kb4` 365-393 s an epoch on 201.7M positions, matches at 3+0.03 five at a time
37-42 games a minute. A match is quoted at its game cap.

| # | Task | Machine time | Status |
|---|---|---|---|
| 0-5 | Setup; cross-machine self-play; network 8's checkpoint; its quantization error; the bias-corrected export (H0, −2.9 ± 10.1); GPU overhead (`--sync-every`: 0.9%, the default stays 1) | - | done (`docs/PC_RESULTS.md` sections 0-5) |
| S | The stalemate change's non-regression SPRT, rerun from 2026-10-30 21:39 (worktree `..\NAGS-fix`, `runs\stalemate_pc.cmd`) | up to about 3 h | **running or finished; result not yet in the repository** |
| R | Round 4 setup: a new worktree on `main`, build, checks, launcher test | 1 h | after S |
| 9 | The fused feature transformer on the RTX 3060, three times each way | 1.5 h | after R |
| 6 | Warm start against training from scratch: R9 data, three training runs, five SPRTs | up to about 47 h | after 9 |
| 7 | NAGS's own deeper labels at equal measured compute, in-game (exact positions) | about 2 days | after 6a |
| 10 | Data mixtures: retained, fresh and targeted data | about 3 days | after 6 |
| 15 | SPSA in small groups against the network chosen by then | about 14 h a group | after 6 / 7 / 10 decide the network |
| 8 | Strength at 4 threads (optional) | 5 h | any time the CPU is free |
| 11 | v3 against v4 on identical data (optional) | 3 h + matches | after 6a |
| 12 | Endgame accuracy on 5-piece tablebases (optional) | 1 h | after R |
| 13 | QAT fine-tune at scale (optional: H0 on the VM) | 1 h + matches | after 6b |

Outcomes not yet recorded anywhere in the repository. When each finishes, its result goes to
`docs/PC_RESULTS.md`, and for an SPRT also to the strength log of `docs/TESTING.md`:

| What | Where it stands | Record it in |
|---|---|---|
| Stalemate change, SPRT [−5, 0] at 3+0.03, cap 6000 (`fix/round3-review` `2384ebb` against `integ/round3` `7464810`) | started on the PC 2026-10-30 21:39; the VM's first run stopped undecided at 2422 games (+4.3 ± 8.6) | `pc/results` (as planned), then the user decides whether `06f55df` goes to `main` |
| Warm start against scratch (section 6) | not started | `pc/round4` |
| Deeper labels at equal compute (section 7) | not started | `pc/round4` |
| Fused path on the GPU (section 9) | not run | `pc/round4` |
| Strength at 4 threads (section 8) | not run | `pc/round4` |

## Rules for this machine (read before running anything)

1. **Do not disturb a job that is running.** First check for running `python`, `nags_datagen` and
   `nags_enhanced` processes (`Get-Process python, nags_datagen, nags_enhanced -ErrorAction SilentlyContinue`).
   Ask the user before continuing if any exists. The stalemate SPRT (S) uses the binaries of the worktrees
   `..\NAGS-round3` and `..\NAGS-fix`. Do not rebuild, update or delete either of them while it runs.
2. **Never overwrite or delete** `nets\nags.nnue`, existing `selfplay_*.bin` files or their `.games` indexes,
   existing `*.ckpt` checkpoints, or `round_*.txt` logs. Every output goes to a new file or directory:
   `nets\candidates\`, `runs\` and `$R9`. The tools enforce this:
   - `train.py` refuses an existing `--out` (or its `.ckpt`, `.json`, `.best.nnue`) unless `--resume`
     continues that run.
   - `sprt.py` refuses a `--pgnout` that holds games unless `--resume` continues that match.
   - `reexport.py --out` and `quant_error.py --json` refuse existing files.
   - `nags_datagen` continues a shard only under identical settings.
   - `Start-Detached` refuses a job name that exists.

   If a tool refuses, do not delete its earlier output: choose another name, or resume.
3. **Use the new worktree** `..\NAGS-main` (section R) for everything from section R on. Leave `..\NAGS-round3`
   and `..\NAGS-fix` as they are.
4. **Start long runs detached from the Claude app**, with `Start-Detached` or `Start-Match` (below). A process
   started from the app's shell with `Start-Process` stays inside the app's Windows job object (`IsProcessInJob`
   reports `True`) and ends when the app updates itself (`docs/PC_RESULTS.md`, section 4).
   - `Resume-Detached` opens the job's `.cmd` through `explorer.exe`, which starts it outside that job (`False`).
   - Each job is a `runs\NAME.cmd` whose output goes to `runs\NAME.log`. The job stops at the first command that
     fails.
   - Only one copy of a job runs at a time (`runs\NAME.lock`).
   - After any stop, `Resume-Detached NAME` runs the same `.cmd` again. Every command in it continues where it
     stopped: shards from their indexes, `train.py --resume`, `sprt.py --resume`.
5. **One heavy job at a time.** Data generation, a match, and GPU training or a GPU timing never overlap: no two
   of them run at the same time. GPU training does not run alongside data generation either. That is the
   user's rule for this machine, and it is stricter than the GPU's limits.
6. **Only an SPRT decides strength, and only as declared before its first game.** Copy each declaration into
   `docs/PC_RESULTS.md`, commit and push it before the first game.
   - H1 passes and H0 fails.
   - A match that reaches its cap without a decision is **inconclusive, never a pass**. No more games beyond
     the cap, no rerun of the same pair, and an Elo estimate inside the bounds is not read as a pass.
   - Record every number as it was printed: games, W/D/L, Elo ± error, LLR, the decision, the `Forfeits:` line.
   - Do not replace `nets\nags.nnue`, merge to `main` or push to `main`: report, and the user decides.
7. **Results go to `docs/PC_RESULTS.md` on branch `pc/round4`**, in the new worktree. Commit and push that
   branch only (`git push -u origin pc/round4`) after each section. The stalemate SPRT's result goes to
   `pc/results`, where it was planned. `runs\` and `nets\candidates\` are not committed.
8. **Write nothing into an earlier round's data directory.** `train.py` caches each data file's SHA-256 next to
   it, so add `--no-data-cache` whenever it reads another round's data (section 9 and section 10's `$DATA8`).

## Paths, threads and detached jobs: `runs\pc_env.ps1`

Every PowerShell block below starts by loading `runs\pc_env.ps1`, made in section R from
`docs\pc_env.example.ps1`:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
```

It defines, with this machine's real paths (it stops if `$ROOT` or `$PY` does not exist):

- **Paths:**
  - `$ROOT`: the worktree.
  - `$PY`: the CUDA venv's `python.exe`.
  - `$DATA8`: network 8's round directory, read only.
  - `$R9`: section 6's new directory.
  - `$ENG` and `$DG`: the worktree's `nags_enhanced.exe` and `nags_datagen.exe`.
  - `$NET8`: its `nets\nags.nnue`.
  - `$SHARDS`: the eight shard paths of section 6, each named, because PowerShell passes a wildcard to a
    program unexpanded and `shards.py` does not expand one.
- **Threads:**
  - `$T`, every logical processor, for self-play (`nags_datagen --threads $T`).
  - `$C = $T - 1` for matches (`sprt.py --concurrency $C`).
- **Helpers:**
  - `ConvertTo-CmdLine ARRAY`: one command line for a `.cmd` file, every argument quoted. **Commands are built
    as PowerShell arrays, never by splitting a string.**
  - `Start-Detached NAME LINES`: the job described in rule 4. `Resume-Detached NAME` continues it, and
    `Test-DetachedRunning NAME` says whether it runs.
  - `Start-Match NAME ARGS`: `Start-Detached` for `tools\sprt.py ARGS --pgnout runs\NAME.pgn --resume
    runs\NAME.log`. Started again, the match continues from its log, and a decided SPRT reports its decision
    without playing on.

Check a job with `Get-Content runs\NAME.log -Tail 5`.

## S. The stalemate change's SPRT (running; do not disturb)

Started 2026-10-30 21:39 from `..\NAGS-fix\runs\stalemate_pc.cmd` under its original plan (`tools/funnel.py --kind
fix`: SPRT [−5, 0] at 3+0.03, cap 6000, 5 at a time). Candidate: `fix/round3-review` `2384ebb`. Base:
`integ/round3` `7464810` (`docs/PC_RESULTS.md`, "Before section 6").

When it ends, record its final lines on `pc/results` as planned, and the row in `docs/TESTING.md`'s experiment
record ("Stalemate with a piece left in quiescence"). The result is one of:

- **H1:** the user decides whether `06f55df` goes into `main`. Until then this round runs on `main` as pinned
  above, without it.
- **H0, or inconclusive at the cap:** nothing changes.

Wait for it to finish before section R starts anything heavy (rule 5).

## R. Round 4 setup: a new worktree on `main` (about 1 hour)

```powershell
cd C:\Users\Alex\Documents\code\NAGS-round3      # the existing worktree, only to reach the repository
$env:GIT_LFS_SKIP_SMUDGE = 1                     # no task needs the 1 GB LFS file AJ-CORR-PGN-000.pgn
git fetch origin
git worktree add -b pc/round4 ..\NAGS-main origin/main
cd ..\NAGS-main
git diff 26b2d6c HEAD -- src CMakeLists.txt cmake tools tests nets   # must print nothing (see the top)
New-Item -ItemType Directory runs | Out-Null
Copy-Item docs\pc_env.example.ps1 runs\pc_env.ps1
```

Edit the four `EDIT` lines of `runs\pc_env.ps1`:

- `$ROOT`: this directory (`(Get-Location).Path`).
- `$PY`: the CUDA venv's `python.exe`, as in `..\NAGS-round3\runs\pc_env.ps1`.
- `$DATA8`: `D:\NAGS\data`, network 8's round.
- `$R9`: a directory that does not exist yet, for example `D:\NAGS\r9`.

Then:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
cmake -B build
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
foreach ($e in "nags_basic","nags_fast","nags_enhanced","nags") { & ".\build\Release\$e.exe" bench | Select-String "Nodes searched" }
& $PY -m pip install -r requirements.txt -r requirements-onnx.txt   # keeps the venv's CUDA torch
& $PY -m pytest -q -rs
```

**Expected:**

- ctest 49/49.
- Benches **1417097, 1340613, 1528850, 1528850**.
- pytest: everything passes except the SIGINT funnel test, which is skipped on Windows by design.

**If a bench differs, stop and report it.** Then check self-play against the VM's reference games (the same 400
games as section 1, into this worktree's `runs\xm\pc.bin`):

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory runs\xm | Out-Null
& $DG --out runs\xm\pc.bin --nodes 5000 --seed 77 --first-game 0 --games 400 --threads $T
& $PY -c "import sys; sys.path.insert(0,'tools/nnue'); import shards; ref=[l.split() for l in open('docs/data/ref_seed77_n5000_g400.games').read().splitlines()[1:]]; r={int(x[0]):(x[1],x[2],x[4]) for x in ref}; h,g=shards.read_index('runs/xm/pc.bin'); p={x.number:(str(x.records),str(x.bytes),x.checksum) for x in g}; bad=sorted(k for k in r if r[k]!=p.get(k)); print('games', len(p), 'identical to the VM:', not bad, 'first differing game:', bad[:1]); print('header equal:', h==open('docs/data/ref_seed77_n5000_g400.games').readline().rstrip())"
```

**Expected:** `games 400 identical to the VM: True` and `header equal: True`. The header ends `search=1528850`.
Then test the launcher once:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
Start-Detached smoke (ConvertTo-CmdLine @($PY, '-c', "import ctypes; k = ctypes.windll.kernel32; r = ctypes.c_int(); k.IsProcessInJob(k.GetCurrentProcess(), None, ctypes.byref(r)); print('smoke: in a job:', bool(r.value))"))
Start-Sleep 5; Get-Content runs\smoke.log
```

**Expected:** `smoke: in a job: False`. If the log is missing or says `True`, stop and report it.

**Record:** the `git diff` check, the benches, the test lines, both comparison lines and the smoke line.

## 9. The fused feature transformer on the RTX 3060 (about 1.5 hours)

`--ft-impl fused` computes the same network as the default path, to float rounding, with about a third of the
rows gathered and scattered (`docs/STRENGTH_PLAN.md`, item 9). On the VM's CPU it was 1.4x faster per batch.
Measure it on network 8's data (read only). Do it three times, alternating, with nothing else running, as in
section 5:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$D8 = @(Get-ChildItem -LiteralPath $DATA8 -Filter 'selfplay*.bin' | Sort-Object Name | ForEach-Object FullName)
"$($D8.Count) files: $($D8[0]) ... $($D8[-1])"
$TRAIN8 = @($PY, 'tools\nnue\train.py', '--data') + $D8 + @('--buckets', 8, '--king-buckets', 4, '--activation', 'screlu', '--device', 'cuda', '--no-data-cache')
$lines = @()
foreach ($r in 1..3) {
    foreach ($impl in 'embedding', 'fused') { $lines += ConvertTo-CmdLine ($TRAIN8 + @('--out', 'runs\prof.nnue', '--profile', 200, '--ft-impl', $impl)) }
}
foreach ($r in 1..3) {
    foreach ($impl in 'embedding', 'fused') { $lines += ConvertTo-CmdLine ($TRAIN8 + @('--out', "runs\ep1_${impl}_r$r.nnue", '--epochs', 1, '--ft-impl', $impl)) }
}
Start-Detached fused_timing $lines
```

When it has finished:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
Select-String -Path runs\fused_timing.log -Pattern '^profile:'
foreach ($r in 1..3) { foreach ($i in 'embedding', 'fused') { $h = (Get-Content -Raw "runs\ep1_${i}_r$r.nnue.json" | ConvertFrom-Json).history[0]; "$i r$r seconds $($h.seconds) val $($h.val)" } }
```

**Decision (declared here):** `fused` becomes the default of later runs (`run_round.py`, and the commands below)
only if both hold:

1. its epoch is at least 1.2 times as fast as `embedding`'s in each of the three pairs;
2. each fused validation loss is within 0.5% of its pair's.

The two paths round differently, so the networks are not byte-identical. Otherwise the default stays
`embedding`.

**Record:** the six profile lines, the six epoch times and validation losses, and the decision.

The training commands of sections 6, 7 and 10 use `$FT`. Set it on its `EDIT` line in `runs\pc_env.ps1`, so that
every shell has it: `@('--ft-impl', 'fused')` if this section adopted the fused path, otherwise `@()`.

## 6. Warm start against training from scratch (up to about 47 hours: the main GPU experiment)

Question: does starting from network 8's weights (`--init-from`) on new data give a network at least as strong as
training from scratch, in less GPU time?

**6a. New data (about 20-24 hours; resumable).** In the new directory `$R9`:

- eight shards of 150,000 games with seeds never used for training (9101-9108), about 104M positions;
- a fixed evaluation set from seed 9999 (3,000 games, about 262,000 positions), generated first.

All of them use `--record-state on`, so the records are exact positions: castling rights, en passant and move
counters in the reserved bytes, which the trainer ignores. Any later offline relabelling of them is then exact
(`tools/nnue/rescore.py` refuses records without that state):

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory -Path $R9 | Out-Null    # fails if it exists: $R9 must be new
$lines = @(ConvertTo-CmdLine @($DG, '--out', "$R9\evalset_9999.bin", '--seed', 9999, '--first-game', 0, '--games', 3000, '--threads', $T, '--record-state', 'on'))
foreach ($s in 9101..9108) {
    $lines += ConvertTo-CmdLine @($DG, '--out', "$R9\selfplay_$s.bin", '--seed', $s, '--first-game', 0, '--games', 150000, '--threads', $T, '--record-state', 'on')
}
Start-Detached r9_datagen $lines
```

Read the rate from the log after an hour (the `games ... positions/s` lines) and re-estimate: 104.4M positions ÷
rate. After a stop, `Resume-Detached r9_datagen`. Then:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
& $PY tools\nnue\shards.py check $SHARDS --checksums all
& $PY tools\nnue\shards.py check "$R9\evalset_9999.bin" --checksums all
```

**Expected:** `8 shard(s), 1200000 games: consistent` and `1 shard(s), 3000 games: consistent`. Every header
ends `... search=1528850 state=1`.

**Record:** positions per second, total positions, the checks, and each `searches in this run` line.

**6b. Three training runs on the same data, one after the other.** A and B take about 1 hour each, C half that.
All three use network 8's design, the evaluation set, and the best epoch kept:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$TRAIN = @($PY, 'tools\nnue\train.py', '--data') + $SHARDS + @('--buckets', 8, '--king-buckets', 4,
          '--activation', 'screlu', '--device', 'cuda', '--eval-set', "$R9\evalset_9999.bin",
          '--select-by', 'eval:evalset_9999.bin', '--keep-best', '--resume') + $FT
Start-Detached r9_train @(
    (ConvertTo-CmdLine ($TRAIN + @('--out', "$R9\A_scratch_e20.nnue", '--epochs', 20))),
    (ConvertTo-CmdLine ($TRAIN + @('--out', "$R9\B_warm_e20.nnue", '--epochs', 20, '--init-from', $NET8))),
    (ConvertTo-CmdLine ($TRAIN + @('--out', "$R9\C_warm_e10.nnue", '--epochs', 10, '--init-from', $NET8)))
)
```

`--resume` makes each line start its run or continue it. A resume refuses other data, settings or evaluation
setup. The log says `split by games`.

**Record from each `<out>.json`:**

- the best epoch and its evaluation-set loss;
- the loss by piece count;
- the seconds per epoch and in total (the GPU time);
- the validation loss.

**6c. Matches, declared before the first one (up to about 4 hours each, one at a time).** Copy this declaration
into `docs/PC_RESULTS.md`, commit and push it before the first game:

> *Every match: both sides `nags_enhanced.exe` of `main` at `26b2d6c` with the network named, 3+0.03, `$C` games
> at a time, `tools\openings\nags_balanced.epd`, resign adjudication 4 moves at 1000 cp, α = β = 0.05, at most
> 8,000 games.*
>
> 1. *A, B and C each against network 8: SPRT [0, 10]. H1 = stronger than network 8. Only these can make a network
>    a candidate for `nets\nags.nnue` (the user's decision).*
> 2. *Warm start against scratch, non-regression: C against A and B against A, each SPRT [−5, 0]. H1 = passes (not
>    worse than A by 5 Elo, at the stated error rates). H0 = fails. 8,000 games without a decision =
>    inconclusive, which is not a pass.*
> 3. *Warm starts become the recommended way to train only if C against A passes and C's GPU time is at most about
>    half of A's (from the manifests). If only B against A passes, a warm start is no worse at the same cost and
>    saves nothing: no recommendation. If neither passes, no recommendation.*
> 4. *An inconclusive or failed match stays as it is (rule 6). A further test is declared (pair, bounds, time
>    control, cap) in `docs/PC_RESULTS.md` and pushed before its first game.*

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

For each next match, run the block again with its `Start-Match` line in place of the first. The load line, `$NET`
and `$M` are needed in every new shell.

**Record:**

- every final status line;
- each decision as declared (H1, H0, or inconclusive at the cap);
- the `Forfeits:` line, if any;
- each network's GPU minutes.

## 7. NAGS's own deeper labels at equal measured compute (about 2 days; after 6a)

Question: are deeper labels for chosen positions worth their compute, compared with spending the same compute on
more games?

The labels are made **during play**. `nags_datagen --relabel-nodes N --relabel-select RULE` plays exactly the
games of the plain run and searches the chosen positions again with N nodes, with a second searcher. The
positions are therefore exact; nothing is reconstructed from records.

**What the VM measured** (300 games of seed 9201, 4 threads):

- `disagree:200` chose 16.2% of the positions and changed their labels by 210.6 cp on average.
- A random 16.4% changed by 65.9 cp.
- The deeper searches cost about as much again as the games: relabelling took 114-120 thread-seconds against
  105-111 for the moves.

Every run ends with `searches in this run: game moves N nodes, S thread-seconds; relabelling M nodes (x% of all),
R thread-seconds (y% of all)`. A resumed shard prints one such line per sitting, so add them up. Arms are
compared at equal **measured** compute, not at equal fractions of positions.

**7a. Pilot: the price of each arm (about 30 minutes).** Run 2,000 games of seed 9201 (no other arm uses it):

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory runs\relabel_pilot | Out-Null
$P = @('--seed', 9201, '--first-game', 0, '--games', 2000, '--threads', $T, '--record-state', 'on')
& $DG --out runs\relabel_pilot\plain.bin @P
& $DG --out runs\relabel_pilot\dis.bin @P --relabel-nodes 40000 --relabel-select disagree:200
```

From the `dis` run, take the fraction relabelled (`relabelled X of Y positions (Z%)`) as a first `random:Z/100`.
Run `--out runs\relabel_pilot\rnd1.bin` with it.

Then scale the fraction by the ratio of the two runs' relabelling thread-seconds and nodes, and run `rnd2.bin`,
`rnd3.bin` and so on. Stop when the random arm's relabelling thread-seconds and nodes are both within 5% of the
`dis` arm's.

Also note the cost ratio of a relabelled game to a plain one: `(S + R) / S_plain`.

**7b. The arms (about 33 hours).** Both relabelled arms use seeds 9101-9103, so the games are those of section 6's
shards and only the labels differ:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$FRACTION = '0.165'   # the pilot's matched random fraction
New-Item -ItemType Directory "$R9\relabel_dis", "$R9\relabel_rnd" | Out-Null
$ARMS = [ordered]@{ relabel_dis = 'disagree:200'; relabel_rnd = "random:$FRACTION" }
$lines = @()
foreach ($s in 9101..9103) {
    foreach ($dir in $ARMS.Keys) {
        $lines += ConvertTo-CmdLine @($DG, '--out', "$R9\$dir\selfplay_$s.bin", '--seed', $s, '--first-game', 0,
                                      '--games', 150000, '--threads', $T, '--record-state', 'on',
                                      '--relabel-nodes', 40000, '--relabel-select', $ARMS[$dir])
    }
}
Start-Detached r9_relabel $lines
```

**The plain arm at equal compute** is made of section 6's shards:

- Take as many of 9101-9108, in order, as cost what the `disagree` arm cost in total. That total is its game and
  relabelling thread-seconds from the `r9_relabel` log, compared with 6a's per-shard cost.
- Round to the nearer whole shard. At the VM's ratio that is 3 × 2.2 = 6.6 shards, so seven.
- Record each arm's actual total.
- Also train on the plain shards 9101-9103 (equal games, less compute), for reference.

**7c. Training (about 3 hours).** Four networks, each trained as A in 6b (from scratch, 20 epochs, the same
evaluation set, `--keep-best --resume`, `$FT`). Outputs go in `$R9\relabel_runs\`:

| Network | Shards |
|---|---|
| `dis` | the `relabel_dis` shards |
| `rnd` | the `relabel_rnd` shards |
| `plain3` | 9101-9103 |
| `plainK` | the equal-compute plain shards |

Compare their evaluation-set losses; a loss decides nothing.

**7d. Matches, declared in `docs/PC_RESULTS.md` and pushed before the first.** Both use 6c's match settings, and no
decision at the cap is inconclusive:

1. `dis` against `rnd`, SPRT [0, 10] at 3+0.03, at most 8,000 games: does choosing the positions beat choosing them
   at random, at the same cost?
2. `dis` against `plainK`, SPRT [0, 10]: does the relabelling beat spending the same compute on more games?

Deeper labels become the way later rounds label only if `dis` beats `plainK` (H1).

**Record:** the pilot lines, every shard's `relabelled ...` and `searches in this run` lines, each arm's total
thread-seconds, the losses, and the matches as declared.

## 10. Data mixtures: retained, fresh and targeted data (about 3 days; after 6)

The question of `docs/STRENGTH_PLAN.md` item 3. `tools/nnue/mix.py build` keeps each unit's identity:

- a whole game (`seed:game@file`);
- independent positions (tablebase data);
- a block of a file without an index. Network 8's old data has no index, so its records are taken in 1,024-record
  blocks, written first, and the trainer splits validation after them (`docs/NNUE.md`).

A damaged index, or one that does not cover its file, is refused.

**10a. Targeted sources (each a few hours of CPU, one at a time).**

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory books, "$R9\targeted" | Out-Null
$D8 = @(Get-ChildItem -LiteralPath $DATA8 -Filter 'selfplay*.bin' | Sort-Object Name | ForEach-Object FullName)
& $PY tools\nnue\mix.py report @SHARDS --json runs\mix_r9.json
& $PY tools\nnue\mix.py report @D8 --json runs\mix_r8.json
& $PY tools\nnue\mix.py book @SHARDS --select imbalanced --count 5000 --max-score 300 --out books\imbalanced.epd
& $DG --out "$R9\targeted\imb_9201.bin" --seed 9201 --first-game 0 --games 60000 --threads $T --record-state on --book books\imbalanced.epd --random-plies 2 --max-opening-score 600
```

**Tablebase endgames,** if the 3-4-5 Syzygy tables are on this machine (`$TB345` in `runs\pc_env.ps1`,
about 1 GB; WDL files suffice). The results are exact, and the scores are NAGS's search without tablebases:

```powershell
& $PY tools\nnue\tb_data.py generate --syzygy $TB345 --material KRvKB KRvKN KBvKP KNvKP KRPvKR KBPvKB KNPvKB KRPvKB KPvKP KRvKBP --count 100000 --engine $ENG --nodes 5000 --workers $T --out "$R9\targeted\tb_1.bin"
```

**Opening variety** from human games uses start positions only, like any opening book. The games played from them
and every label are NAGS's. It needs `git lfs pull` for `AJ-CORR-PGN-000.pgn`:

```powershell
& $PY tools\make_openings.py --engine $ENG --pgn AJ-CORR-PGN-000.pgn --count 5000 --min-plies 8 --max-plies 16 --out books\corr.epd
& $DG --out "$R9\targeted\corr_9202.bin" --seed 9202 --first-game 0 --games 60000 --threads $T --record-state on --book books\corr.epd --random-plies 2
```

**10b. Three mixtures of about 100M positions, each trained like A in 6b** (from scratch, 20 epochs, the
evaluation set, `--keep-best --resume`, `$FT`):

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
New-Item -ItemType Directory "$R9\mix" | Out-Null
& $PY tools\nnue\mix.py build --source "$R9\selfplay_910?.bin=1" --positions 100000000 --max-copies 4 --out "$R9\mix\m0.bin"
& $PY tools\nnue\mix.py build --source "$R9\selfplay_910?.bin=0.5" --source "$DATA8\selfplay_*.bin=0.5" --positions 100000000 --max-copies 4 --out "$R9\mix\m1.bin"
& $PY tools\nnue\mix.py build --source "$R9\selfplay_910?.bin=0.45" --source "$DATA8\selfplay_*.bin=0.45" --source "$R9\targeted\imb_*.bin=0.04" --source "$R9\targeted\corr_*.bin=0.04" --source "$R9\targeted\tb_*.bin=0.02" --positions 100000000 --max-copies 4 --out "$R9\mix\m2.bin"
```

`mix.py` expands the quoted wildcards itself. It reads `$DATA8`'s files without writing anything next to them, but
train.py on a mixture writes `<mixture>.sha256` only in `$R9\mix`. If the tablebase or corr sources were not made,
leave their `--source` out and give their shares to the others.

**10c. Matches, declared and pushed before the first:** m1 against m0, then m2 against the better of the two. Both
are SPRT [0, 10] at 3+0.03 with 6c's settings.

**Record:** each `.mix.json` (shares, records by kind), the evaluation-set losses by piece count, section 12's
endgame accuracy for each network if the tables are here, and the matches as declared.

## 15. SPSA in small groups (about 14 hours a group; after the network is chosen)

`docs/STRENGTH_PLAN.md` item 4. The tuning build makes the search constants of `src/Tune.h` UCI options; with
their defaults it benches like the normal build:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
cmake -B build-tune -DNAGS_TUNE=ON
cmake --build build-tune --config Release --target nags_enhanced --parallel
& .\build-tune\Release\nags_enhanced.exe bench | Select-String "Nodes searched"   # 1528850
$TUNE = Join-Path $ROOT 'build-tune\Release\nags_enhanced.exe'
Start-Detached spsa_A (ConvertTo-CmdLine @($PY, 'tools\spsa.py', '--engine', $TUNE, '--params', 'LmrBase', 'LmrDivisor', 'LmrHistoryDivisor', '--iterations', 2000, '--pairs', 4, '--tc', '3+0.03', '--concurrency', $C, '--state', 'runs\spsa_A.json'))
```

The groups, by the names in `src/Tune.h`, run one at a time:

| Group | Parameters |
|---|---|
| A | `LmrBase LmrDivisor LmrHistoryDivisor` |
| B | `RfpMargin FutilityMargin` |
| C | `NmpBase NmpDivisor` |
| D | `SeeQuietMargin SeeCaptureMargin HistoryPruneMargin` |
| E | `SingularMargin DoubleExtensionMargin AspirationWindow` |

`spsa.py` tunes the network the build embeds, and has no option for another. So tune after the chosen network has
become `nets\nags.nnue` (the user's decision), in a build made after that.

**Declared validation for each group:**

1. SPRT [0, 10] at 3+0.03 of the tuned values against the defaults in the tuning build. It uses openings not used
   in tuning: `tools\make_openings.py --engine $ENG --seed 2 --count 500 --out books\validation.epd`, then
   `--openings books\validation.epd`. The 500 openings of `nags_balanced.epd` are all used by a 2,000-iteration
   run.
2. If H1: the values copied into `src/Tune.h` on a branch, and the normal build's SPRT [0, 10] at 3+0.03.
3. Then [−5, 0] at 10+0.1.

A group whose values fail any step is not tuned again without a new hypothesis. A change to `src/Tune.h` is a code
change for the user to merge.

## 8. Strength at 4 threads (optional, about 5 hours)

The VM measured Threads=2 against 1 at +107.5 ± 14.0, and double the time at +165.4 ± 14.9, so a second thread is
worth about 65% of a time doubling. Here: two fixed-length matches of 1,000 games, one game at a time for the
4-thread one:

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

## 11. v3 against v4 on identical data (optional; about 3 hours of GPU, then matches; after 6a)

Section 6's A is the v3 network. Train v4 on the same shards with the same epochs:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
Start-Detached r9_v4 (ConvertTo-CmdLine (@($PY, 'tools\nnue\train_v4.py', '--data') + $SHARDS + @('--buckets', 8, '--king-buckets', 4, '--activation', 'pairwise', '--device', 'cuda', '--epochs', 20, '--resume', '--out', "$R9\v4_e20.nnue")))
```

Run `speedtest` and `go nodes 1500000` from a middlegame, with each network loaded by `EvalFile`.

**Declared matches:**

1. v4 against A, SPRT [0, 10] at 20,000 nodes a move (`--nodes 20000`).
2. If H1: at 3+0.03.
3. If that passes too: [−5, 0] at 10+0.1.

v4 becomes the default design only if step 2 passes.

**Record:** both validation losses, the `speedtest` lines, nodes per second with each network, and every match.

## 12. Endgame accuracy on 5-piece tablebases (optional, about 1 hour)

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
& $PY tools\nnue\tb_data.py measure --syzygy $TB345 --material KRvKB KRvKN KNNvK KBvKP KNvKP KRPvKR KBPvKB KNPvKB KRPvKB KRvKBP KQvKRP KPvKPP --count 400 --engine $ENG --nodes 10000 --workers $C --json runs\tb_net8.json
```

On the VM network 8 said 75.0% of 3-4 piece results right; KRvKB was 8%, KRvKN 18% and KNNvK 1%.

**Record:** the per-material lines, and the same for each new network (`--option EvalFile=...`).

## 13. QAT fine-tune at scale (optional; after 6b)

On the VM a QAT fine-tune lost its SPRT against the same fine-tune without QAT: H0, −10.9 ± 14.3 after 922 games.
Run this only when the GPU is free. It tests QAT at scale, where the result could differ.

Arm D is B's command with `--qat` added. QAT implies the fused path, so leave out `$FT`:

```powershell
Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
$TRAIN = @($PY, 'tools\nnue\train.py', '--data') + $SHARDS + @('--buckets', 8, '--king-buckets', 4, '--activation', 'screlu', '--device', 'cuda', '--eval-set', "$R9\evalset_9999.bin", '--select-by', 'eval:evalset_9999.bin', '--keep-best', '--resume')
Start-Detached r9_qat (ConvertTo-CmdLine ($TRAIN + @('--out', "$R9\D_warm_qat_e20.nnue", '--epochs', 20, '--init-from', $NET8, '--qat')))
```

Then compare the two kept networks as the engine runs them. This writes the reports into `runs\`; an existing one
is refused:

```powershell
foreach ($x in 'B_warm_e20', 'D_warm_qat_e20') { & $PY tools\nnue\quant_error.py --net "$R9\$x.best.nnue" --data "$R9\evalset_9999.bin" --positions 100000 --json "runs\qe_$x.json" }
```

**Declared match:** D against B, SPRT [0, 10] at 3+0.03, with 6c's settings.

## Reporting back

Results go to `docs/PC_RESULTS.md` on `pc/round4`, one section per task, each with:

- the commands as run, with the real paths;
- the printed results, copied, not paraphrased;
- anything that differed from "Expected", and what was done about it;
- the time taken.

Push after each section, and push each declaration before its first game. When an SPRT finishes, also add its row
in the strength-log format of `docs/TESTING.md`, for example
`| what | SPRT [0, 10], 3+0.03, 5 at a time on the Windows PC | H1 after N games (+W =D -L) | +x ± y | bench |`.
