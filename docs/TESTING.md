# Testing engine changes

Strength changes are only real if a statistically sound match says so. Many
plausible improvements (a new pruning rule, an evaluation tweak) lose Elo in
practice, so every functional change goes through the steps below before it
is merged.

## 1. Correctness: `ctest` and `pytest`

```bash
(cd build && ctest -C Release --output-on-failure)
python -m pytest -q
```

These check move generation (perft), hashing, search results on known
positions, UCI behaviour, the Python services, the training pipeline and the
match runner's statistics.

## 2. Fingerprint: `bench`

```bash
build/nags_enhanced bench        # or: echo bench | build/nags_enhanced
```

`bench` searches a fixed set of 32 positions (`src/Bench.h`) to a fixed depth,
single-threaded and from a clean state. It prints the total node count and the
speed:

```
Nodes searched  : 12213087
Nodes/second    : 2081657
```

The node count is a fingerprint of search behaviour, and it is deterministic
(a `ctest` checks this).

* A **non-functional change** (refactor, speed-up) must leave the node count
  unchanged. Only nodes/second should move.
* A **functional change** changes it. Put the new value at the end of the
  commit message as `Bench: <nodes>` (the convention Stockfish and most
  engines use), so any build can be checked against its commit.

Default depths: `nags_basic` and `nags_fast` 8, `nags_enhanced` 10, `nags` 9
(`nags` benches with the heuristic evaluator and without the Python
services). They were raised from 6/6/7/7 once pruning made those runs take
well under a second, and so that features that only act at higher depths
(singular extensions) show up in the fingerprint. Fingerprints at the time
of writing:

| Engine | Bench nodes |
|--------|-------------|
| `nags_basic` | 1416774 |
| `nags_fast` | 1071658 |
| `nags_enhanced` | 1053769 |
| `nags` | 1053769 |

When search improvements make a run much faster, raise the depth (in the
engine's `main` file) and record the new fingerprints. `bench` always
searches with the default hash size (64 MB), whatever `Hash` is set to.

For a non-functional change, also check that `nags_datagen` still writes
the same games: with `--threads 1` and a fixed `--seed` its output must be
byte-identical to the previous build's (bench covers the search, this also
covers move generation order and the opening playouts).

### Where the time goes

* `nags_enhanced speedtest` (or the UCI command `speedtest`): nanoseconds
  per operation for move generation, legality checks, make/unmake, SEE,
  evaluation (output layer only, and incremental after a move),
  accumulator refreshes and TT probes, on 787 positions.
* A build with `-DNAGS_STATS=ON` counts what the search does (TT hit and
  cutoff rates, node types, cutoffs by move number and by picker stage,
  each pruning and reduction rule) and `bench` prints it. The counters are
  compiled out otherwise.
* `nags` reports MCTS simulations per second and the round trips to
  `rpc_server.py` (count, failures, total/mean/maximum wait) in its
  `info string nags` line.
* `tools/search_benefit.py`: what more search buys, per decision. Each
  sample position is searched with node budgets N, 2N, 4N, ... and each
  chosen move is scored by a deep reference search; it reports how often
  each budget's move differs from the reference's, the value it loses, and
  how often doubling the budget changes the move. Time-management changes
  (and any learned time allocation) should be judged on this per-move
  measure and then by SPRT, not by game results credited to every move of
  a game.

## 3. Strength: SPRT matches with `tools/sprt.py`

Build the baseline (for example the `main` branch) into a separate directory,
then play the candidate against it:

```bash
git worktree add /tmp/base main && cmake -S /tmp/base -B /tmp/base/build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build /tmp/base/build --target nags_enhanced

python tools/sprt.py --engine build/nags_enhanced --engine /tmp/base/build/nags_enhanced \
    --tc 8+0.08 --openings tools/openings/nags_balanced.epd \
    --concurrency 3 --sprt 0 5 --pgnout stc.pgn
```

How the runner works:

* Each opening is played twice with colours reversed.
* Results are scored per game pair (the pentanomial model).
* The match stops when the generalised SPRT accepts H0 or H1. The
  log-likelihood ratio uses maximum-likelihood distributions constrained to
  each hypothesis, as fishtest does.
* Exit status: 0 = H1 accepted (the change is good), 1 = H0 accepted, 2 =
  game limit reached without a decision.
* An interrupted match continues with `--resume <its output>` (same engines,
  settings and `--seed`): the counts are restored from the last status line
  and play goes on with the next opening pair.

A Monte-Carlo simulation of 1,000 SPRTs per setting (bounds [0, 20],
α = β = 0.05) confirms the advertised error rates:

| True Elo | H1 accepted |
|----------|-------------|
| 0 | 4.5% |
| +10 (midpoint) | 50.5% |
| +20 | 94.5% |

A smaller version of this simulation runs in the test suite.

**Which bounds to use** (logistic Elo, α = β = 0.05):

| Change | Bounds | Meaning |
|--------|--------|---------|
| New feature / tweak | `--sprt 0 5` | accept only if likely to gain Elo |
| Simplification / removal | `--sprt -5 0` | accept if it does not lose Elo |
| Large expected gain (early development) | `--sprt 0 10` | faster decisions while gains are big |
| Very large expected gain (e.g. a new subsystem) | `--sprt 0 30` | quick yes/no for big jumps |

**Narrow bounds need many games, however lopsided the match.** H0 and H1
differ by Δs in expected score (Δs ≈ 0.0072 for 5 Elo, 0.0144 for 10 Elo).
Each game pair can raise the exact log-likelihood ratio by at most about
2·Δs. So reaching the acceptance bound of 2.94 takes at least about
2.94 / (2·Δs) pairs, even when one engine wins almost every game:

| Bounds | Minimum games to accept H1 |
|--------|----------------------------|
| `[0, 5]` | ~410 |
| `[0, 10]` | ~205 |
| `[0, 30]` | ~70 |

This is the test working correctly: a handful of games cannot distinguish
"+0 Elo" from "+10 Elo". It is also why narrow bounds are reserved for
small, mature changes, while early development uses wider ones.

**Time controls.** Test at a short time control first (STC, e.g. `8+0.08`).
For changes that pass STC and touch search scaling (pruning, extensions,
time management), confirm at a long time control (LTC, e.g. `40+0.4`). Keep
`--concurrency` at or below the number of physical cores minus one, otherwise
the engines steal time from each other and the results drift.

**Other options.**
* `--nodes N`, `--depth N`, `--movetime S`: fixed limits instead of a clock.
* `--option NAME=VALUE`: UCI options for both engines (`--option1` /
  `--option2` for one engine).
* `--draw-adjudication 40 8 10`, `--resign-adjudication 4 1000`: shorter
  games.
* `--games N`: run a fixed-length match without `--sprt`.

**Openings.** `tools/openings/nags_balanced.epd` holds 500 positions made by
`tools/make_openings.py` from random 6-10 ply playouts, kept only if
`nags_enhanced` evaluates them within ±60 cp at depth 7. They are varied and
roughly balanced, but some look odd. For serious testing, generate a set from
strong games:

```bash
python tools/make_openings.py --engine build/nags_enhanced --pgn AJ-CORR-PGN-000.pgn \
    --count 2000 --min-plies 8 --max-plies 12 --out tools/openings/corr_balanced.epd
```

(run `git lfs pull` first to fetch the PGN), or use an established book.
Both EPD and PGN openings work with `--openings`.

**Other runners.** For very fast time controls or big matches, a dedicated
C++ runner has lower overhead, for example
[fastchess](https://github.com/Disservin/fastchess) (`-each tc=8+0.08 -openings
file=... format=epd -games 2 -repeat -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05`)
or cutechess-cli. OpenBench can distribute SPRTs over several machines; the
`bench` output above follows the format it expects.

## 4. Absolute rating: `tools/calibrate.py`

```bash
python tools/calibrate.py --engine build/nags_enhanced --levels 1500 1800 2100 \
    --games 100 --tc 10+0.1 --concurrency 3
```

This plays a fixed match against Stockfish at several `UCI_LimitStrength` /
`UCI_Elo` levels (`apt install stockfish` provides Stockfish 16) and
combines the results into one rating estimate. Stockfish documents its
`UCI_Elo` scale as calibrated at 60+0.6 and anchored to the CCRL 40/4 list.
Results at faster time controls, and engines that scale differently with
time, make this a rough number, which is still useful for tracking progress
over months.

### Results

| Date | Build | Combined estimate (10+0.1) |
|------|-------|----------------------------|
| 2026-10-03 | `a8e5c0c` (before Steps 1 and 2: no pruning beyond the TT, hand-written evaluation) | 2329 ± 52 |
| 2026-10-04 | `af50fc5` (Step 1 search, NNUE network 5) | 2881 ± 34 |
| 2026-10-06 | `3619db2` (NNUE network 7: 8 output buckets, 56.5M positions) | 2944 ± 34 |
| 2026-10-07 | `b81c8fe` (search batch: quiescence TT, IIR and improving, history rework, continuation history, SEE pruning) | **2994 ± 32** |

#### 2026-10-07: search batch

Same setup (`nags_enhanced` at `b81c8fe`, network 7, 1 thread, 64 MB hash;
80 games per level at 10+0.1, 3 at a time). The build adds the five search
changes admitted on 2026-10-06/07 (together +165 Elo in the 3+0.03 matches
against their predecessors):

| Stockfish `UCI_Elo` | Result | Score | Elo difference | Implied rating | Previous build's score |
|---|---|---|---|---|---|
| 2800 | +39 =30 −11 | 67.5% | +127 ± 54 | 2927 ± 54 | 57% |
| 3000 | +21 =30 −29 | 45% | −35 ± 58 | 2965 ± 58 | 42% |
| 3190 (the maximum) | +9 =38 −33 | 35% | −108 ± 53 | 3082 ± 53 | 29% |

Combined (inverse-variance weighted): **2994 ± 32**, about +50 over the
previous build on the same levels. Against Stockfish the batch is worth
about a third of what the self-play matches measured: pruning gains found
at 3+0.03 shrink at 10+0.1, and self-play exaggerates gains in any case. The
levels again disagree in the same direction (2927 at 2800, 3082 at 3190).
No game was lost on time. At 35% against Stockfish's strongest
strength-limited level, this ladder is close to its ceiling; later
measurements need opponents at full strength.

#### 2026-10-06: NNUE network 7

Same setup as the 2026-10-04 run (`nags_enhanced`, 1 thread, 64 MB hash,
embedded network 7; 80 games per level at 10+0.1, 3 games at a time). The
2600 level was dropped: the previous build already scored 89% there, which
says little about the rating.

| Stockfish `UCI_Elo` | Result | Score | Elo difference | Implied rating | Previous build's score |
|---|---|---|---|---|---|
| 2800 | +35 =21 −24 | 57% | +48 ± 59 | 2848 ± 59 | 49% |
| 3000 | +16 =35 −29 | 42% | −57 ± 56 | 2943 ± 56 | 33% |
| 3190 (the maximum) | +9 =29 −42 | 29% | −152 ± 59 | 3038 ± 59 | 24% |

Combined (inverse-variance weighted): **2944 ± 34** on Stockfish's
`UCI_Elo` scale at 10+0.1. On the same three levels the 2026-10-04 build
combines to 2871, so networks 6 and 7 together are worth about +70 here,
about half of what self-play measured (+101 and +27.4); self-play matches
against the previous network usually overstate gains against other
engines. The levels again disagree in the same direction (2848 at 2800,
3038 at 3190), so "about 2850–3050 on this scale" is a fair summary. No game
was lost on time.

#### 2026-10-04: Step 1 search and NNUE network 5

`nags_enhanced` (1 thread, 64 MB hash, embedded network 5) against
Stockfish 16 with `UCI_LimitStrength`, 80 games per level at 10+0.1 on the
same 4-core VM, 3 games at a time:

| Stockfish `UCI_Elo` | Result | Score | Elo difference | Implied rating |
|---|---|---|---|---|
| 2600 | +68 =7 −5 | 89% | +370 ± 109 | 2970 ± 109 |
| 2800 | +27 =25 −28 | 49% | −4 ± 56 | 2796 ± 56 |
| 3000 | +7 =39 −34 | 33% | −122 ± 62 | 2878 ± 62 |
| 3190 (the maximum) | +5 =28 −47 | 24% | −203 ± 72 | 2987 ± 72 |

Combined (inverse-variance weighted): **2881 ± 34** on Stockfish's
`UCI_Elo` scale at 10+0.1, about 550 points above the first calibration.
The same caveats apply: the levels disagree by more than their error bars
(2796 at 2800 against 2987 at 3190), which again suggests Stockfish's
strength-limited levels are not evenly spaced at 10+0.1, so the ± 34
understates the real uncertainty. A fair summary is "about 2800–3000 on
this scale".

#### 2026-10-03: first calibration

`nags_enhanced` (1 thread, 64 MB hash) against Stockfish 16 with
`UCI_LimitStrength`, 60 games per level at 10+0.1 on a 4-core VM:

| Stockfish `UCI_Elo` | Result | Score | Elo difference | Implied rating |
|---|---|---|---|---|
| 1900 | +44 =7 −9 | 79% | +232 ± 100 | 2132 ± 100 |
| 2100 | +44 =12 −4 | 83% | +280 ± 99 | 2380 ± 99 |
| 2300 | +31 =17 −12 | 66% | +114 ± 76 | 2414 ± 76 |

Combined (inverse-variance weighted): **2329 ± 52** on Stockfish's
`UCI_Elo` scale at 10+0.1.

Treat this as a rough figure:

* The 2100 and 2300 levels agree (about 2380–2410), but the 1900 level is
  about 250 Elo lower, more than the error bars allow. Stockfish's
  strength-limited levels are calibrated at 60+0.6, so they are probably not
  evenly spaced at 10+0.1, and the ± 52 above understates the real
  uncertainty.
* The 1900 level ran on build `aa8472b`. The 2100 and 2300 levels ran on
  `a8e5c0c`, which searches identically but is about 1.6× faster, so the
  1900 result slightly understates the current engine.

A more reliable absolute rating needs longer time controls, or engines with
published CCRL ratings as opponents. Use the same ladder and time control
when comparing future versions.

### Speed log

Nodes/second from `bench` (single thread, same VM, idle). These changes do
not alter the search, so the bench node counts stay identical:

| Commit | Change | `nags_basic` | `nags_fast` | `nags_enhanced` | `nags` |
|---|---|---|---|---|---|
| `aa8472b` vs `f895704` | legality checked only for moves the search tries | 1.44× | 1.30× | 1.41× | 1.30× |
| `a8e5c0c` vs `aa8472b` | stack move list, lazy move ordering | 1.59× | 1.43× | 1.62× | 1.44× |
| `06da148` vs `253627c` | incremental evaluation (FastBoard only) | – | ~1.16× | ~1.14× | – |
| AVX2 NNUE kernels for MSVC (Windows PC, see below) | hand-written AVX2 update, refresh and output kernels, chosen by CPUID | – | 3.03× | 2.61× | 2.69× |
| `c82b96e` (4-core VM) | `makeMove` no longer zeroes a 1 KB accumulator entry (value-initialisation) | – | – | 1.057× | – |
| `6200d9f` (4-core VM) | staged move picker, same order: SEE only once captures are reached, per-stage selection | – | – | 1.05–1.07× | – |
| `80f8bd0` (4-core VM) | captures generated directly in quiescence search | – | – | 1.033× | – |

At `a8e5c0c`, `nags_enhanced` benches at about 4.8M nodes/second. The
`06da148` figures are medians of six alternating runs taken while an SPRT
occupied the other cores, so they are less precise than the rows above.

**From 2026-10-07 development moved from the cloud VM to a local Windows
PC** (AMD Ryzen 5 3500X, 6 cores / 6 threads, NVIDIA RTX 3060 12 GB, MSVC
2019): NNUE training runs on its GPU and self-play data generation on its
CPU. Results measured there are marked "Windows PC". The AVX2 row above is
from that PC (medians of three runs; `nags_enhanced` 0.57M to 1.48M
nodes/second); before it, MSVC builds used the portable NNUE code and ran
about 2.6× slower than they could.

## Strength log

Every change that affects playing strength is listed with the test that
admitted it. Matches are `nags_enhanced` unless noted, at the time control
in the Test column, with 3 games in parallel on a 4-core VM (from
2026-10-07: 5 games in parallel on the 6-core Windows PC described under the
speed log), openings from `nags_balanced.epd` and resign adjudication
(4 moves, 1000 cp).

| Change | Test | Result | Elo | Bench |
|--------|------|--------|-----|-------|
| Transposition table (`nags_enhanced` vs `nags_fast`, validation of the runner) | SPRT [0, 30], 5+0.05 | H1 after 88 games (+52 =22 -14) | +161 ± 67 | 12213087 |
| Null-move pruning (R = 3 + depth/6; not in check, at PV nodes, after a null move, near mate scores or with only pawns) | SPRT [0, 10], 5+0.05 | H1 after 402 games (+192 =100 -110) | +72 ± 30 | 5328177 |
| Reverse futility pruning (depth ≤ 6, static eval − 80·depth ≥ beta; not at PV nodes, in check or near mate scores) | SPRT [0, 10], 5+0.05 | H1 after 328 games (+163 =82 -83) | +87 ± 32 | 3295246 |
| Logarithmic late-move reductions (0.75 + ln d · ln m / 2.25, from the third move, one ply less at PV nodes) and shallow quiet-move pruning (depth ≤ 3: skip quiet moves after 3 + depth² of them, or when static eval + 120·depth ≤ alpha) | SPRT [0, 10], 3+0.03 | H1 after 1950 games (+765 =518 -667) | +17.5 ± 12.7 | 1352416 |
| Static exchange evaluation (losing captures ordered after killers and skipped in quiescence) and the countermove heuristic | SPRT [0, 10], 3+0.03 | H1 after 502 games (+233 =127 -142) | +63.7 ± 27.1 | 714206 |
| Time management: the soft limit is scaled by best-move stability (×2.0 right after the best move changed, down to ×0.85 after four stable iterations) and by score drops of more than 20 cp (up to ×1.5) | SPRT [0, 10], 3+0.03 | H1 after 2376 games (+898 =682 -796; 1 loss on time by the candidate) | +14.9 ± 11.4 | 714206 |
| History malus and gravity: on a quiet cutoff the move gets +min(depth², 1200) and every quiet move searched before it the same penalty, with h += bonus − h·|bonus|/16384 keeping values within ±16384 | SPRT [0, 10], 3+0.03 | H1 after 554 games (+231 =166 -157) | +46.7 ± 22.9 | 642327 |
| Singular extensions (depth ≥ 8, TT entry at depth − 3 or more and not an upper bound: the TT move gets one more ply if the other moves all fail low against TT score − 2·depth in a half-depth search; multi-cut if that bound is still ≥ beta) | SPRT [0, 10], 3+0.03 | H1 after 886 games (+334 =293 -259) | +29.5 ± 17.7 | 5243003 (depth 10) |

| NNUE evaluation, first network `nets/nags.nnue` (4.5M self-play positions, 20 epochs; see `docs/NNUE.md`) vs the hand-written evaluation, same binary | SPRT [0, 30], 20000 nodes per move, 1 game at a time | H1 after 110 games (+66 =21 -23) | +143 ± 71 | – |
| The same, with the clock (includes NNUE's speed cost) | SPRT [0, 30], 3+0.03 | H1 after 122 games (+69 =19 -34) | +103 ± 56 | 6429089 |
| Network 2 (8.1M positions) vs network 1; same architecture and speed, so a fixed-node match | SPRT [0, 10], 20000 nodes per move, 1 game at a time | H1 after 1234 games (+569 =202 -463) | +29.9 ± 17.9 | 7361021 |
| Network 3 (16M positions: network 2's data plus 8.0M from games played by the network-1 engine) vs network 2 | SPRT [0, 10], 3+0.03 | H1 after 214 games (+157 =35 -22) | +258 ± 51 | 5281653 |
| Network 3 vs the hand-written evaluation (consistency check) | SPRT [0, 30], 3+0.03 | H1 after 78 games (+66 =6 -6) | +354 ± 132 | – |
| Network 4a (18.8M positions, NNUE-engine games only) vs network 3 | SPRT [0, 10], 20000 nodes, 2 at a time | H1 after 400 games (+204 =79 -117) | +76.8 ± 30.4 | – |
| Network 4b (26.9M positions, all data) vs network 4a | SPRT [0, 10], 20000 nodes | H1 after 2592 games (+1124 =461 -1007) | +15.7 ± 11.8 | – |
| Network 4b vs network 3 | SPRT [0, 10], 3+0.03 | H1 after 316 games (+156 =86 -74) | +92.3 ± 32.1 | 5365610 |
| Network 5 (38.4M positions: network 4's data plus 11.5M from games by the network-4 engine) vs network 4 | SPRT [0, 10], 20000 nodes | H1 after 440 games (+220 =86 -134) | +68.8 ± 28.1 | 4496143 |
| 512-wide network on network 5's data (`NAGS_NNUE_HIDDEN=512` build, validation loss 0.00778 vs 0.00815) vs network 5 (256); the 512 build searches about 25% fewer nodes per second | SPRT [0, 10], 3+0.03 | H0 after 450 games (+122 =144 -184; 1 loss on time by the 512 build) | −48.2 ± 25.4 | – (not adopted) |
| Network 6: 8 output buckets by piece count, 47.7M positions (network 5's data plus 9.3M from games by the network-5 engine) vs network 5 | SPRT [0, 10], 3+0.03 | H1 after 304 games (+155 =80 -69) | +101.0 ± 34.2 | 5692115 |
| Network 7: 56.5M positions (network 6's data plus 8.86M from ~104,000 games by the network-6 engine), 8 buckets, vs network 6 | SPRT [0, 10], 3+0.03 | H1 after 1042 games (+388 =348 -306) | +27.4 ± 16.9 | 4780523 |
| Transposition table in quiescence search (probe with cutoffs at non-PV nodes, TT move first, TT score as a better stand-pat, results stored at depth 0) | SPRT [0, 10], 3+0.03 | H1 after 1066 games (+360 =416 -290; 1 loss on time by the baseline) | +22.8 ± 15.2 | 4318033 |
| Continuation history (quiet-move history after the previous move and the one before, in move ordering and history updates) | SPRT [0, 10], 3+0.03 | H0 after 914 games (+258 =358 -298) | −15.2 ± 16.1 | – (not adopted) |
| Internal iterative reduction (depth ≥ 4 without a TT move: one ply less) and "improving" (static eval above two plies earlier: reverse futility margin 80·(depth − improving); otherwise late-move pruning after half as many quiet moves and one more ply of reduction) | SPRT [0, 10], 3+0.03 | H1 after 772 games (+269 =299 -204) | +29.3 ± 17.7 | 2146785 |
| Correction history (per side to move and pawn structure, a running average of search score − static eval, added to the evaluation; needs a pawn hash in FastBoard) | SPRT [0, 10], 3+0.03 | H0 after 1664 games (+484 =660 -520; 1 loss on time by the candidate) | −7.5 ± 12.6 | – (not adopted) |
| History bonus min(150·depth − 100, 1500) instead of min(depth², 1200) (the old bonus left most scores under ±1000), with history pruning of quiet moves (depth ≤ 3, history < −4096·depth) and history-adjusted reductions (−history/8192 plies) | SPRT [0, 10], 3+0.03 | H1 after 1078 games (+353 =444 -281; 1 loss on time by the baseline) | +23.2 ± 15.3 | 2047814 |
| Continuation history again, on top of the new history bonus (also counted in history pruning and reductions) | SPRT [0, 10], 3+0.03 | H1 after 2396 games (+741 =1004 -651) | +13.1 ± 10.2 | 2067267 |
| SEE pruning in the main search (depth ≤ 8, not at PV nodes or in check, after the first legal move, not the TT move: quiet moves with SEE < −60·depth, captures with SEE < −20·depth²) | SPRT [0, 10], 3+0.03 | H1 after 294 games (+120 =118 -56) | +76.9 ± 30.1 | 1947402 |
| Null-move reduction 3 + depth/3 + min((eval − beta)/200, 3) and razoring (depth ≤ 3, eval + 250·depth ≤ alpha: quiescence decides), together | SPRT [0, 10], 3+0.03 | H0 after 292 games (+56 =117 -119) | −76.2 ± 29.8 | – (not adopted) |
| Time management by the best move's share of the root nodes (soft limit × clamp((1.6 − share) · 1.15, 0.6, 1.8) from depth 6) | SPRT [0, 10], 3+0.03 (resumed once) | no decision after 5032 games (+1473 =2155 -1404), LLR −0.19; stopped | +4.8 ± 6.8 | – (not adopted) |
| Transposition table in buckets of four entries (one cache line), replacing the shallowest entry with 8 plies of depth subtracted per search of age | SPRT [0, 10], 3+0.03, Hash 4 MB for both (resumed once) | H0 after 5020 games (+1410 =2221 -1389) | +1.5 ± 6.7 | – (not adopted) |
| Null-move reduction 4 + depth/6 + min((eval − beta)/200, 3) alone (gentler than the rejected one above) | SPRT [0, 10], 3+0.03 | H0 after 1148 games (+302 =512 -334) | −9.7 ± 13.9 | – (not adopted) |
| Razoring alone at depth ≤ 2 (eval + 300·depth ≤ alpha: quiescence decides) | SPRT [0, 10], 3+0.03 | H0 after 720 games (+188 =303 -229) | −19.8 ± 18.0 | – (not adopted) |
| Moves generated and scored only when their stage is reached: TT move before any generation, captures at the capture stage, quiet moves after the killers and scored with the history as it is then (`10d3d5a`) | SPRT [0, 10], 3+0.03, 3 at a time on the 4-core VM | H1 after 1246 games (+400 =516 -330) | +19.5 ± 13.8 | 1595609 |
| Fail-low nodes store no TT move (the stored move had just the highest fail-soft upper bound; it also kept IIR from applying), against `10d3d5a` (`b5bc3b5`) | SPRT [0, 10], 3+0.03, 3 at a time on the 4-core VM | H1 after 1200 games (+402 =470 -328; 1 loss on time by the candidate) | +21.5 ± 14.6 | 1053769 |
| Time management by the best move's node share, second attempt: the nodes of every aspiration attempt of the iteration are counted (the first attempt above counted only the last one, so a fail-high on a new move looked "clear"), soft limit × (1.5 − share) · 1.35 from depth 7, not under `movetime`, on top of the stability and score-drop scaling; against `8f1867b` (merged in `exp/node-tm`) | SPRT [0, 10], 3+0.03, 3 at a time on the 4-core VM | H1 after 826 games (+242 =402 -182) | +25.3 ± 16.0 | 1053769 (bench is depth-limited; unchanged) |
| Correction history again, from several keys: pawn structure, each side's non-pawn pieces and the previous move (per side to move), weighted mean ±256 cp, depth-weighted updates; the root's evaluation corrected too, and fail-low nodes update whatever move scored highest (the single-key attempt above was rejected); against `10d3d5a`, merged on top of the fail-low TT move and node-share time management | SPRT [0, 10], 3+0.03, 3 at a time on the 4-core VM | H1 after 1074 games (+319 =501 -254) | +21.1 ± 14.4 | 1831919 on `10d3d5a`; 1125414 merged |
| Singular extensions from depth 6 (was 8), two plies when far below the margin (outside PV, at most six per line, the count carried through null moves), one ply less when not singular but the TT score beats beta; against `10d3d5a`, merged on top of the changes above | SPRT [0, 10], 3+0.03, 3 at a time on the 4-core VM | H1 after 1426 games (+447 =601 -378) | +16.8 ± 12.4 | 2211180 on `10d3d5a`; 1393642 merged |
| Search correctness fixes (audit findings): quiescence returns a draw for insufficient material and scores a quiet check evasion that completes a repetition or the fifty-move rule as a draw; no TT cutoffs or TT stand-pat from a halfmove clock of 90; after an unconfirmed fail-high the aspiration loop falls back to the last completed iteration's best move (also when the fail-high move's own re-search fails low before an abort); on the head, against the head | SPRT [−5, 0] (non-regression), 3+0.03, 3 at a time on the 4-core VM | H1 after 3542 games (+913 =1778 -851) | +6.1 ± 7.4 | 1399785 |

### Experiment record

From 2026-10-14 every experiment is kept on its own `exp/*` branch and
tested against the current development head, so rejected ideas stay
available and are not repeated without a new hypothesis. For each: the
hypothesis, the bench and speed effect, the test, the decision, and
whether it touches anything a network in training depends on
(docs/COMPATIBILITY.md). Non-functional speed-ups (same bench, same
`nags_datagen` output) are in the speed log above and need no SPRT.

From 2026-10-18 the remaining experiments are merged onto the development
head first and tested against the head binary (`exp/<name>-head` branches):
correction history, +21.1 against `10d3d5a`, added only +4.9 ± 6.9 on the
head, so results against an older base overstate what a change adds now.

| Experiment (branch) | Hypothesis | Bench / speed | Test | Decision | Training impact |
|---|---|---|---|---|---|
| Lazy move generation (`exp/lazy-picker`) | Fresher history for quiet moves orders them better; no generation at TT cutoffs saves time | 1947402 → 1595609 at depth 10, 1.044× nodes/s | SPRT [0, 10]: H1, +19.5 ± 13.8 (1246 games) | accepted (`10d3d5a`) | changes future datagen games (better search), not their format |
| No TT move at fail-low nodes (`exp/faillow-ttmove`, audit finding) | The stored move at an all-node is noise: it misorders re-visits and disables IIR | 1595609 → 1053769 | SPRT [0, 10]: H1, +21.5 ± 14.6 (1200 games) | accepted (`b5bc3b5`) | as above |
| Node-share time management (`exp/node-tm`) | When the best move took most of the last iteration's nodes, more time rarely changes it; when it took little, it was contested | no bench change (time only); `nags_datagen` output identical | SPRT [0, 10]: H1, +25.3 ± 16.0 (826 games); two review rounds fixed the attempt counting, `movetime` and a MultiPV leak before the test | accepted | none (datagen uses node limits) |
| Correction history from several keys (`exp/correction-history`) | Evaluation errors repeat by pawn structure, piece set and previous move, so a learned correction makes pruning decisions more accurate | 1595609 → 1831919 on `10d3d5a` | SPRT [0, 10]: H1, +21.1 ± 14.4 (1074 games); review fixes (root eval, fail-low gate) in the tested binary; port onto the head (with the fail-low TT move and node-share time management), [−5, 0] against the head before it: H1, +4.9 ± 6.9 (4742 games), so most of the gain measured on `10d3d5a` overlaps with what the head already had | accepted (`8c39652`) | changes future datagen games, not their format |
| Singular extensions from depth 6, double and negative (`exp/singular-extensions`) | Extending forced moves earlier and harder, and searching non-singular TT moves that already beat beta less, spends the tree where the decision is | 1595609 → 2211180 on `10d3d5a` | SPRT [0, 10]: H1, +16.8 ± 12.4 (1426 games); review fix (count through null moves) in the tested binary; port onto the head (with correction history), [−5, 0] against the head before it: H1, +11.1 ± 9.4 (2404 games), so most of the gain carries over | accepted (`1826064`) | changes future datagen games, not their format |
| Search correctness fixes (`exp/search-correctness`, tested as `exp/search-correctness-head`) | Draws the search could not see (insufficient material and repetitions in quiescence, fifty-move-stale TT scores) and an unconfirmed fail-high played on an abort are errors; fixing them should not cost strength | 1393642 → 1399785 on the head | SPRT [−5, 0] on the head: H1, +6.1 ± 7.4 (3542 games); two review rounds before the test | accepted (`exp/search-correctness-head` merged) | changes future datagen games, not their format |
| Capture history (`exp/capture-history`) | History of captures orders them better than MVV/LVA alone | 1595609 → 1561627 (on `10d3d5a`) | SPRT [0, 10]: +4.9 ± 5.5 after 7566 games, LLR −0.16; paused (true effect near the middle of the bounds) and moved to the end of the queue | pending | changes future datagen games, not their format |
| v3 GNN training objectives (`tools/gnn_compare.py`) | Distilling the search's MultiPV scores gives a better move prior and value than training on the moves played | no engine change | 456 held-out positions (30 games), same data for every arm: v3 doubles v2's top-1 (22.8% vs 10.1%); with `rest_mass_scale` 0.5 distillation = played moves for the policy, better for the value (MSE vs results 0.379 vs 0.434); with 0.02 it is ahead on both (top-1 24.8%, NLL 2.45 vs 2.47; value 0.387); docs/ARCHITECTURE.md | supported at this scale (small policy margins); default `rest_mass_scale` 0.5 → 0.02; no strength claim | none (v3 is a separate format) |
| NNUE format version 4 (`exp/nnue-v4`) | A hidden stack (pairwise inputs, 16-32) after the accumulators evaluates better per position than one output layer | v1-3 bench identical; with a v4 network evaluate 33 → 117 ns, -21% nodes/s | exact engine = trainer reference on all platforms; strength needs a trained v4 and a v3 network on the same data, then an SPRT | merged as an opt-in format, not used by default | none: `train.py` and formats 1-3 untouched |

Experiments in this batch are each tested against `10d3d5a` and merged in
turn; the combination is checked by a final match of the new head against
`10d3d5a`.

Roadmap batch, 2026-10-19: on the head `105f7c3`, each against the head
binary; SPRTs at 3+0.03, 2 at a time (GNN distillation shared the 4-core VM
for part of the time).

| Experiment (branch) | Hypothesis | Bench / speed | Test | Decision | Training impact |
|---|---|---|---|---|---|
| Upcoming-repetition detection (`exp/upcoming-repetition`) | A move that can repeat an earlier position (Stockfish's cuckoo tables) lets the side to move claim at least a draw, so alpha can be raised to 0 before searching | 1399785 → 1391069 | SPRT [−5, 0]: H1, +4.4 ± 6.7 (4548 games; resumed after two machine restarts); the regression suite at 10k nodes: 258 against the head's 259 | accepted (no regression, probably a small gain) | changes future datagen games, not their format |
| ProbCut (`exp/probcut`) | A capture whose shallow search beats beta by a margin (200, less 50 when improving) makes the node a cut node; cut it after a depth − 4 search | 1399785 → 1325024 | SPRT [0, 10]: H0, −7.7 ± 12.8 (1260 games); regression suite at 10k nodes: 259, as the head | rejected (branch kept) | as above |
| TT capture in LMR (`exp/ttcapture-lmr`) | When the TT move is a capture, quiet moves are less likely to matter: one more ply of reduction | 1399785 → 1337361 | SPRT [0, 10]: H0, −6.0 ± 11.9 (1280 games); regression suite at 10k nodes: 260 against the head's 259 | rejected (branch kept) | as above |
| Fifty-move evaluation scaling (`exp/rule50-scaling`) | eval · (1 − halfmove/200) prefers progress over shuffling and sees a stale advantage for what it is | 1399785 → 1397756; `search_tests`' KR-K mate in 2 is found a ply later (depth 7), so CI is red on the branch | SPRT [0, 10]: queued | pending | as above |
| Capture history on the head (`exp/capture-history-head`) | As `exp/capture-history` (inconclusive against `10d3d5a`), measured on the head | 1399785 → 1346220; first-move cutoffs 84.6% → 85.4% | SPRT [0, 10]: queued last | pending | as above |
| GNN serving (`exp/phase1-inference`, `exp/cpp-graph`) | Same replies, less time: fast graph builder, gnn_infer backends (torch bit-identical, ONNX Runtime), threads, batching, reply cache, batched MCTS with virtual loss, engine-built graphs | no bench change; one-position round trip 24.8 → 5.6 ms (bit-identical), MCTS 41 → about 250 simulations/s on one core | tests: bit-identical replies and graphs (thousands of positions), CI | non-functional; options off by default | none: checkpoints, configs and the protocol unchanged (new fields optional) |
| Hybrid search options (`exp/phase2-hybrid`) | Tree reuse, PUCT growth, FPU, policy temperature, smart stop, time share, TT-valued leaves, root order by prior | no bench change (off by default) | per-move decisions above | kept off: no network good enough to test them in play | none |
| Game histories for v3 (`exp/gnn-history`) | A network that sees repetitions can avoid or claim them | no bench change | tests only; no network trained with histories | opt-in (checkpoint flag) | none unless the flag is set |

`nags` benches like `nags_enhanced` (without the Python services the MCTS
arm does not run); before `nags` was rebuilt on FastBoard (see below) its
bench used the ray-based board and the hand-written evaluation.

Bench values before the singular-extension row are at the old default depth
(`nags_enhanced` 7); from that row on they are at depth 10. With the network
embedded, `nags_fast`, `nags_enhanced` and `nags` bench with NNUE;
`nags_basic` (ray-based Board) keeps the hand-written evaluation.

### The NAGS hybrid

| Version | Match | Result |
|---------|-------|--------|
| Old controller (one thread shared by a bandit between alpha-beta and MCTS, ray-based board, hand-written evaluation; no Python services) vs `nags_enhanced` with network 4 | 60 games, 3+0.03 | +0 =2 -58 |
| Rebuilt controller without services | `nags_equals_nags_enhanced` test | identical bench node count (5365610 at depth 10) |
| Rebuilt controller with an untrained GNN and the meta-learner running, first version | 200 games, 3+0.03, 2 at a time | +2 =16 -182, 172 losses on time: the end of each move waited for the MCTS thread's in-flight network request (up to 3 s), and a 1 ms limit scaled to 85% became 0 ("no limit") |
| The same after the fix (MCTS only from `MctsMinTime`, pre-search time counted, network waits capped at a tenth of the hard limit) | 100 games, 3+0.03, 2 at a time | +32 =38 -30, +6.9 ± 53.1 (MCTS off at this speed; no time losses) |
| The same with `MctsMinTime` 0 (MCTS always on, untrained GNN) | 60 games, 10+0.1, 2 at a time | +12 =25 -23, −64 ± 62: the cost of the arm (15% less alpha-beta time, the GNN server competing for CPU, verified proposals up to 25 cp worse) when the network knows nothing |
| Hybrid with a stand-in v3 network distilled on the CPU (below), default settings (`exp/phase2-hybrid` and later) | `tools/hybrid_eval.py decisions`: 150 positions, 2 s a move, both moves scored by a 2M-node reference search | agree 47%; MCTS differs but not proposed 41% (its move −135 cp, worse in 51 of 54 scored); proposal rejected 11% (−208 cp, worse 14 of 16); verified override 1 (−19 cp). Alpha-beta move = reference 85.3%, MCTS top move 50.0%. 297 simulations a move (186/s) |
| The same with `NNBatch` 8, `MctsSmartStop`, `MctsTTValue` 6, `MctsCpuctBase` 19652 | as above, same positions | agree 49%; not proposed 43% (−91 cp); rejected 7% (−195 cp); verified override 1 (+13 cp). Alpha-beta = reference 89.3% (timing noise on a loaded machine), MCTS top move 52.0%. 351 simulations a move (253/s, +36%) |

The stand-in network (2026-10-19): `nags-gnn-v3` defaults (hidden 128, 6
layers, relational-v1), distilled for 5 epochs on 134k positions from
`nags_datagen` self-play, labelled with 4 MultiPV scores by `nags_enhanced`
(about 12,000-30,000 nodes, depth 7). On 8,977 positions of whole games kept
out of training: prior top-1 21.2% (uniform 5.2%), top-3 46.9%, NLL 2.481
(uniform 3.182), value correlation with the search 0.964; 3.5 ms a
position on one CPU core.

So the hybrid cannot gain with such a network: its MCTS move agrees with a
deeper search about half as often as the alpha-beta move does, verification
(correctly) rejects almost every proposal, and the arm changes about 1% of
the moves while costing 15% of the alpha-beta time and a core. No match was
played: it would measure that time cost, not the network. The hybrid stays
off by default. The decisions measurement is the one to repeat first with a
network trained at scale (GPU); a match is worth playing once the MCTS top
move agrees with the reference about as often as the alpha-beta move does.

The reduction table alone was tested first, at 5+0.05, and stopped
undecided at +21.2 ± 21.9 after 674 games (LLR 1.29); it was then tested
together with the pruning at the faster 3+0.03 to get more games per hour.
