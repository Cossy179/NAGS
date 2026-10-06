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
| `nags_basic` | 1457248 |
| `nags_fast` | 1357100 |
| `nags_enhanced` | 2067267 |
| `nags` | 2067267 |

When search improvements make a run much faster, raise the depth (in the
engine's `main` file) and record the new fingerprints.

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
| 2026-10-06 | `3619db2` (NNUE network 7: 8 output buckets, 56.5M positions) | **2944 ± 34** |

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

At `a8e5c0c`, `nags_enhanced` benches at about 4.8M nodes/second. The
`06da148` figures are medians of six alternating runs taken while an SPRT
occupied the other cores, so they are less precise than the rows above.

## Strength log

Every change that affects playing strength is listed with the test that
admitted it. Matches are `nags_enhanced` unless noted, at the time control
in the Test column, with 3 games in parallel on a 4-core VM, openings from
`nags_balanced.epd` and resign adjudication (4 moves, 1000 cp).

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

Whether the hybrid can gain strength depends on a trained GNN, which has not
been trained yet.

The reduction table alone was tested first, at 5+0.05, and stopped
undecided at +21.2 ± 21.9 after 674 games (LLR 1.29); it was then tested
together with the pruning at the faster 3+0.03 to get more games per hour.
