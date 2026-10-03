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

Default depths: `nags_basic` and `nags_fast` 6, `nags_enhanced` 7, `nags` 7
(`nags` benches with the heuristic evaluator and without the Python
services). Fingerprints at the time of writing:

| Engine | Bench nodes |
|--------|-------------|
| `nags_basic` | 6858427 |
| `nags_fast` | 6839010 |
| `nags_enhanced` | 12213087 |
| `nags` | 12377207 |

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

Calibration results will be recorded here as they are measured.
