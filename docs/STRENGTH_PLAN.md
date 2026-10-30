# Programme: much greater strength

Top-engine strength needs a better evaluation, a better search and sustained testing together. More
positions from the same teacher may only preserve the same weaknesses. This document turns that into
twelve lines of work, in priority order. For each it gives the evidence, the tools, and the experiment
with its decision rule. Each rule is written down before the experiment runs.

The starting point is **network 8** on `integ/round3` (bench 1528850). It rates 3115 ± 31 on
Stockfish's `UCI_Elo` scale on the Windows PC, a scale whose top is nearly reached. On the node ladder
an older build needed about **32 times Stockfish 16's nodes** for the same results: `989184c` with
network 7 rated −13 ± 37 at 256k nodes against sf16 at 8k-32k. That figure is for network 7, so
refreshing it for network 8 is the first measurement (item 11, step 1). It is also the best reason to
judge every change by the quality of its decisions per node, not only by self-play.

The rules of `docs/TESTING.md` hold throughout:

- Only an SPRT decides strength. Gain tests use [0, 10] at 3+0.03 with an 8,000-game cap.
  Non-regression checks use [−5, 0] at 10+0.1 with a 6,000-game cap.
- Changes that affect scaling get an LTC check.
- Every test is recorded, whether it passes or fails.

What must not change is in `docs/COMPATIBILITY.md`. Commands for the training PC are in
`docs/PC_TASKS.md` (sections 9-14). This round's VM measurements are under "Results so far" at the end.

| # | Line of work | What exists now | Next experiment | Where |
|---|---|---|---|---|
| 1 | Stronger teacher | `tools/nnue/rescore.py` (any UCI engine, resumable, equal-compute comparison); `nags_datagen --relabel-nodes` | teacher vs NAGS's own deeper labels at equal engine time, on the same games | PC (CPU days) |
| 2 | Quantization-aware training | `train.py --qat`; `quant_error.py` reports the export's loss | QAT fine-tune vs plain fine-tune, same data and settings | VM (small), PC (R9) |
| 3 | Data composition | `tools/nnue/mix.py` (report, build, book); `nags_datagen --book` | mixtures of retained and fresh data, with and without targeted books | PC |
| 4 | Search tuning for network 8 | `src/Tune.h` + `tools/spsa.py` (`exp/spsa-tune`, merged here) | SPSA in groups of 2-4 parameters, then SPRTs on other openings and at 10+0.1 | VM or PC (CPU) |
| 5 | The v4 architecture, properly | `train_v4.py`, engine support for format 4 | v3 and v4 on identical data: fixed-node quality and equal-time strength | PC |
| 6 | Richer cheap input features | design notes below | one feature family at a time, update/refresh cost first | later |
| 7 | Endgame supervision | `tools/nnue/tb_data.py` (generate, measure) | tablebase positions in the training mix; endgame accuracy tracked per material | PC + VM |
| 8 | Search mistakes, systematically | `tools/mistakes.py` (collect, summary, EPD) | a census on positions from real games; the largest class decides the next change | VM |
| 9 | The training bottleneck | `train.py --ft-impl fused` | the 3060's `--profile` for each path; adopt the faster one | PC (20 min) |
| 10 | Parallel search, measured | Threads=2: +107.5 ± 14.0 (time doubling +165.4) | nodes/s, time to depth and redundant work at 1/2/4 threads; equal-time SPRT at 4 | PC |
| 11 | Independent testing | `tools/ladder.py`, `tools/calibrate.py`, `tools/suite.py` | network 8 on the node ladder; higher rungs; other engines; more openings; longer controls | VM / PC |
| 12 | GNN / hybrid, conditional | the neural baseline protocol (`docs/TESTING.md`) | nothing until a network trained at scale passes its readiness step | - |

## 1. Learn from a stronger teacher

**Why.** Every network so far learned from NAGS's own 5000-node searches. More of those positions
teach more of the same judgement, including its errors. A stronger engine's scores are different
information. But a stronger teacher's labels cost more per position, so they have to beat NAGS's own
deeper labels at the same cost, not just its shallow ones.

**Tools.** `tools/nnue/rescore.py label` rescores whole games of indexed shards with any UCI engine at a
fixed limit, one engine process per worker. It writes a new shard with a consistent index and records
the engine time per position in `<out>.teacher.json`. `rescore.py compare` joins the base shard with
relabelled versions of it and reports four things for each label set:

- its fit to the game results: the sigmoid scale K and the error at that scale;
- the factor that puts it on the base's scale, to use as `--scale`;
- the mean difference from the base once scaled;
- its cost per position.

Limits of the records: they hold no castling rights, en-passant square or move counters. Castling is
inferred from kings and rooks on their home squares, which differs from the real rights in 3.8% of
positions (measured on 25,177 positions of 300 games). Positions the teacher scores as mate are dropped,
as `nags_datagen` drops them.

**Experiment** (`docs/PC_TASKS.md` §10). It uses a 10% sample of the R9 games (§6a): about 10M
positions with base labels from 5000 nodes. Relabel the same games twice:

- **T:** Stockfish (or another stronger engine) at a node count that takes about t ms a position on the
  PC.
- **N:** `nags_enhanced` at the node count that takes the same t, so engine time is equal. The VM's
  pilot gives the ratio (below).

Then:

1. Run `compare`, and use its factor as T's `--scale`.
2. Fine-tune network 8 three times with identical settings (`--init-from nets/nags.nnue`, 10 epochs,
   lr 1e-4): on the base labels (B), on T and on N.
3. Play SPRT [0, 10] at 3+0.03: T against N. If T passes, play T against network 8.

**Decision.** Teacher labels are expanded to more data only if T beats N (H1). Equal compute is the
point: if NAGS's own deeper search does as well, it is the cheaper way to get the same information, and
it keeps the engine's lineage its own. If T passes, the shard headers and sidecar files name the teacher
and its settings. Before such a network is released, check whether the rating lists NAGS may enter accept
networks trained on another engine's evaluations.

## 2. Quantization-aware training

**Why.** Network 8 loses 9.4 cp on average to the export, with a +5.8 cp bias. In output bucket 7 (the
opening) the loss is 21 cp. Nearly all of it is the rounding of the weights; the integer arithmetic adds
0.51 cp (`docs/PC_RESULTS.md` §3). The bias correction removed the bias and a third of the error and
still played −2.9 ± 10.1 (H0). So lower error alone is not enough, and only games decide.

**Tool.** `train.py --qat` rounds parts of the network to the export's integer grid in the forward pass,
with straight-through gradients:

- the feature-transformer rows with the king-bucket factor folded in, exactly as the export folds them;
- the feature biases;
- the output layer.

So the network trained is the one the engine runs. Its forward pass equals the exported weights in
floating point to 1e-4 cp, and the engine's integer evaluation to within its truncations (1 cp). Both
are tested in `tests/python/test_train_qat.py`. QAT is meant as a fine-tune (`--init-from`), with the
same data and settings as a plain fine-tune it is compared with. `quant_error.py` now also reports each
evaluation's loss against the labels, which compares two networks' exports on equal terms.

**Experiment.**

- VM, now: network 8 fine-tuned on 5.4M fresh positions, plain (P) and with `--qat` (Q), with identical
  settings otherwise. Measure the export's loss on a held-out evaluation set, then play SPRT [0, 10]
  Q against P at 3+0.03.
- PC (§11): the same on R9, as arm D next to §6's B (warm start, 20 epochs) and C (warm start, 10
  epochs).

**Decision.** `--qat` becomes the recommended setting for the last epochs of a training run only if
both hold:

1. its arm beats the plain arm by SPRT [0, 10] at 3+0.03;
2. the winner passes [−5, 0] at 10+0.1.

A lower export loss without a match result changes nothing.

## 3. Training-data composition

**Why.** Network 8 learned from 201.7M positions of one engine's self-play, all of it after 8 random
plies from the start position. A 200,000-position sample of fresh network-8 self-play is:

| Category | Share of positions |
|---|---|
| Pawn endings | 0.5% |
| Material two or more pawns apart | 35.8% |
| Level material with different pieces | 7.3% |
| Positions occurring more than once | 0.5% |

The data covers some phases and imbalances thinly. Near-duplicates are not the problem in this data,
though they would be in data from a narrow book.

**Tools.**

- `tools/nnue/mix.py report` gives the composition of a data set.
- `mix.py build` mixes sources in given shares, by whole games, keeps at most `--max-copies` records of
  any position, and writes the games in random order with an index.
- `mix.py book` draws start positions of a kind from existing data: imbalanced material, pawn endings,
  a piece range, and so on.
- `nags_datagen --book FILE` starts games from such positions. Without `--book` the output is
  byte-identical to before.

A book of 2,000 imbalanced positions gave games with material two or more pawns apart in 63.6% of
positions instead of 35.8%. Other targeted sources:

- the EPD of `tools/mistakes.py summary` (positions NAGS got wrong);
- `tools/make_openings.py --pgn` on human or correspondence games (unfamiliar openings);
- `tools/nnue/tb_data.py generate` (endgames, item 7).

**Experiment** (§12). Four mixtures of the same size, about 100M positions, each trained with the same
design and settings:

| Mixture | Contents |
|---|---|
| M0 | R9 only |
| M1 | 50% R9, 50% network 8's data |
| M2 | M1 with 10% replaced by targeted data: imbalanced and unfamiliar-opening books, mistake positions, tablebase endgames |
| M3 | M2 with `--max-copies 1` |

Compare evaluation-set losses by piece count and the endgame accuracy (item 7), then play SPRTs of M1
against M0, and of M2 against the better of the two.

**Decision.** A composition is adopted for the next round only by SPRT. A mixture that wins on loss but
not in games changes nothing.

## 4. Tune the search for the new network

**Why.** The search's margins were tuned by hand against networks 5-7, and network 8 evaluates
differently. `exp/spsa-tune` (merged into this branch) makes 13 constants UCI options in a
`-DNAGS_TUNE=ON` build. In a normal build they stay compile-time constants and the bench is unchanged.
`tools/spsa.py` runs fishtest's schedule.

**Experiment.** SPSA in small groups, so each group converges within its budget:

| Group | Parameters |
|---|---|
| A | LMR base and divisor, history divisor |
| B | reverse futility and futility margins |
| C | null-move reduction terms |
| D | SEE pruning margins |
| E | singular and double-extension margins |

Each group runs about 2,000 iterations of 4 pairs at 3+0.03, which is about 14 hours on the 4-core VM.
Then each tuned set must pass three tests:

1. SPRT [0, 10] against the defaults in the tuning build, with openings not used in tuning (another
   `--seed`, or a second book);
2. once its values are copied into `src/Tune.h`, the normal-build SPRT [0, 10] at 3+0.03;
3. [−5, 0] at 10+0.1.

**Decision.** A group's values are kept only if all three pass. A group that fails is not re-tuned
without a new hypothesis.

## 5. Test the v4 architecture properly

**Why.** Format 4 (a 16-32 hidden stack after the accumulators) costs 21-26% of nodes per second with
untrained networks. Trained networks are sparser, so faster. The 512-wide v3 network fitted better and
lost 48 Elo to its speed. A v4 network has to evaluate well enough to pay for its speed, and only a
like-for-like comparison shows whether it does.

**Experiment** (§13). Train `train.py` (v3, network 8's design) and `train_v4.py` (pairwise, 16-32) on
the same R9 data, with the same epochs, batch and learning rate, both from scratch. Then:

1. fixed-node SPRT [0, 10] at 20,000 nodes: the quality of the evaluation alone;
2. equal-time SPRT [0, 10] at 3+0.03: whether it pays for its cost;
3. if step 2 passes, [−5, 0] at 10+0.1.

`train_v4.py` has no `--eval-set`, so the comparison is by matches.

**Decision.** v4 becomes the default design only if step 2 passes. If step 1 passes and step 2 fails, the
next work is v4's inference speed, not more v4 training.

## 6. Richer, inexpensive input features

**Why.** The 768-feature (×4 king buckets) input knows where each piece is but not how pieces relate.
Engines that added threat inputs (which pieces attack which) or pawn-structure inputs gained from them.
Each such feature costs incremental-update time, and that cost is paid at every node.

**Plan, not started.** Work on one feature family at a time, behind a new format version, so formats 1-4
are untouched:

1. Prototype the features in the trainer only, and measure the loss gain on R9 at equal epochs.
2. Implement the incremental update. Measure `speedtest` (update, refresh) and bench nodes per second
   before any games.
3. Run the equal-time SPRT as in item 5.

Candidates, cheapest first:

- passed and isolated pawns as extra features: these change only on pawn moves and captures;
- "piece attacked by a lower-valued piece" bits: these change on most moves.

This waits for items 1-3: a new input needs a new network trained at scale anyway, so it should be
trained on the better data.

## 7. Endgame supervision

**Why.** `tools/nnue/tb_data.py measure` asks, per material, whether NAGS's score at 10,000 nodes
(without tablebases) says the tablebase's result. A score beyond ±100 cp counts as a win or loss,
within it a draw. Network 8 is right in 75.0% of 2,100 random quiet 3-4 piece positions. Its errors are
concentrated in draws with a material advantage:

| Material | Right | Errors |
|---|---:|---|
| KQvKR, KRPvK, KBNvK | 100% | |
| KPvK, KPvKP, KBPvK, KNPvK, KRvKP, KQvKP | 95-99% | |
| KNvKP | 73% | mostly draws scored as wins or losses |
| KBvKP | 68% | mostly draws scored as wins or losses |
| KRvKN | 18% | draws scored as wins or losses |
| KRvKB | 8% | draws scored as wins or losses |
| KNNvK | 1% | all draws scored as wins or losses |

These are the positions where a material count misleads. The training data rarely contains them, and
its result labels come from adjudicated self-play.

**Tool.** `tb_data.py generate` writes random legal quiet positions of chosen materials. Each has the
tablebase result under the fifty-move rule (cursed wins and blessed losses are draws, as at a halfmove
clock of 0) and NAGS's search score without tablebases, as 32-byte records ready for `mix.py build`.

**Experiment.**

1. Measure on 5-piece materials on the PC, with the 3-4-5 tables, to choose the materials that matter
   (KRPvKR, KRvKB with pawns, KBPvKB, KNPvKB and others).
2. Add 1-2% tablebase positions to mixture M2 (item 3).
3. Track `measure` per material for every new network, next to the evaluation set's loss by piece
   count.
4. In parallel, as an evaluation experiment, the UCI option `DrawScaling` (off by default, so the bench
   is unchanged: 1528850; with it on, 1548643). It scales the network's score towards zero when the side
   it favours has no pawns and at most a minor piece more: by 4/64 if the other side has at most a minor
   piece, otherwise by 14/64, and to 0 for a lone minor piece. It also scores two knights against a bare
   king as 0. On the same 2,100 positions with `measure` it raises the share right from 75.0% to
   **97.5%**:

   | Material | Off | On |
   |---|---:|---:|
   | KRvKB | 8% | 100% |
   | KRvKN | 18% | 94% (9 wins now called draws) |
   | KNNvK | 1% | 100% |
   | KBvKP | 68% | 97% |
   | KNvKP | 73% | 93% |
   | the others | unchanged within ±2 positions | |

   The SPRT, on against off in the same binary, [0, 10] at 3+0.03, decides. These endings are rare in
   games, so a small true gain may end undecided at the cap.

**Decision.** By SPRT, as always. `measure` is a regression check and a diagnostic, not a strength
claim.

## 8. Find search mistakes systematically

**Why.** Self-play shows that a change helps, not where NAGS goes wrong. A census of mistakes against a
deeper reference shows where its decisions fail.

**Tool.** `tools/mistakes.py collect`. NAGS chooses a move with N nodes. A deeper reference scores that
move and its own best move, each searched alone with `searchmoves`. A loss of at least `--threshold` is a
mistake, and each mistake is classified by what fixes it:

| Class | What fixes it |
|---|---|
| time | more nodes (4N, 16N) find a good move |
| selectivity | the reference's move, searched alone with the same N nodes, scores above NAGS's choice (ordering, pruning or reductions kept it from being searched as deeply) |
| evaluation | neither: NAGS misjudges where the move leads |

`summary` breaks the classes down by phase and by kind of move, and writes the mistakes as an EPD suite
for `tools/suite.py run`.

**Experiment.** A census on positions from real games: PGNs of 10+0.1 matches against the ladder, N =
40,000 nodes, Stockfish 16 at 2M nodes as the reference, `--ref-scale` from item 1's fit. The largest
class decides what to work on next:

- time: the soft limit (`tools/decision_data.py` measures allocations);
- selectivity: the pruning or reduction rule involved, tested with `NAGS_STATS` counters and SPRT;
- evaluation: the mistake positions go into the data (item 3) and the teacher labels (item 1).

## 9. Optimise the measured training bottleneck

**Why.** On the 3060, the backward pass is 68% of a kb4 batch (22.45 of 32.94 ms). Reading the loss
less often saved 0.9%. The `embedding` path looks up 64 rows per perspective: 32 piece slots, padding
included, plus their 32 factor rows. Its gradient is scattered over all of them, twice per position.

**Tool.** `train.py --ft-impl fused` folds the factor rows into the bucket rows once per batch, as the
export does. It then makes one `embedding_bag` call over both perspectives' pieces only. It computes the
same outputs and gradients to float rounding, with the same parameters and checkpoints, so a run can
switch paths on `--resume`. On the 4-core VM (kb4 + SCReLU, batch 16384): 28,400 → 40,400
positions/s; backward 301 → 193 ms a batch; forward 184 → 104 ms. QAT adds about 5%.

**Experiment** (§9). `--profile 50` on the 3060 with each path, then one epoch each with the same seed:
compare the seconds and the validation loss.

**Decision.** If the fused path is at least 1.2× faster per epoch and its validation loss after one epoch
is within 0.5% of the embedding path's, it becomes `run_round.py`'s default. Precision experiments
(bf16) come after, and only for the output layer: the accumulators sum about 30 weights at a resolution
of 1/255, more than bf16's 8-bit mantissa can hold.

## 10. Improve parallel search after measuring it

**Why.** A second thread is worth about 65% of a time doubling (+107.5 against +165.4). Lazy SMP loses to
redundant work and contention in the shared table, and aggregate nodes per second overstates its value.

**Experiment** (PC §8, then §14).

1. Measure at 1, 2 and 4 threads on 20 positions: nodes per second, time to depth 20, and the fraction
   of TT probes that hit an entry written by another thread (a `NAGS_STATS` counter to add).
2. Play the equal-time SPRT of 4 threads against 1, next to the time-quadrupling match.

A change (for example, helper threads starting at staggered depths, or skipping depths another thread
already searches) is judged by the 4-thread equal-time SPRT, never by nodes per second.

## 11. Expand independent strength testing

**Why.** Self-play gains carried over to Stockfish at a third to all of their size (`docs/TESTING.md`).
The `UCI_Elo` scale ends at 3190, where NAGS now scores about 45%.

**Steps.**

1. Refresh the benchmark: rate network 8 on the node ladder (`ladder.py rate --nodes 256000` against
   sf16-8k/16k/32k). This is the like-for-like comparison with network 7's −13 ± 37.
2. Anchor the higher rungs (64k-512k) and rate at 10+0.1 against the rungs near NAGS.
3. Add unrestricted opponents with published CCRL ratings as rungs (any UCI engine can be one), with a
   broader opening set (`make_openings.py --pgn` on correspondence games) and 60+0.6 for the final check
   of a round.

**Schedule.** Steps 1-2 for every network that replaces `nets/nags.nnue`, and step 3 once per round.

## 12. Keep the GNN/hybrid conditional

The neural baseline protocol in `docs/TESTING.md` stands unchanged. No work on the hybrid's search until
a GNN trained at scale passes readiness: its MCTS top move matches the reference within 10 points of
alpha-beta's rate. Even then, an arm must pass the [0, 10] confirmation against alpha-beta with the same
CPU threads. The work in items 1-11 improves the alpha-beta engine that any hybrid would have to beat.

## Results so far

Measured on the 4-core VM (Intel Xeon, 2.1 GHz, no GPU) on 2026-10-30, with network 8 on this branch
(bench 1528850).

- **Node ladder (item 11, step 1):** network 8 at 256,000 nodes a move rates **172 ± 36** on the sf16-nodes
  scale (100 games a rung):

  | Rung | Result for NAGS | Elo |
  |---|---|---:|
  | sf16-8k | +59 =30 −11 | +182 ± 60 |
  | sf16-16k | +26 =47 −27 | −3 ± 48 |
  | sf16-32k | +10 =36 −54 | −164 ± 49 |

  `989184c` with network 7 rated −13 ± 37. So NAGS now needs about **16 times** Stockfish 16's nodes for the
  same results, not 32: a doubling of efficiency, probably nearly all of it network 8 (the search changes since are
  correctness fixes). The 8k rung is now too weak to
  rate against. The next ladder needs the 64k-512k rungs anchored (step 2).
- **Endgames (item 7):** 75.0% of 2,100 tablebase positions right (table above).
- **Data composition (item 3):** the network-8 self-play composition and the imbalanced book's effect
  (above).
- **Training speed (item 9):** the CPU profile above. With the CPU otherwise idle, a fused epoch over 5.46M
  positions took 62-63 s (about 87,000 positions/s); with `--qat`, 62-69 s.
- **QAT (item 2):**
  - Setup: network 8 fine-tuned twice on the same 5.46M fresh positions (seeds 9201-9202, 30,000 games each
    by the network-8 engine at 5000 nodes). Both runs: 2 epochs, lr 1e-4 cosine, `--ft-impl fused`,
    `--init-from nets/nags.nnue`. P plain, Q with `--qat`.
  - Loss against the labels (λ 0.75) on 100,000 positions of the evaluation set (seed 9299, 1,500 games, not
    trained on), as the engine computes each export:

    | Network | Export (engine) | Float network |
    |---|---:|---:|
    | network 8 | 0.009317 | - |
    | P | 0.008757 | 0.008699 |
    | Q | 0.008716 | 0.008718 (rounded by design) |

  - Rounding costs P 12.6 cp on average (bias +3.5) and 0.000058 of loss. Q's export recovers 70% of that
    loss. Q's own weight-rounding error is 0 by construction; only the 0.51 cp of integer arithmetic is left.
  - Both fine-tunes have a lower loss than network 8 on positions like their training data. That is no
    strength claim: an evaluation set from the same engine and settings favours them. The SPRT of Q against
    P decides QAT.
