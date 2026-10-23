#pragma once

// Experimental (UCI option TimeModel, off by default): the soft time limit's
// stability and score-drop factors replaced by a logistic model of
// P(searching twice as long changes the move), fitted by
// tools/decision_data.py on positions of real games (docs/TESTING.md,
// "Decisions"). The best move's node share still scales the result.

#include <algorithm>
#include <cmath>

namespace timemodel {

// What the search knows at the end of an iteration, as the measurement
// records it (tools/decision_data.py search_features): scores capped at
// kScoreCap, mates included.
struct Features {
    int depth = 0;
    int bestChanges = 0;      // best-move changes over all iterations so far
    int stableIterations = 0; // completed iterations since the best move last changed
    int scoreAbs = 0;
    int scoreDrop = 0;        // previous iteration's score minus this one's, if positive
    int scoreSwing = 0;       // highest minus lowest of the last three iterations' scores
    int legalMoves = 0;
    int pieces = 0;
    bool inCheck = false;
};

constexpr int kScoreCap = 2000;

// z = kWeights[0] + sum kWeights[i] * feature i, in the order of Features.
// Fitted 2026-10-23 on 1,026 positions of 3+0.03 games between nags_enhanced
// builds (4,104 decisions at 40k to 640k nodes; twice the nodes changed the
// move in 24.7%), depth capped at 30 and stable iterations at 20 as here:
//   decision_data.py fit --features depth,best_changes,stable_iterations,score_abs,
//       score_drop,score_swing,legal_moves,in_check,pieces
// Held-out log-loss 0.517 (base rate 0.583), AUC 0.734.
constexpr double kWeights[10] = {-0.718578,   0.148724,   0.0717681, -0.22216,  -0.000397416,
                                 0.0109518,   -0.00668061, -0.0093152, -0.410399, -0.036346};
constexpr int kDepthCap = 30;
constexpr int kStableCap = 20;
// The scale for a change probability p: (p / kReference)^kExponent, kept in
// [kMinScale, kMaxScale]. On the measured decisions it has the old rule's
// ratio between "best move just changed" and "stable 4+ iterations" (2.0 /
// 0.85) and its mean (1.08).
constexpr double kReference = 0.203961;
constexpr double kExponent = 0.67856;
constexpr double kMinScale = 0.4;
constexpr double kMaxScale = 3.0;

inline double changeProbability(const Features &f) {
    const double x[9] = {static_cast<double>(std::min(f.depth, kDepthCap)),
                         static_cast<double>(f.bestChanges),
                         static_cast<double>(std::min(f.stableIterations, kStableCap)),
                         static_cast<double>(std::min(f.scoreAbs, kScoreCap)),
                         static_cast<double>(std::min(f.scoreDrop, 2 * kScoreCap)),
                         static_cast<double>(std::min(f.scoreSwing, 2 * kScoreCap)),
                         static_cast<double>(f.legalMoves),
                         f.inCheck ? 1.0 : 0.0,
                         static_cast<double>(f.pieces)};
    double z = kWeights[0];
    for (int i = 0; i < 9; ++i) z += kWeights[i + 1] * x[i];
    return 1.0 / (1.0 + std::exp(-std::clamp(z, -50.0, 50.0)));
}

inline double softScale(const Features &f) {
    return std::clamp(std::pow(changeProbability(f) / kReference, kExponent), kMinScale, kMaxScale);
}

} // namespace timemodel
