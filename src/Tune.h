#pragma once

// Search parameters that tools/spsa.py can tune. In a normal build each one is
// a compile-time constant, so the search (and its bench) is exactly what the
// literal would give. Built with -DNAGS_TUNE=ON, each becomes a variable that
// the engines expose as a UCI spin option of the same name (tuning builds only:
// their search is slower and must not be used for anything else).

#include <vector>

namespace tune {

struct Param {
    const char *name;
    int *value;
    int def, min, max;
};

#ifdef NAGS_TUNE
inline std::vector<Param> &params() {
    static std::vector<Param> p;
    return p;
}
struct Registrar {
    Registrar(const char *name, int *value, int def, int lo, int hi) { params().push_back({name, value, def, lo, hi}); }
};
#define NAGS_TUNABLE(name, def, lo, hi) \
    inline int name = def;              \
    inline Registrar name##Registrar{#name, &name, def, lo, hi};
#else
inline std::vector<Param> &params() {
    static std::vector<Param> p;
    return p;
}
#define NAGS_TUNABLE(name, def, lo, hi) inline constexpr int name = def;
#endif

// The parameters (default, range). Change a default only with an SPRT.
NAGS_TUNABLE(AspirationWindow, 25, 8, 80)       // first aspiration half-width (cp)
NAGS_TUNABLE(RfpMargin, 80, 30, 200)            // reverse futility: eval - margin * depth >= beta
NAGS_TUNABLE(NmpBase, 3, 1, 5)                  // null-move reduction: base + depth / divisor
NAGS_TUNABLE(NmpDivisor, 6, 3, 12)
NAGS_TUNABLE(FutilityMargin, 120, 50, 300)      // futility: eval + margin * depth <= alpha
NAGS_TUNABLE(HistoryPruneMargin, 4096, 1024, 12288) // history pruning: history < -margin * depth
NAGS_TUNABLE(SeeQuietMargin, 60, 20, 150)       // SEE pruning of quiet moves: see < -margin * depth
NAGS_TUNABLE(SeeCaptureMargin, 20, 5, 60)       // ... of captures: see < -margin * depth^2
NAGS_TUNABLE(SingularMargin, 2, 1, 5)           // singular beta: tt score - margin * depth
NAGS_TUNABLE(DoubleExtensionMargin, 20, 5, 60)  // two plies when below singular beta by this much
NAGS_TUNABLE(LmrBase, 75, 0, 150)               // reduction = base / 100 + ln d ln m / (divisor / 100)
NAGS_TUNABLE(LmrDivisor, 225, 150, 400)
NAGS_TUNABLE(LmrHistoryDivisor, 8192, 2048, 32768) // reduction - history / divisor

} // namespace tune
