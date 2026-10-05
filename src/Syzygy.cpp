#include "Syzygy.h"

#include "tbprobe.h"

#include <algorithm>

namespace syzygy {

namespace {
int probeLimit = 7;

void updateCardinality() { detail::cardinality = std::min(static_cast<int>(TB_LARGEST), probeLimit); }

Wdl toWdl(unsigned v) {
    switch (v) {
    case TB_LOSS: return Wdl::Loss;
    case TB_BLESSED_LOSS: return Wdl::BlessedLoss;
    case TB_CURSED_WIN: return Wdl::CursedWin;
    case TB_WIN: return Wdl::Win;
    default: return Wdl::Draw;
    }
}
} // namespace

int init(const std::string &path) {
    if (!tb_init(path.c_str())) TB_LARGEST = 0;
    updateCardinality();
    return static_cast<int>(TB_LARGEST);
}

int largest() { return static_cast<int>(TB_LARGEST); }

void setProbeLimit(int pieces) {
    probeLimit = std::clamp(pieces, 0, 7);
    updateCardinality();
}

namespace {
// Fathom must not be asked about positions larger than its tables (after
// unloading, its lookup structures still point at freed entries).
bool covered(const Position &p) { return popcount(p.white | p.black) <= static_cast<int>(TB_LARGEST); }
} // namespace

bool probeWdl(const Position &p, Wdl &wdl) {
    if (!covered(p)) return false;
    unsigned r = tb_probe_wdl(p.white, p.black, p.kings, p.queens, p.rooks, p.bishops, p.knights, p.pawns, p.rule50,
                              p.castling, p.ep, p.whiteToMove);
    if (r == TB_RESULT_FAILED) return false;
    wdl = toWdl(r);
    return true;
}

bool probeRoot(const Position &p, RootResult &out) {
    if (!covered(p)) return false;
    unsigned r = tb_probe_root(p.white, p.black, p.kings, p.queens, p.rooks, p.bishops, p.knights, p.pawns, p.rule50,
                               p.castling, p.ep, p.whiteToMove, nullptr);
    // Checkmate and stalemate need no tablebase; the search handles them.
    if (r == TB_RESULT_FAILED || r == TB_RESULT_CHECKMATE || r == TB_RESULT_STALEMATE) return false;
    out.from = static_cast<int>(TB_GET_FROM(r));
    out.to = static_cast<int>(TB_GET_TO(r));
    switch (TB_GET_PROMOTES(r)) {
    case TB_PROMOTES_QUEEN: out.promotion = QUEEN; break;
    case TB_PROMOTES_ROOK: out.promotion = ROOK; break;
    case TB_PROMOTES_BISHOP: out.promotion = BISHOP; break;
    case TB_PROMOTES_KNIGHT: out.promotion = KNIGHT; break;
    default: out.promotion = -1; break;
    }
    out.wdl = toWdl(TB_GET_WDL(r));
    out.dtz = TB_GET_DTZ(r);
    return true;
}

} // namespace syzygy
