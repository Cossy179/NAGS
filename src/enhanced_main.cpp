// nags_enhanced: alpha-beta search on FastBoard with a shared transposition
// table and Lazy SMP helper threads (UCI options Hash / Threads).

#include "ClassicEngine.h"
#include "FastBoard.h"

int main() {
    ClassicEngine<FastBoard> engine("NAGS Enhanced", /*useHashTable=*/true);
    return uci::run(engine);
}
