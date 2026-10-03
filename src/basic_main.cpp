// nags_basic: alpha-beta search on the ray-based Board, no hash table.

#include "Board.h"
#include "ClassicEngine.h"

int main(int argc, char **argv) {
    ClassicEngine<Board> engine("NAGS Basic", /*useHashTable=*/false);
    engine.setDefaultBenchDepth(6);
    return uci::run(engine, argc, argv);
}
