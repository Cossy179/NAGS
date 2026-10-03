// nags_fast: alpha-beta search on the magic-bitboard FastBoard, no hash table.

#include "ClassicEngine.h"
#include "FastBoard.h"

int main(int argc, char **argv) {
    ClassicEngine<FastBoard> engine("NAGS Fast", /*useHashTable=*/false);
    engine.setDefaultBenchDepth(6);
    return uci::run(engine, argc, argv);
}
