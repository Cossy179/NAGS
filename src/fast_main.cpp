// nags_fast: alpha-beta search on the magic-bitboard FastBoard, no hash table.

#include "ClassicEngine.h"
#include "FastBoard.h"

int main() {
    ClassicEngine<FastBoard> engine("NAGS Fast", /*useHashTable=*/false);
    return uci::run(engine);
}
