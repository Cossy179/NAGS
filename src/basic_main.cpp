// nags_basic: alpha-beta search on the ray-based Board, no hash table.

#include "Board.h"
#include "ClassicEngine.h"

int main() {
    ClassicEngine<Board> engine("NAGS Basic", /*useHashTable=*/false);
    return uci::run(engine);
}
