#include "Board.h"
#include "Search.h"
#include "NAGS.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string &s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Board board;

    std::mt19937 rng(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));

    std::string line;
    int hashMB = 16;
    int numThreads = 1;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto tokens = split(line);
        if (tokens.empty()) continue;
        const std::string &cmd = tokens[0];

        if (cmd == "uci") {
            std::cout << "id name NAGS\n";
            std::cout << "id author Alex\n";
            std::cout << "option name Hash type spin default 16 min 1 max 4096\n";
            std::cout << "option name Threads type spin default 1 min 1 max 64\n";
            std::cout << "option name Clear Hash type button\n";
            std::cout << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "ucinewgame") {
            board.setStartPos();
        } else if (cmd == "position") {
            // position [fen <fen> | startpos ]  moves ...
            size_t i = 1;
            if (i < tokens.size() && tokens[i] == "startpos") {
                board.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                i++;
            } else if (i < tokens.size() && tokens[i] == "fen") {
                std::string fen;
                // FEN is 6 fields; gather until we have 6 fields collected
                int fields = 0;
                i++;
                for (; i < tokens.size() && fields < 6; ++i) {
                    if (!fen.empty()) fen += ' ';
                    fen += tokens[i];
                    fields++;
                }
                board.setFromFEN(fen);
            }
            if (i < tokens.size() && tokens[i] == "moves") {
                ++i;
                std::vector<std::string> moves;
                for (; i < tokens.size(); ++i) moves.push_back(tokens[i]);
                board.applyMovesUCI(moves);
            }
        } else if (cmd == "setoption") {
            // setoption name <Name> value <Value>
            // Simple parser
            std::string name, value;
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "name") {
                    name.clear();
                    ++i; while (i < tokens.size() && tokens[i] != "value") { if (!name.empty()) name += ' '; name += tokens[i]; ++i; }
                    if (i < tokens.size() && tokens[i] == "value") { ++i; }
                    value.clear();
                    while (i < tokens.size()) { if (!value.empty()) value += ' '; value += tokens[i]; ++i; }
                    break;
                }
            }
            if (!name.empty()) {
                if (name == "Hash") {
                    try { hashMB = std::max(1, std::min(4096, std::stoi(value))); } catch (...) {}
                } else if (name == "Threads") {
                    try { numThreads = std::max(1, std::min(64, std::stoi(value))); } catch (...) {}
                } else if (name == "Clear Hash") {
                    // Will clear on next Search construction
                }
            }
        } else if (cmd == "go") {
            // Parse time controls and depth
            SearchLimits limits;
            for (size_t i = 1; i < tokens.size(); ++i) {
                const std::string &t = tokens[i];
                if (t == "wtime" && i + 1 < tokens.size()) {
                    long long w = std::stoll(tokens[++i]);
                    if (board.sideToMove() == Color::White) limits.timeMs = w;
                } else if (t == "btime" && i + 1 < tokens.size()) {
                    long long btm = std::stoll(tokens[++i]);
                    if (board.sideToMove() == Color::Black) limits.timeMs = btm;
                } else if (t == "movestogo" && i + 1 < tokens.size()) {
                    limits.movesToGo = static_cast<int>(std::stoll(tokens[++i]));
                } else if (t == "depth" && i + 1 < tokens.size()) {
                    limits.depth = static_cast<int>(std::stoll(tokens[++i]));
                }
            }
            
            // Use NAGS hybrid controller
            NAGSController nags(board);
            Move bestMove = nags.search(limits);
            
            if (bestMove.from == -1) {
                std::cout << "bestmove 0000\n" << std::flush;
            } else {
                std::cout << "bestmove " << Board::moveToUci(bestMove) << "\n" << std::flush;
            }
        } else if (cmd == "stop") {
            // No search implemented; ignore
        } else if (cmd == "quit") {
            break;
        } else if (cmd == "d") {
            std::cout << board.getFEN() << "\n";
        }
    }
    return 0;
}


