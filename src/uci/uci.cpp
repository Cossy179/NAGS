#include "uci.h"
#include "../core/engine.h"

#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

namespace nags {

UCI::UCI(Engine* engine)
    : engine_(engine), quit_(false) {
    // Initialize default options
    options_["Hash"] = "128";
    options_["Threads"] = "1";
    options_["GNN Server"] = "localhost:50051";
}

UCI::~UCI() {
    handleQuit();
}

void UCI::loop() {
    std::string line;
    
    while (!quit_ && std::getline(std::cin, line)) {
        auto tokens = tokenize(line);
        
        if (tokens.empty()) {
            continue;
        }
        
        const std::string& command = tokens[0];
        
        if (command == "uci") {
            handleUCI();
        } else if (command == "isready") {
            handleIsReady();
        } else if (command == "ucinewgame") {
            handleUCINewGame();
        } else if (command == "position") {
            handlePosition(tokens);
        } else if (command == "go") {
            handleGo(tokens);
        } else if (command == "stop") {
            handleStop();
        } else if (command == "quit") {
            handleQuit();
            break;
        } else if (command == "setoption") {
            handleSetOption(tokens);
        } else if (command == "d" || command == "display") {
            // Debug command: display current board
            std::cout << engine_->getBoardString() << std::endl;
        }
    }
}

void UCI::handleUCI() {
    std::cout << "id name NAGS v1.0" << std::endl;
    std::cout << "id author NAGS Team" << std::endl;
    
    // Send options
    std::cout << "option name Hash type spin default 128 min 1 max 65536" << std::endl;
    std::cout << "option name Threads type spin default 1 min 1 max 256" << std::endl;
    std::cout << "option name GNN Server type string default localhost:50051" << std::endl;
    
    std::cout << "uciok" << std::endl;
}

void UCI::handleIsReady() {
    // TODO: Implement proper GNN model loading check
    std::cout << "readyok" << std::endl;
}

void UCI::handleUCINewGame() {
    engine_->newGame();
}

void UCI::handlePosition(const std::vector<std::string>& tokens) {
    if (tokens.size() < 2) {
        return;
    }
    
    size_t movesIndex = 0;
    std::string fen;
    
    if (tokens[1] == "startpos") {
        fen = "startpos";
        movesIndex = 2;
    } else if (tokens[1] == "fen") {
        // Collect FEN string (may contain spaces)
        fen = "";
        size_t i = 2;
        while (i < tokens.size() && tokens[i] != "moves") {
            if (!fen.empty()) {
                fen += " ";
            }
            fen += tokens[i++];
        }
        movesIndex = i;
    }
    
    // Process moves if they exist
    std::vector<std::string> moves;
    if (movesIndex < tokens.size() && tokens[movesIndex] == "moves") {
        for (size_t i = movesIndex + 1; i < tokens.size(); ++i) {
            moves.push_back(tokens[i]);
        }
    }
    
    engine_->setPosition(fen, moves);
}

void UCI::handleGo(const std::vector<std::string>& tokens) {
    SearchOptions options;
    
    for (size_t i = 1; i < tokens.size(); i += 2) {
        if (tokens[i] == "searchmoves") {
            // Skip move list for now
            while (i + 1 < tokens.size() && !(tokens[i + 1] == "ponder" || 
                   tokens[i + 1] == "wtime" || tokens[i + 1] == "btime" || 
                   tokens[i + 1] == "winc" || tokens[i + 1] == "binc" || 
                   tokens[i + 1] == "movestogo" || tokens[i + 1] == "depth" || 
                   tokens[i + 1] == "nodes" || tokens[i + 1] == "mate" || 
                   tokens[i + 1] == "movetime" || tokens[i + 1] == "infinite")) {
                i++;
            }
        } else if (tokens[i] == "ponder") {
            options.ponder = true;
            i--; // No value follows
        } else if (tokens[i] == "infinite") {
            options.infinite = true;
            i--; // No value follows
        } else if (i + 1 < tokens.size()) {
            if (tokens[i] == "wtime") {
                options.wtime = std::chrono::milliseconds(std::stoi(tokens[i + 1]));
            } else if (tokens[i] == "btime") {
                options.btime = std::chrono::milliseconds(std::stoi(tokens[i + 1]));
            } else if (tokens[i] == "winc") {
                options.winc = std::chrono::milliseconds(std::stoi(tokens[i + 1]));
            } else if (tokens[i] == "binc") {
                options.binc = std::chrono::milliseconds(std::stoi(tokens[i + 1]));
            } else if (tokens[i] == "movestogo") {
                options.movestogo = std::stoi(tokens[i + 1]);
            } else if (tokens[i] == "depth") {
                options.depth = std::stoi(tokens[i + 1]);
            } else if (tokens[i] == "nodes") {
                options.nodes = std::stoi(tokens[i + 1]);
            } else if (tokens[i] == "movetime") {
                options.moveTime = std::chrono::milliseconds(std::stoi(tokens[i + 1]));
            }
        }
    }
    
    // Start search in a separate thread
    std::thread searchThread([this, options]() {
        SearchResult result = engine_->startSearch(options);
        
        // Output bestmove
        std::cout << "bestmove " << result.bestMove.toUCI();
        
        // If we have a ponder move (first move in PV after bestmove)
        if (options.ponder && result.principalVariation.size() > 1) {
            std::cout << " ponder " << result.principalVariation[1].toUCI();
        }
        
        std::cout << std::endl;
    });
    
    searchThread.detach();
}

void UCI::handleStop() {
    engine_->stopSearch();
}

void UCI::handleQuit() {
    engine_->stopSearch();
    quit_ = true;
}

void UCI::handleSetOption(const std::vector<std::string>& tokens) {
    // Format: setoption name <name> value <value>
    if (tokens.size() < 4 || tokens[1] != "name") {
        return;
    }
    
    std::string name = tokens[2];
    std::string value;
    
    // Find "value" token
    size_t valueIndex = 3;
    while (valueIndex < tokens.size() && tokens[valueIndex] != "value") {
        name += " " + tokens[valueIndex++];
    }
    
    // Extract value
    if (valueIndex + 1 < tokens.size()) {
        value = tokens[valueIndex + 1];
        for (size_t i = valueIndex + 2; i < tokens.size(); ++i) {
            value += " " + tokens[i];
        }
    }
    
    // Store option
    options_[name] = value;
    
    // Apply option if needed
    if (name == "Hash") {
        // TODO: Implement hash table resizing
    } else if (name == "Threads") {
        // TODO: Implement thread count adjustment
    } else if (name == "GNN Server") {
        // TODO: Implement GNN server connection
    }
}

std::vector<std::string> UCI::tokenize(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

void UCI::sendInfo(const std::string& info) {
    std::cout << "info string " << info << std::endl;
}

} // namespace nags 