#include <iostream>
#include <string>
#include <thread>
#include <memory>

#include "uci/uci.h"
#include "core/engine.h"

int main(int argc, char* argv[]) {
    std::cout << "NAGS - Neuro-Adaptive Graph Search Chess Engine v1.0" << std::endl;
    
    // Initialize the engine
    std::unique_ptr<nags::Engine> engine = std::make_unique<nags::Engine>();
    
    // Create UCI interface
    nags::UCI uci(engine.get());
    
    // Start UCI loop
    uci.loop();
    
    return 0;
} 