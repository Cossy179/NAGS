#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace nags {

// Forward declarations
class Engine;

class UCI {
public:
    UCI(Engine* engine);
    ~UCI();
    
    // Main UCI loop
    void loop();
    
private:
    // Command handlers
    void handleUCI();
    void handleIsReady();
    void handleUCINewGame();
    void handlePosition(const std::vector<std::string>& tokens);
    void handleGo(const std::vector<std::string>& tokens);
    void handleStop();
    void handleQuit();
    void handleSetOption(const std::vector<std::string>& tokens);
    
    // Helper methods
    std::vector<std::string> tokenize(const std::string& input);
    void sendInfo(const std::string& info);
    
    // Engine reference
    Engine* engine_;
    
    // Options storage
    std::unordered_map<std::string, std::string> options_;
    
    // Control variables
    bool quit_;
};

} // namespace nags 