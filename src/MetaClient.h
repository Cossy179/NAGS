#pragma once

#include <string>
#include <chrono>

struct MetaDeltas {
    float dfs_depth_delta = 0.0f;      // -1 to +1
    float mcts_budget_delta = 0.0f;    // -1 to +1  
    float bandit_exploration_delta = 0.0f; // -1 to +1
};

class MetaClient {
public:
    MetaClient(const std::string& host = "127.0.0.1", int port = 5556);
    ~MetaClient();
    
    // Get hyperparameter adjustments for current position
    MetaDeltas predict(const std::string& fen, int time_left_ms, 
                      float last_uncertainty = 0.1f, float tactical_shot_ratio = 0.2f);
    
    // Add training sample from completed search
    bool add_sample(const std::string& fen, int time_left_ms, float last_uncertainty,
                   float tactical_shot_ratio, const MetaDeltas& chosen_deltas, 
                   float elo_gain_per_sec);
    
    // Trigger training on server
    bool train(int steps = 100);
    
    bool is_connected() const { return connected; }
    
private:
    std::string host;
    int port;
    bool connected = false;
    
    // Socket management
    int create_connection();
    void close_connection();
    std::string send_request(const std::string& json_request);
    
    // Current connection (reused)
    int sock = -1;
};
