#pragma once

// Client for the meta-learner service (meta_learner.py). Requests are
// newline-delimited JSON over one persistent TCP connection. All calls are
// bounded by short timeouts, and after a failed connection attempt the
// client backs off for a while so a missing server costs almost nothing.

#include "Net.h"

#include <chrono>
#include <string>

struct MetaDeltas {
    float dfs_depth_delta = 0.0f;          // -1 .. +1
    float mcts_budget_delta = 0.0f;        // -1 .. +1
    float bandit_exploration_delta = 0.0f; // -1 .. +1
};

class MetaClient {
public:
    MetaClient(std::string host = "127.0.0.1", int port = 5556);

    void configure(const std::string &host, int port);

    // Returns false (and leaves `out` at zero deltas) if the service is unavailable.
    bool predict(const std::string &fen, int time_left_ms, float last_uncertainty, float tactical_shot_ratio,
                 MetaDeltas &out);

    // Adds a training sample. `reward` is whatever outcome signal the caller
    // has (the training pipeline uses game results).
    bool add_sample(const std::string &fen, int time_left_ms, float last_uncertainty, float tactical_shot_ratio,
                    const MetaDeltas &chosen_deltas, float reward);

    bool train(int steps = 100);

    bool is_connected() const { return socket.isOpen(); }

private:
    bool roundTrip(const std::string &request, std::string &response);

    std::string host;
    int port;
    LineSocket socket;
    std::chrono::steady_clock::time_point retryAfter{};
};
