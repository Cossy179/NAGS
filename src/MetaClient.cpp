#include "MetaClient.h"

#include <sstream>

namespace {
constexpr int kConnectTimeoutMs = 150;
constexpr int kRequestTimeoutMs = 500;
constexpr int kBackoffSeconds = 30;

std::string features(const std::string &fen, int time_left_ms, float last_uncertainty, float tactical_shot_ratio) {
    std::ostringstream oss;
    oss << "\"fen\":\"" << jsonlite::escape(fen) << "\","
        << "\"time_left_ms\":" << time_left_ms << ","
        << "\"last_uncertainty\":" << last_uncertainty << ","
        << "\"tactical_shot_ratio\":" << tactical_shot_ratio;
    return oss.str();
}
} // namespace

MetaClient::MetaClient(std::string h, int p) : host(std::move(h)), port(p) {}

void MetaClient::configure(const std::string &h, int p) {
    if (h == host && p == port) return;
    host = h;
    port = p;
    socket.close();
    retryAfter = {};
}

bool MetaClient::roundTrip(const std::string &request, std::string &response) {
    auto now = std::chrono::steady_clock::now();
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!socket.isOpen()) {
            if (now < retryAfter) return false;
            if (!socket.connect(host, port, kConnectTimeoutMs)) {
                retryAfter = now + std::chrono::seconds(kBackoffSeconds);
                return false;
            }
        }
        // A stale connection (server restarted) fails here; reconnect once.
        if (socket.request(request, response, kRequestTimeoutMs)) return true;
    }
    retryAfter = now + std::chrono::seconds(kBackoffSeconds);
    return false;
}

bool MetaClient::predict(const std::string &fen, int time_left_ms, float last_uncertainty, float tactical_shot_ratio,
                         MetaDeltas &out) {
    out = MetaDeltas{};
    std::string response;
    if (!roundTrip("{\"command\":\"predict\"," + features(fen, time_left_ms, last_uncertainty, tactical_shot_ratio) + "}",
                   response) ||
        !jsonlite::statusOk(response))
        return false;
    double d = 0, b = 0, e = 0;
    if (!jsonlite::findNumber(response, "dfs_depth_delta", d) || !jsonlite::findNumber(response, "mcts_budget_delta", b) ||
        !jsonlite::findNumber(response, "bandit_exploration_delta", e))
        return false;
    out.dfs_depth_delta = static_cast<float>(d);
    out.mcts_budget_delta = static_cast<float>(b);
    out.bandit_exploration_delta = static_cast<float>(e);
    return true;
}

bool MetaClient::add_sample(const std::string &fen, int time_left_ms, float last_uncertainty, float tactical_shot_ratio,
                            const MetaDeltas &d, float reward) {
    std::ostringstream oss;
    oss << "{\"command\":\"add_sample\"," << features(fen, time_left_ms, last_uncertainty, tactical_shot_ratio)
        << ",\"chosen_deltas\":{\"dfs_depth_delta\":" << d.dfs_depth_delta
        << ",\"mcts_budget_delta\":" << d.mcts_budget_delta
        << ",\"bandit_exploration_delta\":" << d.bandit_exploration_delta << "},\"reward\":" << reward << "}";
    std::string response;
    return roundTrip(oss.str(), response) && jsonlite::statusOk(response);
}

bool MetaClient::train(int steps) {
    std::string response;
    return roundTrip("{\"command\":\"train\",\"steps\":" + std::to_string(steps) + "}", response) &&
           jsonlite::statusOk(response);
}
