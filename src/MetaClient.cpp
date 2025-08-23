#include "MetaClient.h"

#include <iostream>
#include <sstream>
#include <cstring>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
   typedef int socklen_t;
#else
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#endif

// Simple JSON building helpers
static std::string escape_json_string(const std::string& str) {
    std::string result;
    for (char c : str) {
        if (c == '"' || c == '\\') result += '\\';
        result += c;
    }
    return result;
}

static std::string build_predict_request(const std::string& fen, int time_left_ms, 
                                        float last_uncertainty, float tactical_shot_ratio) {
    std::ostringstream oss;
    oss << "{"
        << "\"command\":\"predict\","
        << "\"fen\":\"" << escape_json_string(fen) << "\","
        << "\"time_left_ms\":" << time_left_ms << ","
        << "\"last_uncertainty\":" << last_uncertainty << ","
        << "\"tactical_shot_ratio\":" << tactical_shot_ratio
        << "}";
    return oss.str();
}

static std::string build_sample_request(const std::string& fen, int time_left_ms,
                                       float last_uncertainty, float tactical_shot_ratio,
                                       const MetaDeltas& deltas, float elo_gain_per_sec) {
    std::ostringstream oss;
    oss << "{"
        << "\"command\":\"add_sample\","
        << "\"fen\":\"" << escape_json_string(fen) << "\","
        << "\"time_left_ms\":" << time_left_ms << ","
        << "\"last_uncertainty\":" << last_uncertainty << ","
        << "\"tactical_shot_ratio\":" << tactical_shot_ratio << ","
        << "\"chosen_deltas\":{"
        << "\"dfs_depth_delta\":" << deltas.dfs_depth_delta << ","
        << "\"mcts_budget_delta\":" << deltas.mcts_budget_delta << ","
        << "\"bandit_exploration_delta\":" << deltas.bandit_exploration_delta
        << "},"
        << "\"elo_gain_per_sec\":" << elo_gain_per_sec
        << "}";
    return oss.str();
}

static std::string build_train_request(int steps) {
    std::ostringstream oss;
    oss << "{\"command\":\"train\",\"steps\":" << steps << "}";
    return oss.str();
}

// Simple JSON parsing (extract float values)
static float extract_float(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return 0.0f;
    
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    size_t end = pos;
    while (end < json.length() && (std::isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == 'e' || json[end] == 'E' || json[end] == '+')) {
        end++;
    }
    
    if (end > pos) {
        try {
            return std::stof(json.substr(pos, end - pos));
        } catch (...) {
            return 0.0f;
        }
    }
    return 0.0f;
}

MetaClient::MetaClient(const std::string& h, int p) : host(h), port(p) {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
#endif
}

MetaClient::~MetaClient() {
    close_connection();
#ifdef _WIN32
    WSACleanup();
#endif
}

int MetaClient::create_connection() {
    if (sock >= 0) return sock; // Reuse existing connection
    
#ifdef _WIN32
    sock = static_cast<int>(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (sock == INVALID_SOCKET) return -1;
#else
    sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = inet_addr(host.c_str());

#ifdef _WIN32
    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        closesocket(sock);
        sock = -1;
        return -1;
    }
#else
    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(sock);
        sock = -1;
        return -1;
    }
#endif

    connected = true;
    return sock;
}

void MetaClient::close_connection() {
    if (sock >= 0) {
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        sock = -1;
    }
    connected = false;
}

std::string MetaClient::send_request(const std::string& json_request) {
    if (create_connection() < 0) {
        return "{\"status\":\"error\",\"message\":\"connection failed\"}";
    }
    
    std::string request = json_request + "\n";
    
    // Send request
#ifdef _WIN32
    if (::send(sock, request.c_str(), static_cast<int>(request.size()), 0) == SOCKET_ERROR) {
#else
    if (::send(sock, request.c_str(), request.size(), 0) < 0) {
#endif
        close_connection();
        return "{\"status\":\"error\",\"message\":\"send failed\"}";
    }
    
    // Read response
    std::string response;
    char buffer[4096];
    while (true) {
#ifdef _WIN32
        int n = ::recv(sock, buffer, sizeof(buffer), 0);
#else
        ssize_t n = ::recv(sock, buffer, sizeof(buffer), 0);
#endif
        if (n <= 0) break;
        
        response.append(buffer, buffer + n);
        if (response.find('\n') != std::string::npos) break;
    }
    
    return response;
}

MetaDeltas MetaClient::predict(const std::string& fen, int time_left_ms, 
                              float last_uncertainty, float tactical_shot_ratio) {
    MetaDeltas result;
    
    std::string request = build_predict_request(fen, time_left_ms, last_uncertainty, tactical_shot_ratio);
    std::string response = send_request(request);
    
    // Parse deltas from response
    result.dfs_depth_delta = extract_float(response, "dfs_depth_delta");
    result.mcts_budget_delta = extract_float(response, "mcts_budget_delta");
    result.bandit_exploration_delta = extract_float(response, "bandit_exploration_delta");
    
    return result;
}

bool MetaClient::add_sample(const std::string& fen, int time_left_ms, float last_uncertainty,
                           float tactical_shot_ratio, const MetaDeltas& chosen_deltas, 
                           float elo_gain_per_sec) {
    std::string request = build_sample_request(fen, time_left_ms, last_uncertainty, 
                                              tactical_shot_ratio, chosen_deltas, elo_gain_per_sec);
    std::string response = send_request(request);
    
    return response.find("\"status\":\"ok\"") != std::string::npos;
}

bool MetaClient::train(int steps) {
    std::string request = build_train_request(steps);
    std::string response = send_request(request);
    
    return response.find("\"status\":\"ok\"") != std::string::npos;
}
