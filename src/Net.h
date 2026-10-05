#pragma once

// Minimal line-oriented TCP client used to talk to the Python services
// (rpc_server.py, meta_learner.py). Every operation has a timeout so a slow or
// missing server can never hang the engine, writes never raise SIGPIPE, and a
// connection is kept open across requests.

#include <cstdint>
#include <string>
#include <vector>

class LineSocket {
public:
    LineSocket() = default;
    ~LineSocket();
    LineSocket(const LineSocket &) = delete;
    LineSocket &operator=(const LineSocket &) = delete;

    bool connect(const std::string &host, int port, int timeoutMs);
    bool isOpen() const { return fd != kInvalid; }
    void close();

    // Sends `line` followed by '\n'.
    bool sendLine(const std::string &line, int timeoutMs);
    // Reads up to (not including) the next '\n'. False on timeout/EOF/error.
    bool recvLine(std::string &line, int timeoutMs);

    // One request/response round trip. On failure the connection is closed.
    bool request(const std::string &line, std::string &response, int timeoutMs);

private:
#ifdef _WIN32
    using Handle = uintptr_t;
#else
    using Handle = int;
#endif
    static constexpr Handle kInvalid = static_cast<Handle>(-1);
    Handle fd = kInvalid;
    std::string buffer;
};

// Tiny helpers for the flat JSON the services emit. They tolerate
// whitespace and key order but are not a general JSON parser.
namespace jsonlite {
std::string escape(const std::string &s);
bool findNumber(const std::string &json, const std::string &key, double &out);
bool findNumberArray(const std::string &json, const std::string &key, std::vector<double> &out);
bool findString(const std::string &json, const std::string &key, std::string &out);
inline bool statusOk(const std::string &json) {
    std::string status;
    return findString(json, "status", status) && status == "ok";
}
} // namespace jsonlite
