#include "Net.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {

#ifdef _WIN32
struct WinsockInit {
    WinsockInit() {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    }
    ~WinsockInit() { WSACleanup(); }
};
void ensureWinsock() { static WinsockInit init; }
#else
void ensureWinsock() {}
#endif

#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

using Clock = std::chrono::steady_clock;

int remainingMs(Clock::time_point deadline) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    return ms > 0 ? static_cast<int>(ms) : 0;
}

template <class H>
bool waitFor(H fd, bool forWrite, int timeoutMs) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(fd, &set);
    timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
#ifdef _WIN32
    int n = select(0, forWrite ? nullptr : &set, forWrite ? &set : nullptr, nullptr, &tv); // nfds is ignored on Windows
#else
    int n = select(static_cast<int>(fd) + 1, forWrite ? nullptr : &set, forWrite ? &set : nullptr, nullptr, &tv);
#endif
    return n > 0;
}

} // namespace

LineSocket::~LineSocket() { close(); }

void LineSocket::close() {
    if (fd == kInvalid) return;
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(fd));
#else
    ::close(fd);
#endif
    fd = kInvalid;
    buffer.clear();
}

bool LineSocket::connect(const std::string &host, int port, int timeoutMs) {
    close();
    ensureWinsock();
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    std::string portStr = std::to_string(port);
    if (getaddrinfo(host.c_str(), portStr.c_str(), &hints, &res) != 0 || !res) return false;

    auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    bool connected = false;
    for (addrinfo *ai = res; ai && !connected; ai = ai->ai_next) {
#ifdef _WIN32
        SOCKET s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == INVALID_SOCKET) continue;
        u_long nonBlocking = 1;
        ioctlsocket(s, FIONBIO, &nonBlocking);
        int rc = ::connect(s, ai->ai_addr, static_cast<int>(ai->ai_addrlen));
        bool inProgress = rc != 0 && WSAGetLastError() == WSAEWOULDBLOCK;
#else
        int s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s < 0) continue;
#ifdef SO_NOSIGPIPE
        int one = 1;
        setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
        fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
        int rc = ::connect(s, ai->ai_addr, ai->ai_addrlen);
        bool inProgress = rc != 0 && errno == EINPROGRESS;
#endif
        if (rc == 0) {
            connected = true;
        } else if (inProgress && waitFor(s, true, remainingMs(deadline))) {
            int err = 0;
            socklen_t len = sizeof(err);
            getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&err), &len);
            connected = err == 0;
        }
        if (connected) {
            int one = 1;
            setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof(one));
            fd = static_cast<Handle>(s);
        } else {
#ifdef _WIN32
            closesocket(s);
#else
            ::close(s);
#endif
        }
    }
    freeaddrinfo(res);
    return connected;
}

bool LineSocket::sendLine(const std::string &line, int timeoutMs) {
    if (fd == kInvalid) return false;
    std::string data = line + "\n";
    auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    size_t sent = 0;
    while (sent < data.size()) {
        if (!waitFor(fd, true, remainingMs(deadline))) return false;
#ifdef _WIN32
        int n = send(static_cast<SOCKET>(fd), data.data() + sent, static_cast<int>(data.size() - sent), 0);
#else
        ssize_t n = send(fd, data.data() + sent, data.size() - sent, kSendFlags);
#endif
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool LineSocket::recvLine(std::string &line, int timeoutMs) {
    if (fd == kInvalid) return false;
    auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        size_t nl = buffer.find('\n');
        if (nl != std::string::npos) {
            line = buffer.substr(0, nl);
            buffer.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
        if (buffer.size() > (64u << 20)) return false; // runaway response
        if (!waitFor(fd, false, remainingMs(deadline))) return false;
        char chunk[16384];
#ifdef _WIN32
        int n = recv(static_cast<SOCKET>(fd), chunk, sizeof(chunk), 0);
#else
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
#endif
        if (n <= 0) return false;
        buffer.append(chunk, static_cast<size_t>(n));
    }
}

bool LineSocket::request(const std::string &line, std::string &response, int timeoutMs) {
    if (!sendLine(line, timeoutMs) || !recvLine(response, timeoutMs)) {
        close();
        return false;
    }
    return true;
}

namespace jsonlite {

std::string escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

namespace {
// Position just after `"key"<ws>:<ws>`, or npos.
size_t valueStart(const std::string &json, const std::string &key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        size_t p = pos + needle.size();
        while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
        if (p < json.size() && json[p] == ':') {
            ++p;
            while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
            return p;
        }
        pos += needle.size();
    }
    return std::string::npos;
}

bool parseNumberAt(const std::string &json, size_t &p, double &out) {
    const char *begin = json.c_str() + p;
    char *end = nullptr;
    double v = std::strtod(begin, &end);
    if (end == begin) return false;
    out = v;
    p += static_cast<size_t>(end - begin);
    return true;
}
} // namespace

bool findNumber(const std::string &json, const std::string &key, double &out) {
    size_t p = valueStart(json, key);
    return p != std::string::npos && parseNumberAt(json, p, out);
}

bool findNumberArray(const std::string &json, const std::string &key, std::vector<double> &out) {
    size_t p = valueStart(json, key);
    if (p == std::string::npos || p >= json.size() || json[p] != '[') return false;
    out.clear();
    ++p;
    for (;;) {
        while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
        if (p >= json.size()) return false;
        if (json[p] == ']') return true;
        double v;
        if (!parseNumberAt(json, p, v)) return false;
        out.push_back(v);
        while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
        if (p < json.size() && json[p] == ',') ++p;
    }
}

bool findString(const std::string &json, const std::string &key, std::string &out) {
    size_t p = valueStart(json, key);
    if (p == std::string::npos || p >= json.size() || json[p] != '"') return false;
    out.clear();
    for (++p; p < json.size(); ++p) {
        char c = json[p];
        if (c == '\\' && p + 1 < json.size()) { out += json[++p]; continue; }
        if (c == '"') return true;
        out += c;
    }
    return false;
}

} // namespace jsonlite
