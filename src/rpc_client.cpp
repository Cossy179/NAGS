#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
#else
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#endif

static bool send_all(int sock, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
#ifdef _WIN32
        int n = ::send(sock, data + sent, static_cast<int>(len - sent), 0);
#else
        ssize_t n = ::send(sock, data + sent, len - sent, 0);
#endif
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

int main() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return 1;
    }
#endif

#ifdef _WIN32
    SOCKET sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) { std::cerr << "socket() failed" << std::endl; return 1; }
#else
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { std::cerr << "socket() failed" << std::endl; return 1; }
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(5555);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    // Retry connect for up to ~5 seconds
    bool connected = false;
    for (int attempt = 0; attempt < 50 && !connected; ++attempt) {
#ifdef _WIN32
        if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) connected = true;
#else
        if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) connected = true;
#endif
        if (!connected) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!connected) { std::cerr << "connect() failed" << std::endl; return 1; }

    // Prepare 8 FENs (startpos duplicated)
    std::vector<std::string> fens(8, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::string json = "{\"fens\":[";
    for (size_t i = 0; i < fens.size(); ++i) {
        if (i) json += ",";
        // escape quotes if any (not expected in FEN)
        json += "\"" + fens[i] + "\"";
    }
    json += "]}\n";

    if (!send_all(sock, json.c_str(), json.size())) {
        std::cerr << "send failed" << std::endl;
        return 1;
    }

    // Read one line response
    std::string resp;
    char buf[4096];
    while (true) {
#ifdef _WIN32
        int n = ::recv(sock, buf, sizeof(buf), 0);
#else
        ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        resp.append(buf, buf + n);
        if (resp.find('\n') != std::string::npos) break;
    }

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif

    // Naive check: count occurrences of "\"value\":" as proxy for results length
    size_t count = 0, pos = 0;
    const std::string needle = "\"value\":";
    while ((pos = resp.find(needle, pos)) != std::string::npos) { ++count; pos += needle.size(); }
    std::cout << "Response length: " << resp.size() << " bytes, value-count: " << count << std::endl;
    if (count != 8) {
        std::cerr << "Expected 8 evaluations, got " << count << std::endl;
        return 2;
    }
    std::cout << "OK: received 8 evaluations in one RPC call." << std::endl;
    return 0;
}


