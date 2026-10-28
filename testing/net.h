// Small TCP helpers shared by link.cpp and linktest.cpp (POSIX / Linux).
#pragma once
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <csignal>
#include <cstring>
#include <stdexcept>
#include <string>

inline void net_init() { signal(SIGPIPE, SIG_IGN); }

inline void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// Listen on 0.0.0.0:port and block until one client connects.
inline int tcp_listen_accept(int port, std::string* peer_ip = nullptr) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) throw std::runtime_error("socket() failed");
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0)
        throw std::runtime_error(std::string("bind failed: ") + strerror(errno));
    if (listen(srv, 1) < 0) throw std::runtime_error("listen() failed");

    sockaddr_in cli{};
    socklen_t len = sizeof(cli);
    int fd = accept(srv, (sockaddr*)&cli, &len);
    close(srv);
    if (fd < 0) throw std::runtime_error("accept() failed");
    if (peer_ip) *peer_ip = inet_ntoa(cli.sin_addr);
    set_nodelay(fd);
    return fd;
}

// Connect to host:port (IP address or hostname).
inline int tcp_connect(const std::string& host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);
    if (rc != 0) throw std::runtime_error(std::string("resolve failed: ") + gai_strerror(rc));

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        std::string err = strerror(errno);
        if (fd >= 0) close(fd);
        freeaddrinfo(res);
        throw std::runtime_error("connect failed: " + err);
    }
    freeaddrinfo(res);
    set_nodelay(fd);
    return fd;
}

inline bool send_all(int fd, const std::string& s) {
    size_t sent = 0;
    while (sent < s.size()) {
        ssize_t n = send(fd, s.data() + sent, s.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

// Buffered newline-delimited reader.
struct LineReader {
    int fd;
    std::string buf;
    explicit LineReader(int f) : fd(f) {}

    // Returns false when the peer closes the connection.
    bool read_line(std::string& out) {
        for (;;) {
            size_t pos = buf.find('\n');
            if (pos != std::string::npos) {
                out = buf.substr(0, pos);
                buf.erase(0, pos + 1);
                return true;
            }
            char tmp[4096];
            ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf.append(tmp, n);
        }
    }
};