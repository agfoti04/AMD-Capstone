// Interactive two-way link. One side listens, the other connects.
//   Listener:  ./link listen --port 5000
//   Connector: ./link connect --host <OTHER_IP> --port 5000
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "net.h"

static void usage() {
    std::fprintf(stderr, "usage: link listen [--port N]\n       link connect --host IP [--port N]\n");
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) usage();
    std::string mode = argv[1], host;
    int port = 5000;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else usage();
    }

    net_init();
    int fd;
    try {
        if (mode == "listen") {
            std::printf("[listening on port %d...]\n", port);
            std::fflush(stdout);
            std::string peer;
            fd = tcp_listen_accept(port, &peer);
            std::printf("[connected from %s]\n", peer.c_str());
        } else if (mode == "connect") {
            if (host.empty()) usage();
            fd = tcp_connect(host, port);
            std::printf("[connected to %s:%d]\n", host.c_str(), port);
        } else usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::thread rx([fd] {
        LineReader r(fd);
        std::string line;
        while (r.read_line(line)) {
            std::printf("\r< %s\n> ", line.c_str());
            std::fflush(stdout);
        }
        std::printf("\n[peer disconnected]\n");
        std::_Exit(0);
    });
    rx.detach();

    std::string line;
    std::printf("> ");
    std::fflush(stdout);
    while (std::getline(std::cin, line)) {
        if (!send_all(fd, line + "\n")) break;
        std::printf("> ");
        std::fflush(stdout);
    }
    close(fd);
    return 0;
}