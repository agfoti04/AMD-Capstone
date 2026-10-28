// Automated bidirectional test.
//   Server: ./linktest server --port 5001
//   Client: ./linktest client --host <SERVER_IP> --port 5001
//
// Phase 1: client -> server -> client round trips (latency)
// Phase 2: server -> client -> server round trips (latency)
// Phase 3: simultaneous streaming both directions, order/integrity checked
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

#include "net.h"

using Clock = std::chrono::steady_clock;
static const int N_PING = 200;
static const int N_STREAM = 2000;
static const std::string PAYLOAD(1024, 'x');

static void check(bool ok, const std::string& msg) {
    if (!ok) throw std::runtime_error(msg);
}

static void ping_side(int fd, LineReader& r, const char* label) {
    std::vector<double> rtt;
    std::string line;
    for (int i = 0; i < N_PING; i++) {
        auto t0 = Clock::now();
        check(send_all(fd, "PING " + std::to_string(i) + "\n"), "send failed");
        check(r.read_line(line), "peer closed connection");
        check(line == "PONG " + std::to_string(i), "bad reply: " + line);
        rtt.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(rtt.begin(), rtt.end());
    std::printf("  %s: %d/%d OK | RTT ms  min %.2f  median %.2f  max %.2f\n", label, N_PING, N_PING,
                rtt.front(), rtt[rtt.size() / 2], rtt.back());
}

static void pong_side(int fd, LineReader& r) {
    std::string line;
    for (int i = 0; i < N_PING; i++) {
        check(r.read_line(line), "peer closed connection");
        check(line.rfind("PING ", 0) == 0, "expected PING, got: " + line);
        check(send_all(fd, "PONG " + line.substr(5) + "\n"), "send failed");
    }
}

static void stream_phase(int fd, LineReader& r) {
    auto t0 = Clock::now();
    bool send_ok = true;
    std::thread tx([&] {
        for (int i = 0; i < N_STREAM && send_ok; i++)
            send_ok = send_all(fd, "DATA " + std::to_string(i) + " " + PAYLOAD + "\n");
    });
    std::string line;
    try {
        for (int i = 0; i < N_STREAM; i++) {
            check(r.read_line(line), "peer closed connection");
            check(line == "DATA " + std::to_string(i) + " " + PAYLOAD,
                  "out-of-order or corrupt message at index " + std::to_string(i));
        }
    } catch (...) {
        tx.join();
        throw;
    }
    tx.join();
    check(send_ok, "send failed during streaming");
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    double mb = N_STREAM * (PAYLOAD.size() + 12.0) * 2 / 1e6;
    std::printf("  %d msgs each way OK in %.2fs (~%.1f MB/s combined)\n", N_STREAM, dt, mb / dt);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: linktest server [--port N]\n       linktest client --host IP [--port N]\n");
        return 1;
    }
    std::string mode = argv[1], host;
    int port = 5001;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    }
    bool client = (mode == "client");
    net_init();

    try {
        int fd;
        if (client) {
            check(!host.empty(), "client mode needs --host <SERVER_IP>");
            fd = tcp_connect(host, port);
            std::printf("Connected to %s:%d\n", host.c_str(), port);
        } else {
            std::printf("Waiting for client on port %d...\n", port);
            std::fflush(stdout);
            std::string peer;
            fd = tcp_listen_accept(port, &peer);
            std::printf("Client connected: %s\n", peer.c_str());
        }
        LineReader r(fd);

        std::printf("Phase 1: client -> server -> client\n");
        if (client) ping_side(fd, r, "client"); else pong_side(fd, r);

        std::printf("Phase 2: server -> client -> server\n");
        if (client) pong_side(fd, r); else ping_side(fd, r, "server");

        std::printf("Phase 3: simultaneous streaming both directions\n");
        stream_phase(fd, r);

        std::printf("\nALL TESTS PASSED\n");
        close(fd);
    } catch (const std::exception& e) {
        std::printf("\nTEST FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}