// Runs on the PC. Sends DriveCommands to the Pi and prints the Pi's replies.
//   ./drive_sender --host <PI_IP> --port 5003 [--hz 5]
//
// For now commands come from a canned demo sequence. To use ROS2 later, only
// get_drive_command() below needs to change.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "drive_cmd.h"
#include "net.h"

// =====================================================================
// ROS2 HOOK: command source.
// Fill 'cmd' with the latest velocity request and return true to send it,
// or return false to end the session.
//
// Later, replace the demo with something like (inside your rclcpp node):
//     void cmd_vel_cb(const geometry_msgs::msg::Twist::SharedPtr msg) {
//         latest.linear_x  = msg->linear.x;
//         latest.angular_z = msg->angular.z;
//     }
// and have this function copy 'latest' (mutex-protected) into 'cmd'.
// Only linear_x / angular_z / stop need to be set; seq is added by main().
// =====================================================================
static bool get_drive_command(DriveCommand& cmd) {
    // TEMP DEMO DATA: {linear_x m/s, angular_z rad/s}, 5 messages each
    static const double demo[][2] = {
        {0.50, 0.0},   // forward
        {0.0, 1.0},    // spin left in place
        {0.30, -0.5},  // arc right
        {-0.30, 0.0},  // reverse
        {0.0, 0.0},    // zero velocity
    };
    static const int REPEAT = 5;
    static int n = 0;
    const int total = (int)(sizeof(demo) / sizeof(demo[0])) * REPEAT;

    if (n >= total) return false;
    cmd.linear_x = demo[n / REPEAT][0];
    cmd.angular_z = demo[n / REPEAT][1];
    cmd.stop = false;
    n++;
    return true;
}

int main(int argc, char** argv) {
    std::string host;
    int port = 5003, hz = 5;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--hz" && i + 1 < argc) hz = std::atoi(argv[++i]);
    }
    if (host.empty() || hz < 1) {
        std::fprintf(stderr, "usage: drive_sender --host <PI_IP> [--port N] [--hz N]\n");
        return 1;
    }
    net_init();

    try {
        int fd = tcp_connect(host, port);
        std::printf("Connected to %s:%d, sending at %d Hz\n", host.c_str(), port, hz);
        LineReader r(fd);
        unsigned seq = 0;
        int failures = 0;
        DriveCommand cmd;
        std::string reply;
        auto period = std::chrono::milliseconds(1000 / hz);

        while (get_drive_command(cmd)) {
            cmd.seq = seq++;
            std::string wire = serialize(cmd);
            if (!send_all(fd, wire) || !r.read_line(reply)) {
                std::printf("Pi disconnected.\n");
                return 1;
            }
            std::printf("sent seq=%-3u lin=%+.2f ang=%+.2f -> Pi: %s\n", cmd.seq, cmd.linear_x,
                        cmd.angular_z, reply.c_str());
            if (reply.rfind("OK", 0) != 0) failures++;
            std::this_thread::sleep_for(period);
        }

        // Always finish with an explicit STOP.
        DriveCommand stop;
        stop.stop = true;
        stop.seq = seq++;
        send_all(fd, serialize(stop));
        r.read_line(reply);
        std::printf("sent STOP seq=%u -> Pi: %s\n", stop.seq, reply.c_str());
        close(fd);

        std::printf("\n%s (%u commands, %d rejected)\n", failures == 0 ? "DONE: all accepted" : "DONE: some rejected", seq, failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}