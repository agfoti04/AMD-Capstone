// Drive command for a 4-motor differential (skid-steer) base.
// Shared by the PC (sender) and the Pi (receiver).
//
// Wire format, one line per command:
//   DRIVE <seq> <linear_x> <angular_z>
//   STOP  <seq>
//
// This matches ROS2 geometry_msgs/Twist: linear.x -> linear_x (m/s, +forward),
// angular.z -> angular_z (rad/s, +counter-clockwise / left turn).
#pragma once
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

struct DriveCommand {
    unsigned seq = 0;       // sequence number, lets the Pi detect dropped/out-of-order commands
    double linear_x = 0.0;  // m/s
    double angular_z = 0.0; // rad/s
    bool stop = false;      // true = emergency stop, ignore velocities
};

// ---- Robot geometry (PLACEHOLDERS: measure your robot and update) ----
constexpr double TRACK_WIDTH_M = 0.30;  // distance between left and right wheels
constexpr double WHEEL_RADIUS_M = 0.05;
constexpr double MAX_LINEAR = 1.0;      // m/s   (commands beyond this are rejected)
constexpr double MAX_ANGULAR = 3.0;     // rad/s

// ---- Serialize / parse ----
inline std::string serialize(const DriveCommand& c) {
    char buf[128];
    if (c.stop) std::snprintf(buf, sizeof(buf), "STOP %u\n", c.seq);
    else std::snprintf(buf, sizeof(buf), "DRIVE %u %.4f %.4f\n", c.seq, c.linear_x, c.angular_z);
    return buf;
}

inline bool parse(const std::string& line, DriveCommand& c, std::string& err) {
    std::istringstream in(line);
    std::string word;
    c = DriveCommand{};
    if (!(in >> word)) { err = "empty command"; return false; }
    if (word == "STOP") {
        if (!(in >> c.seq)) { err = "usage: STOP <seq>"; return false; }
        c.stop = true;
        return true;
    }
    if (word == "DRIVE") {
        if (!(in >> c.seq >> c.linear_x >> c.angular_z)) { err = "usage: DRIVE <seq> <linear_x> <angular_z>"; return false; }
        if (!std::isfinite(c.linear_x) || !std::isfinite(c.angular_z)) { err = "non-finite value"; return false; }
        if (std::fabs(c.linear_x) > MAX_LINEAR) { err = "linear_x out of range"; return false; }
        if (std::fabs(c.angular_z) > MAX_ANGULAR) { err = "angular_z out of range"; return false; }
        return true;
    }
    err = "unknown command '" + word + "'";
    return false;
}

// ---- Diff-drive kinematics: body velocity -> 4 wheel speeds ----
struct WheelSpeeds {
    double front_left, rear_left, front_right, rear_right;  // wheel surface speed, m/s
};

inline WheelSpeeds to_wheel_speeds(const DriveCommand& c) {
    if (c.stop) return {0, 0, 0, 0};
    double left = c.linear_x - c.angular_z * TRACK_WIDTH_M / 2.0;
    double right = c.linear_x + c.angular_z * TRACK_WIDTH_M / 2.0;
    return {left, left, right, right};  // both wheels on a side get the same speed
}