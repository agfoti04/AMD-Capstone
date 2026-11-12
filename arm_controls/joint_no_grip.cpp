// pca9685_arm_teleop.cpp
//
// Keyboard teleop for the 5-DOF arm through a PCA9685 servo driver on a
// Raspberry Pi. Mirrors arm_teleop.py: each key press jogs one joint by a fixed
// step and targets are clamped to the URDF joint limits (and to the servo's
// safe pulse range).
//
// Build:  g++ -O2 -std=c++17 pca9685_arm_teleop.cpp -o pca9685_arm_teleop
// Run:    ./pca9685_arm_teleop
//
// Controls:
//   Joint_1: q / a     Joint_2: w / s     Joint_3: e / d
//   Joint_4: r / f     Joint_6: y / h
//   Gripper: u (open) / j (close)
//   c: go to every joint's zero position (opens the gripper)
//   x / Ctrl+C: quit

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

// ----------------------------- Configuration -------------------------------

// I2C
static const char*   I2C_DEVICE   = "/dev/i2c-1";
static const uint8_t PCA9685_ADDR = 0x40;

// PCA9685
static const double PWM_FREQ_HZ = 50.0;
static const double OSC_HZ      = 25000000.0;

// Teleop
static const double ARM_STEP_RADIANS     = 0.02;   // same as arm_teleop.py
static const double GRIPPER_STEP_RADIANS = 0.05;   // same as arm_teleop.py

// Pulse width change per radian of joint motion. 900 us per 1.57 rad is the
// scale used in the earlier two-joint version, so jog behaviour is unchanged.
// (A true 180-degree servo spanning 500-2500 us would be 2000/pi = 636.6.)
static const double US_PER_RAD = 900.0 / 1.57;

struct JointConfig {
    const char* name;
    const char* label;        // short name for the status line
    int    channel;           // PCA9685 channel (0-15)
    double min_rad;           // URDF lower limit
    double max_rad;           // URDF upper limit
    double sign;              // +1.0 or -1.0 to reverse the direction
    double center_us;         // pulse at 0 rad (the joint's "straight" position)
    double pulse_min_us;      // never command a pulse below this
    double pulse_max_us;      // never command a pulse above this
    double step_rad;          // radians moved per key press
};

// One row per joint. Limits match JOINT_LIMITS in arm_teleop.py.
// center_us is where 0 rad sits: 1500 us is the servo's middle. Joint_2 uses
// 1260 us because its straight-up position is about 24 deg from the servo
// middle (24 deg * 10 us/deg = 240 us below 1500).
// The pulse limits are clamped, so a joint's real travel is the overlap of its
// URDF limits and what these pulse limits allow (reported at startup).
static const std::vector<JointConfig> JOINTS = {
    // name       label  ch   min     max    sign  center  pmin    pmax   step
    {"Joint_1",   "J1",   0, -1.57,  1.57,   1.0, 1500.0, 600.0, 2400.0, ARM_STEP_RADIANS},
    {"Joint_2",   "J2",   1, -1.57,  1.57,   1.0, 1260.0, 600.0, 2400.0, ARM_STEP_RADIANS},
    {"Joint_3",   "J3",   2, -2.20,  2.27,   1.0, 1500.0, 600.0, 2400.0, ARM_STEP_RADIANS},
    {"Joint_4",   "J4",   3, -1.57,  1.57,   1.0, 1500.0, 600.0, 2400.0, ARM_STEP_RADIANS},
    {"Joint_6",   "J6",   4, -1.57,  1.57,   1.0, 1500.0, 600.0, 2400.0, ARM_STEP_RADIANS},
    // Gripper: 0 rad = fully open, -1.57 rad = closed (u opens / j closes, as in
    // arm_teleop.py). center_us is the OPEN pulse, so this range maps to
    // 1500 us (open) down to 600 us (closed). Tune center_us and pmin to your
    // gripper's real open/closed positions, and flip sign if it moves backwards.
    {"Gripper",   "G",    5, -1.57,  0.00,   1.0, 1500.0, 600.0, 1500.0, GRIPPER_STEP_RADIANS},
};

struct KeyBinding {
    char key;
    int  joint;       // index into JOINTS
    int  direction;   // +1 or -1
};

static const std::vector<KeyBinding> KEY_BINDINGS = {
    {'q', 0, +1}, {'a', 0, -1},
    {'w', 1, +1}, {'s', 1, -1},
    {'e', 2, +1}, {'d', 2, -1},
    {'r', 3, +1}, {'f', 3, -1},
    {'y', 4, +1}, {'h', 4, -1},
    {'u', 5, +1}, {'j', 5, -1},   // gripper: u opens, j closes
};

// ----------------------------- PCA9685 driver ------------------------------

namespace reg {
constexpr uint8_t MODE1     = 0x00;
constexpr uint8_t MODE2     = 0x01;
constexpr uint8_t LED0_ON_L = 0x06;
constexpr uint8_t PRESCALE  = 0xFE;
}  // namespace reg

class PCA9685 {
public:
    ~PCA9685() { if (fd_ >= 0) close(fd_); }

    bool open_bus(const char* dev, uint8_t addr) {
        fd_ = open(dev, O_RDWR);
        if (fd_ < 0) {
            std::fprintf(stderr, "Cannot open %s: %s\n", dev, std::strerror(errno));
            return false;
        }
        if (ioctl(fd_, I2C_SLAVE, addr) < 0) {
            std::fprintf(stderr, "Cannot select I2C address 0x%02X: %s\n",
                         addr, std::strerror(errno));
            return false;
        }
        return true;
    }

    bool init(double freq_hz) {
        if (!write_reg(reg::MODE2, 0x04)) return false;   // totem-pole outputs
        if (!write_reg(reg::MODE1, 0x00)) return false;
        usleep(5000);
        if (!set_frequency(freq_hz)) return false;
        for (int ch = 0; ch < 16; ++ch)
            if (!set_raw(ch, 0, 0)) return false;
        return true;
    }

    bool set_frequency(double freq_hz) {
        double prescale_f = OSC_HZ / (4096.0 * freq_hz) - 1.0;
        uint8_t prescale = static_cast<uint8_t>(std::floor(prescale_f + 0.5));

        uint8_t old_mode;
        if (!read_reg(reg::MODE1, old_mode)) return false;
        uint8_t sleep_mode = (old_mode & 0x7F) | 0x10;    // clear RESTART, set SLEEP
        if (!write_reg(reg::MODE1, sleep_mode)) return false;
        if (!write_reg(reg::PRESCALE, prescale)) return false;
        if (!write_reg(reg::MODE1, old_mode & ~0x10)) return false;
        usleep(5000);
        return write_reg(reg::MODE1, (old_mode & ~0x10) | 0xA0);  // RESTART + auto-increment
    }

    bool set_raw(int channel, uint16_t on, uint16_t off) {
        uint8_t base = reg::LED0_ON_L + 4 * channel;
        return write_reg(base,     on & 0xFF) &&
               write_reg(base + 1, on >> 8) &&
               write_reg(base + 2, off & 0xFF) &&
               write_reg(base + 3, off >> 8);
    }

    bool set_pulse_us(int channel, double pulse_us, double freq_hz) {
        double period_us = 1e6 / freq_hz;
        int ticks = static_cast<int>(std::floor(pulse_us / period_us * 4096.0 + 0.5));
        ticks = std::min(std::max(ticks, 0), 4095);
        return set_raw(channel, 0, static_cast<uint16_t>(ticks));
    }

    bool release(int channel) { return set_raw(channel, 0, 0); }

private:
    bool write_reg(uint8_t r, uint8_t value) {
        uint8_t buf[2] = {r, value};
        if (write(fd_, buf, 2) != 2) {
            std::fprintf(stderr, "\nI2C write failed (reg 0x%02X): %s\n", r, std::strerror(errno));
            return false;
        }
        return true;
    }

    bool read_reg(uint8_t r, uint8_t& value) {
        if (write(fd_, &r, 1) != 1 || read(fd_, &value, 1) != 1) {
            std::fprintf(stderr, "\nI2C read failed (reg 0x%02X): %s\n", r, std::strerror(errno));
            return false;
        }
        return true;
    }

    int fd_ = -1;
};

// ------------------------------ Joint mapping ------------------------------

// Linear map: pulse = center + sign * rad * US_PER_RAD
static double angle_to_pulse_us(const JointConfig& j, double rad) {
    return j.center_us + j.sign * rad * US_PER_RAD;
}

// Usable angle range: the URDF limits narrowed by the allowed pulse range.
struct Range { double lo, hi; };

static Range effective_range(const JointConfig& j) {
    double a = (j.pulse_min_us - j.center_us) / (j.sign * US_PER_RAD);
    double b = (j.pulse_max_us - j.center_us) / (j.sign * US_PER_RAD);
    double pulse_lo = std::min(a, b);
    double pulse_hi = std::max(a, b);
    return {std::max(j.min_rad, pulse_lo), std::min(j.max_rad, pulse_hi)};
}

// --------------------------------- Terminal --------------------------------

static termios g_saved_term;
static bool    g_term_saved = false;

static void restore_terminal() {
    if (g_term_saved) tcsetattr(STDIN_FILENO, TCSADRAIN, &g_saved_term);
}

static bool enter_raw_mode() {
    if (tcgetattr(STDIN_FILENO, &g_saved_term) != 0) return false;
    g_term_saved = true;
    termios raw = g_saved_term;
    raw.c_lflag &= ~(ICANON | ECHO | ISIG);   // ISIG off: Ctrl+C arrives as '\x03'
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    return tcsetattr(STDIN_FILENO, TCSADRAIN, &raw) == 0;
}

static void print_controls(const std::vector<Range>& ranges) {
    std::printf("PCA9685 arm teleop (arm keys jog %.2f rad, about %.2f deg; gripper %.2f rad)\n",
                ARM_STEP_RADIANS, ARM_STEP_RADIANS * 180.0 / M_PI, GRIPPER_STEP_RADIANS);
    std::printf("Positive / negative joint jogs:\n");
    std::printf("  Joint_1: q / a     Joint_2: w / s     Joint_3: e / d\n");
    std::printf("  Joint_4: r / f     Joint_6: y / h\n");
    std::printf("  Gripper: u opens / j closes\n");
    std::printf("  c: zero all joints (this opens the gripper)\n");
    std::printf("Press x or Ctrl+C to quit (servos keep holding their last position).\n\n");
    std::printf("Usable range per joint (URDF limits narrowed by servo pulse limits):\n");
    for (size_t i = 0; i < JOINTS.size(); ++i) {
        const JointConfig& j = JOINTS[i];
        const Range& r = ranges[i];
        bool narrowed = (r.lo > j.min_rad + 1e-9) || (r.hi < j.max_rad - 1e-9);
        std::printf("  %s (ch %d): %+6.1f to %+6.1f deg%s\n", j.name, j.channel,
                    r.lo * 180.0 / M_PI, r.hi * 180.0 / M_PI,
                    narrowed ? "   <- narrower than URDF" : "");
    }
    std::printf("\n");
}

static void print_status(const std::vector<double>& angles, const std::string& note) {
    std::printf("\r");
    for (size_t i = 0; i < JOINTS.size(); ++i)
        std::printf("%s:%+6.1f  ", JOINTS[i].label, angles[i] * 180.0 / M_PI);
    std::printf("%-12s", note.c_str());
    std::fflush(stdout);
}

// ----------------------------------- Main ----------------------------------

static bool command_joint(PCA9685& pca, size_t i, double rad) {
    return pca.set_pulse_us(JOINTS[i].channel, angle_to_pulse_us(JOINTS[i], rad), PWM_FREQ_HZ);
}

int main() {
    if (!isatty(STDIN_FILENO)) {
        std::fprintf(stderr, "pca9685_arm_teleop requires an interactive terminal.\n");
        return 1;
    }

    PCA9685 pca;
    if (!pca.open_bus(I2C_DEVICE, PCA9685_ADDR)) return 1;
    if (!pca.init(PWM_FREQ_HZ)) {
        std::fprintf(stderr, "Failed to initialise PCA9685 (check wiring / address).\n");
        return 1;
    }

    std::vector<Range> ranges;
    for (const JointConfig& j : JOINTS) ranges.push_back(effective_range(j));

    // No position feedback from hobby servos, so every joint starts at 0 rad
    // and moves there immediately. They are started one at a time, 100 ms
    // apart, to avoid one large current spike on the supply.
    std::vector<double> angles(JOINTS.size(), 0.0);
    for (size_t i = 0; i < JOINTS.size(); ++i) {
        if (!command_joint(pca, i, angles[i])) return 1;
        usleep(100000);
    }

    print_controls(ranges);
    std::printf("All joints start at 0 rad; waiting for jog input.\n");

    if (!enter_raw_mode()) {
        std::fprintf(stderr, "Failed to set terminal to raw mode.\n");
        return 1;
    }

    int exit_code = 0;
    while (true) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        timeval tv{0, 100000};   // 100 ms
        int r = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);
        if (r < 0) { if (errno == EINTR) continue; break; }
        if (r == 0) continue;

        char key;
        if (read(STDIN_FILENO, &key, 1) != 1) break;
        if (key == 'x' || key == '\x03') break;

        if (key == 'c') {
            for (size_t i = 0; i < JOINTS.size(); ++i) {
                angles[i] = 0.0;
                if (!command_joint(pca, i, 0.0)) { exit_code = 1; break; }
                usleep(100000);
            }
            if (exit_code) break;
            print_status(angles, "(zeroed)");
            continue;
        }

        for (const KeyBinding& kb : KEY_BINDINGS) {
            if (kb.key != key) continue;
            const JointConfig& j = JOINTS[kb.joint];
            const Range& rg = ranges[kb.joint];
            double requested = angles[kb.joint] + kb.direction * j.step_rad;
            double target = std::min(std::max(requested, rg.lo), rg.hi);
            angles[kb.joint] = target;
            if (!command_joint(pca, kb.joint, target)) { exit_code = 1; break; }

            std::string note;
            if (std::fabs(target - requested) > 1e-9)
                note = std::string("(") + j.label + " limit)";
            print_status(angles, note);
            break;
        }
        if (exit_code) break;
    }

    restore_terminal();
    std::printf("\n");
    return exit_code;
}
