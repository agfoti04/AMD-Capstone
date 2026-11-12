// pca9685_teleop.cpp
//
// Keyboard teleop for one servo (Joint_1 on PCA9685 channel 0) on a Raspberry Pi.
// Mirrors the behaviour of arm_teleop.py: each key press jogs the joint by a
// fixed step, targets are clamped to the URDF joint limits.
//
// Build:  g++ -O2 -std=c++17 pca9685_teleop.cpp -o pca9685_teleop
// Run:    ./pca9685_teleop            (needs access to /dev/i2c-1, see notes)
//
// Controls:  q = Joint_1 +     a = Joint_1 -     c = center     x / Ctrl+C = quit

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

// ----------------------------- Configuration -------------------------------

// I2C
static const char*   I2C_DEVICE   = "/dev/i2c-1";   // Pi 3 header pins 3 (SDA) / 5 (SCL)
static const uint8_t PCA9685_ADDR = 0x40;           // default address

// PCA9685
static const double  PWM_FREQ_HZ  = 50.0;           // standard hobby servo rate
static const double  OSC_HZ       = 25000000.0;     // internal oscillator

// Joint_1 (matches the URDF limits used in arm_teleop.py)
static const int    JOINT1_CHANNEL = 0;
static const double JOINT1_MIN_RAD = -1.57;
static const double JOINT1_MAX_RAD =  1.57;
static const double JOINT1_SIGN    =  1.0;          // set to -1.0 to reverse direction

// Servo pulse widths. Angle -> pulse is a linear map:
//   JOINT1_MIN_RAD -> PULSE_MIN_US,  0 rad -> PULSE_CENTER_US,  JOINT1_MAX_RAD -> PULSE_MAX_US
// Start conservative and widen once you know your servo's real range, otherwise
// it may push against its mechanical end stops (buzzing/stalling).
static const double PULSE_MIN_US    =  600.0;
static const double PULSE_CENTER_US = 1500.0;
static const double PULSE_MAX_US    = 2400.0;

// Teleop
static const double STEP_RADIANS = 0.02;            // same as arm_teleop.py

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
        // Totem-pole outputs, all channels off, then set the PWM frequency.
        if (!write_reg(reg::MODE2, 0x04)) return false;
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

        // Prescale can only be written while the chip is asleep.
        uint8_t old_mode;
        if (!read_reg(reg::MODE1, old_mode)) return false;
        uint8_t sleep_mode = (old_mode & 0x7F) | 0x10;   // clear RESTART, set SLEEP
        if (!write_reg(reg::MODE1, sleep_mode)) return false;
        if (!write_reg(reg::PRESCALE, prescale)) return false;
        if (!write_reg(reg::MODE1, old_mode & ~0x10)) return false;  // wake
        usleep(5000);
        return write_reg(reg::MODE1, (old_mode & ~0x10) | 0xA0);     // RESTART + auto-increment
    }

    // Set channel PWM: output goes high at tick `on` and low at tick `off` (0..4095).
    bool set_raw(int channel, uint16_t on, uint16_t off) {
        uint8_t base = reg::LED0_ON_L + 4 * channel;
        return write_reg(base,     on & 0xFF) &&
               write_reg(base + 1, on >> 8) &&
               write_reg(base + 2, off & 0xFF) &&
               write_reg(base + 3, off >> 8);
    }

    // Set channel pulse width in microseconds (at the configured frequency).
    bool set_pulse_us(int channel, double pulse_us, double freq_hz) {
        double period_us = 1e6 / freq_hz;
        double ticks_f = pulse_us / period_us * 4096.0;
        int ticks = static_cast<int>(std::floor(ticks_f + 0.5));
        ticks = std::min(std::max(ticks, 0), 4095);
        return set_raw(channel, 0, static_cast<uint16_t>(ticks));
    }

    // Stop driving a channel (servo goes limp).
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

static double angle_to_pulse_us(double rad) {
    double a = rad * JOINT1_SIGN;
    // Piecewise linear: negative side uses MIN..CENTER, positive side CENTER..MAX.
    if (a >= 0.0) return PULSE_CENTER_US + (a / JOINT1_MAX_RAD) * (PULSE_MAX_US - PULSE_CENTER_US);
    return PULSE_CENTER_US + (a / JOINT1_MIN_RAD) * (PULSE_MIN_US - PULSE_CENTER_US);
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

static void print_controls() {
    std::printf("PCA9685 servo teleop (each key jogs %.2f rad, about %.2f deg)\n",
                STEP_RADIANS, STEP_RADIANS * 180.0 / M_PI);
    std::printf("  Joint_1: q (+) / a (-)     c: center\n");
    std::printf("Press x or Ctrl+C to quit (servo keeps holding its last position).\n");
    std::printf("Joint_1 target is clamped to [%.2f, %.2f] rad.\n\n",
                JOINT1_MIN_RAD, JOINT1_MAX_RAD);
}

// ----------------------------------- Main ----------------------------------

int main() {
    if (!isatty(STDIN_FILENO)) {
        std::fprintf(stderr, "pca9685_teleop requires an interactive terminal.\n");
        return 1;
    }

    PCA9685 pca;
    if (!pca.open_bus(I2C_DEVICE, PCA9685_ADDR)) return 1;
    if (!pca.init(PWM_FREQ_HZ)) {
        std::fprintf(stderr, "Failed to initialise PCA9685 (check wiring / address).\n");
        return 1;
    }

    // No position feedback from a hobby servo, so we start from the center
    // position (0 rad). The servo will move there immediately.
    double joint1 = 0.0;
    if (!pca.set_pulse_us(JOINT1_CHANNEL, angle_to_pulse_us(joint1), PWM_FREQ_HZ)) return 1;

    print_controls();
    std::printf("Joint_1 starts at 0 rad (center); waiting for jog input.\n");

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

        double requested = joint1;
        if      (key == 'q') requested = joint1 + STEP_RADIANS;
        else if (key == 'a') requested = joint1 - STEP_RADIANS;
        else if (key == 'c') requested = 0.0;
        else continue;

        joint1 = std::min(std::max(requested, JOINT1_MIN_RAD), JOINT1_MAX_RAD);
        double pulse = angle_to_pulse_us(joint1);
        if (!pca.set_pulse_us(JOINT1_CHANNEL, pulse, PWM_FREQ_HZ)) { exit_code = 1; break; }

        const char* status = (std::fabs(joint1 - requested) > 1e-9) ? " (limit reached)" : "";
        std::printf("\rJoint_1: %+6.1f deg  (%.0f us)%s          ",
                    joint1 * 180.0 / M_PI, pulse, status);
        std::fflush(stdout);
    }

    restore_terminal();
    std::printf("\n");
    return exit_code;
}
