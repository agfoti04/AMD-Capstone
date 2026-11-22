// WASD drive teleop for four VNH5019 motor drivers, Raspberry Pi 3 (C++ / lgpio)
//
// Pins are BCM / GPIO numbers (NOT physical header pin numbers):
//
// Front Left            Front Right           Back Left             Back Right
//   GPIO 4  -> INA        GPIO 22 -> INA        GPIO 10 -> INA        GPIO 25 -> INA
//   GPIO 17 -> INB        GPIO 23 -> INB        GPIO 9  -> INB        GPIO 8  -> INB
//   GPIO 27 -> PWM        GPIO 24 -> PWM        GPIO 11 -> PWM        GPIO 7  -> PWM
//
// Controls (commands are LATCHED: the robot keeps doing the last command until
// you change it or stop it):
//   W            : drive forward
//   S            : drive backward
//   A            : rotate left in place
//   D            : rotate right in place
//   Space (or X) : STOP immediately (brake all motors)
//   Up / '+'     : increase speed limit
//   Down / '-'   : decrease speed limit
//   Z            : flip left-side motor polarity  (only while stopped)
//   C            : flip right-side motor polarity (only while stopped)
//   Q (or Ctrl+C): brake everything and quit
//
// Speed changes are ramped: motors accelerate/decelerate gradually, and a
// direction change (e.g. W then S) slows through zero before reversing.
//
// Install: sudo apt install liblgpio-dev g++
// Build:   g++ -o vnh5019_drive vnh5019_drive.cpp -llgpio
// Run:     ./vnh5019_drive   (run it in a real terminal; use sudo if needed)

#include <lgpio.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

// ---- Settings (adjust to taste) ----
constexpr float PWM_FREQ_HZ = 1000.0f;
constexpr int INITIAL_SPEED = 2;      // percent duty
constexpr int MAX_SPEED = 30;         // safety cap in percent; raise it when you're ready
constexpr int SPEED_STEP = 1;         // percent per key press
constexpr int RAMP_STEP = 1;          // percent change per ramp interval
constexpr int RAMP_INTERVAL_MS = 50;  // 1% / 50 ms = 20% per second

// ---- Motors ----
struct Motor {
    const char* name;
    const char* abbr;
    int pwm;
    int ina;
    int inb;
    bool leftSide;
    bool invert;      // true = wires are mirrored, so flip this motor's polarity
    int target = 0;   // desired signed duty (+ = wheel drives robot forward)
    int current = 0;  // signed duty currently being output (same sense as target)
    int applied = 0;  // last electrical signed value written to hardware
};

constexpr int NUM_MOTORS = 4;

// {name, abbr, PWM, INA, INB, leftSide, invert}
// The invert defaults are a guess (right side mirrored). Use Z / C at runtime
// to flip a side if it spins the wrong way, then update these defaults.
static Motor g_motors[NUM_MOTORS] = {
    {"Front Left",  "FL", 27, 4,  17, true,  false},
    {"Front Right", "FR", 24, 22, 23, false, true},
    {"Back Left",   "BL", 11, 10, 9,  true,  false},
    {"Back Right",  "BR", 7,  25, 8,  false, true},
};

static volatile std::sig_atomic_t g_stop = 0;
static int g_h = -1;  // lgpio chip handle

static void onSignal(int) { g_stop = 1; }

// ---- Terminal raw-mode helper (restores terminal automatically) ----
struct RawTerminal {
    termios original{};
    bool active = false;

    void enable() {
        if (tcgetattr(STDIN_FILENO, &original) != 0) return;
        termios raw = original;
        raw.c_lflag &= ~(ICANON | ECHO);  // no line buffering, no echo
        raw.c_cc[VMIN] = 0;                // read() returns even if no key...
        raw.c_cc[VTIME] = 1;               // ...after 0.1 s
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        active = true;
    }
    ~RawTerminal() {
        if (active) tcsetattr(STDIN_FILENO, TCSANOW, &original);
    }
};

enum class Key {
    None, Forward, Backward, RotLeft, RotRight, Stop,
    SpeedUp, SpeedDown, FlipLeft, FlipRight, Quit
};

static Key readKey() {
    char c;
    if (read(STDIN_FILENO, &c, 1) <= 0) return Key::None;

    if (c == 27) {  // escape sequence, e.g. arrow keys: ESC [ A
        char seq[2];
        if (read(STDIN_FILENO, &seq[0], 1) <= 0) return Key::None;
        if (read(STDIN_FILENO, &seq[1], 1) <= 0) return Key::None;
        if (seq[0] == '[') {
            if (seq[1] == 'A') return Key::SpeedUp;
            if (seq[1] == 'B') return Key::SpeedDown;
        }
        return Key::None;
    }

    switch (c) {
        case 'w': case 'W': return Key::Forward;
        case 's': case 'S': return Key::Backward;
        case 'a': case 'A': return Key::RotLeft;
        case 'd': case 'D': return Key::RotRight;
        case ' ':
        case 'x': case 'X': return Key::Stop;
        case '+': case '=': return Key::SpeedUp;
        case '-': case '_': return Key::SpeedDown;
        case 'z': case 'Z': return Key::FlipLeft;
        case 'c': case 'C': return Key::FlipRight;
        case 'q': case 'Q': return Key::Quit;
        default:            return Key::None;
    }
}

// ---- Drive commands ----
enum class Cmd { Stop, Forward, Backward, RotateLeft, RotateRight };

static const char* cmdName(Cmd c) {
    switch (c) {
        case Cmd::Forward:     return "FORWARD";
        case Cmd::Backward:    return "BACKWARD";
        case Cmd::RotateLeft:  return "ROTATE LEFT";
        case Cmd::RotateRight: return "ROTATE RIGHT";
        default:               return "STOPPED";
    }
}

// +1 = wheel drives robot forward, -1 = backward, 0 = off
static int wheelSign(Cmd c, bool leftSide) {
    switch (c) {
        case Cmd::Forward:     return +1;
        case Cmd::Backward:    return -1;
        case Cmd::RotateLeft:  return leftSide ? -1 : +1;
        case Cmd::RotateRight: return leftSide ? +1 : -1;
        default:               return 0;
    }
}

static void updateTargets(Cmd c, int speed) {
    for (Motor& m : g_motors) m.target = wheelSign(c, m.leftSide) * speed;
}

// ---- Motor helpers ----
static int sgn(int v) { return (v > 0) - (v < 0); }

// PWM off, INA = INB = 0 (brake to ground).
static void brakePins(const Motor& m) {
    lgTxPwm(g_h, m.pwm, PWM_FREQ_HZ, 0.0f, 0, 0);
    lgGpioWrite(g_h, m.pwm, 0);
    lgGpioWrite(g_h, m.ina, 0);
    lgGpioWrite(g_h, m.inb, 0);
}

// Push m.current out to the hardware (only touches pins if something changed).
static void writeOutput(Motor& m) {
    int s = m.current * (m.invert ? -1 : 1);  // electrical sign
    if (s == m.applied) return;

    int prevSign = sgn(m.applied);
    m.applied = s;

    if (s == 0) {
        brakePins(m);
        return;
    }
    if (sgn(s) != prevSign) {  // direction pins only change when the sign changes
        lgGpioWrite(g_h, m.ina, s > 0 ? 1 : 0);
        lgGpioWrite(g_h, m.inb, s > 0 ? 0 : 1);
    }
    // pwmCycles = 0 means run until changed
    lgTxPwm(g_h, m.pwm, PWM_FREQ_HZ, static_cast<float>(std::abs(s)), 0, 0);
}

// Immediate hard stop of every motor (no ramp).
static void emergencyStop() {
    for (Motor& m : g_motors) {
        m.target = 0;
        m.current = 0;
        m.applied = 0;
        brakePins(m);
    }
}

// Move cur toward tgt by at most `step`; passes through zero before reversing.
static int stepToward(int cur, int tgt, int step) {
    if ((cur > 0 && tgt < 0) || (cur < 0 && tgt > 0)) tgt = 0;
    if (cur < tgt) return std::min(cur + step, tgt);
    if (cur > tgt) return std::max(cur - step, tgt);
    return cur;
}

static bool allStopped() {
    for (const Motor& m : g_motors)
        if (m.current != 0) return false;
    return true;
}

static std::string signedStr(int v) {
    return (v > 0 ? "+" : "") + std::to_string(v);
}

static void printStatus(Cmd cmd, int speed, const char* msg = "") {
    std::cout << "\r\033[K" << cmdName(cmd) << "  speed=" << speed << "% (max "
              << MAX_SPEED << "%)  ";
    for (const Motor& m : g_motors)
        std::cout << m.abbr << " " << signedStr(m.current) << "  ";
    std::cout << "[flip L:" << (g_motors[0].invert ? "on" : "off")
              << " R:" << (g_motors[1].invert ? "on" : "off") << "]  "
              << msg << std::flush;
}

int main() {
    g_h = lgGpiochipOpen(0);  // gpiochip0 on Raspberry Pi 3
    if (g_h < 0) {
        std::cerr << "Failed to open gpiochip0 (error " << g_h
                  << "). Try running with sudo.\n";
        return 1;
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    // Claim every pin as an output, starting low
    for (const Motor& m : g_motors) {
        const int pins[3] = {m.pwm, m.ina, m.inb};
        for (int pin : pins) {
            int rc = lgGpioClaimOutput(g_h, 0, pin, 0);
            if (rc < 0) {
                std::cerr << "Failed to claim GPIO" << pin << " (" << m.name
                          << ", error " << rc << ").\n"
                          << "Is another program using it, or is SPI/1-Wire "
                             "enabled on that pin?\n";
                lgGpiochipClose(g_h);
                return 1;
            }
        }
    }

    std::cout << "VNH5019 4-motor drive teleop\n\n"
              << "  W / S   : forward / backward\n"
              << "  A / D   : rotate left / right in place\n"
              << "  Space   : STOP (also X)\n"
              << "  Up / Down (or + / -) : speed +/-" << SPEED_STEP << "%\n"
              << "  Z / C   : flip left / right polarity (while stopped)\n"
              << "  Q       : quit\n\n"
              << "Commands are latched: the robot keeps moving until you stop it.\n\n";

    RawTerminal term;
    term.enable();

    Cmd cmd = Cmd::Stop;
    int speed = INITIAL_SPEED;

    emergencyStop();
    printStatus(cmd, speed);

    auto lastRamp = std::chrono::steady_clock::now();

    while (!g_stop) {
        Key key = readKey();  // waits up to 0.1 s
        if (key == Key::Quit) break;

        bool dirty = (key != Key::None);
        const char* msg = "";

        switch (key) {
            case Key::Forward:  cmd = Cmd::Forward;     updateTargets(cmd, speed); break;
            case Key::Backward: cmd = Cmd::Backward;    updateTargets(cmd, speed); break;
            case Key::RotLeft:  cmd = Cmd::RotateLeft;  updateTargets(cmd, speed); break;
            case Key::RotRight: cmd = Cmd::RotateRight; updateTargets(cmd, speed); break;

            case Key::Stop:
                cmd = Cmd::Stop;
                emergencyStop();
                break;

            case Key::SpeedUp:
                if (speed < MAX_SPEED) speed = std::min(speed + SPEED_STEP, MAX_SPEED);
                else msg = "(at max)";
                updateTargets(cmd, speed);
                break;

            case Key::SpeedDown:
                if (speed > 0) speed = std::max(speed - SPEED_STEP, 0);
                else msg = "(at min)";
                updateTargets(cmd, speed);
                break;

            case Key::FlipLeft:
            case Key::FlipRight: {
                if (cmd != Cmd::Stop || !allStopped()) {
                    msg = "(stop first)";
                } else {
                    bool left = (key == Key::FlipLeft);
                    for (Motor& m : g_motors)
                        if (m.leftSide == left) m.invert = !m.invert;
                }
                break;
            }

            default:
                break;
        }

        // Time-based ramping so acceleration is the same however fast keys arrive
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastRamp).count();
        long ticks = elapsedMs / RAMP_INTERVAL_MS;
        if (ticks > 0) {
            lastRamp += std::chrono::milliseconds(ticks * RAMP_INTERVAL_MS);
            int step = static_cast<int>(ticks) * RAMP_STEP;
            for (Motor& m : g_motors) {
                int next = stepToward(m.current, m.target, step);
                if (next != m.current) {
                    m.current = next;
                    writeOutput(m);
                    dirty = true;
                }
            }
        }

        if (dirty) printStatus(cmd, speed, msg);
    }

    emergencyStop();
    lgGpiochipClose(g_h);
    std::cout << "\nAll motors stopped. Bye.\n";
    return 0;
}
