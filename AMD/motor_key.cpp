// Keyboard-controlled VNH5019 motor test for Raspberry Pi 3 (C++ / lgpio)
//
// Pins are BCM / GPIO numbers (NOT physical header pin numbers):
//   GPIO21 -> PWM
//   GPIO22 -> INA
//   GPIO23 -> INB
//
// Controls:
//   Up arrow   (or W)  : increase duty cycle
//   Down arrow (or S)  : decrease duty cycle
//   Space              : motor ON / OFF (OFF = brake)
//   R                  : toggle direction (only while motor is OFF)
//   Q (or Ctrl+C)      : quit
//
// The motor always starts OFF. Duty can be adjusted while OFF; it takes
// effect when you press Space.
//
// Install: sudo apt install liblgpio-dev g++
// Build:   g++ -o vnh5019_keyboard vnh5019_keyboard.cpp -llgpio
// Run:     ./vnh5019_keyboard   (run it in a real terminal; use sudo if needed)

#include <lgpio.h>
#include <termios.h>
#include <unistd.h>

#include <csignal>
#include <iostream>

// ---- Pins (BCM numbering) ----
constexpr int PWM_PIN = 13;
constexpr int INA_PIN = 6;
constexpr int INB_PIN = 5;

// ---- Settings (adjust to taste) ----
constexpr float PWM_FREQ_HZ = 1000.0f;
constexpr int INITIAL_DUTY = 2;  // percent
constexpr int MAX_DUTY = 30;     // safety cap in percent; raise it when you're ready
constexpr int DUTY_STEP = 1;     // percent per key press

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

enum class Key { None, Up, Down, Space, Reverse, Quit };

static Key readKey() {
    char c;
    if (read(STDIN_FILENO, &c, 1) <= 0) return Key::None;

    if (c == 27) {  // escape sequence, e.g. arrow keys: ESC [ A
        char seq[2];
        if (read(STDIN_FILENO, &seq[0], 1) <= 0) return Key::None;
        if (read(STDIN_FILENO, &seq[1], 1) <= 0) return Key::None;
        if (seq[0] == '[') {
            if (seq[1] == 'A') return Key::Up;
            if (seq[1] == 'B') return Key::Down;
        }
        return Key::None;
    }

    switch (c) {
        case 'w': case 'W': return Key::Up;
        case 's': case 'S': return Key::Down;
        case ' ':           return Key::Space;
        case 'r': case 'R': return Key::Reverse;
        case 'q': case 'Q': return Key::Quit;
        default:            return Key::None;
    }
}

// ---- Motor helpers ----
static void applyDuty(int percent) {
    // pwmCycles = 0 means run until changed
    lgTxPwm(g_h, PWM_PIN, PWM_FREQ_HZ, static_cast<float>(percent), 0, 0);
}

// PWM off, INA = INB = 0 (brake to ground).
static void brake() {
    lgTxPwm(g_h, PWM_PIN, PWM_FREQ_HZ, 0.0f, 0, 0);
    lgGpioWrite(g_h, PWM_PIN, 0);
    lgGpioWrite(g_h, INA_PIN, 0);
    lgGpioWrite(g_h, INB_PIN, 0);
}

static void applyDirection(bool forward) {
    lgGpioWrite(g_h, INA_PIN, forward ? 1 : 0);
    lgGpioWrite(g_h, INB_PIN, forward ? 0 : 1);
}

static void printStatus(bool running, bool forward, int duty, const char* msg = "") {
    std::cout << "\r\033[K"
              << (running ? "[ ON ] " : "[ OFF] ")
              << (forward ? "FORWARD  " : "REVERSE  ")
              << "duty = " << duty << "% (max " << MAX_DUTY << "%)  "
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

    if (lgGpioClaimOutput(g_h, 0, PWM_PIN, 0) < 0 ||
        lgGpioClaimOutput(g_h, 0, INA_PIN, 0) < 0 ||
        lgGpioClaimOutput(g_h, 0, INB_PIN, 0) < 0) {
        std::cerr << "Failed to claim GPIO pins. Is another program using them?\n";
        lgGpiochipClose(g_h);
        return 1;
    }

    std::cout << "VNH5019 keyboard control\n"
              << "PWM=GPIO" << PWM_PIN << "  INA=GPIO" << INA_PIN
              << "  INB=GPIO" << INB_PIN << "\n\n"
              << "  Up/W    : duty +" << DUTY_STEP << "%\n"
              << "  Down/S  : duty -" << DUTY_STEP << "%\n"
              << "  Space   : motor ON/OFF\n"
              << "  R       : toggle direction (motor must be OFF)\n"
              << "  Q       : quit\n\n";

    RawTerminal term;
    term.enable();

    bool running = false;
    bool forward = true;
    int duty = INITIAL_DUTY;

    brake();
    printStatus(running, forward, duty);

    while (!g_stop) {
        Key key = readKey();
        if (key == Key::None) continue;
        if (key == Key::Quit) break;

        const char* msg = "";

        switch (key) {
            case Key::Up:
                if (duty < MAX_DUTY) duty += DUTY_STEP;
                else msg = "(at max)";
                if (duty > MAX_DUTY) duty = MAX_DUTY;
                if (running) applyDuty(duty);
                break;

            case Key::Down:
                if (duty > 0) duty -= DUTY_STEP;
                else msg = "(at min)";
                if (duty < 0) duty = 0;
                if (running) applyDuty(duty);
                break;

            case Key::Space:
                running = !running;
                if (running) {
                    applyDirection(forward);
                    applyDuty(duty);
                } else {
                    brake();
                }
                break;

            case Key::Reverse:
                if (running) {
                    msg = "(stop the motor first)";
                } else {
                    forward = !forward;
                }
                break;

            default:
                break;
        }

        printStatus(running, forward, duty, msg);
    }

    brake();
    lgGpiochipClose(g_h);
    std::cout << "\nMotor stopped. Bye.\n";
    return 0;
}
