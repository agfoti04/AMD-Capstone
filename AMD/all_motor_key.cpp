// Keyboard teleop for four VNH5019 motor drivers, Raspberry Pi 3 (C++ / lgpio)
//
// Pins are BCM / GPIO numbers (NOT physical header pin numbers):
//
// 1: Front Left
//   GPIO 4  -> INA
//   GPIO 17 -> INB
//   GPIO 27 -> PWM
// 2: Front Right
//   GPIO 24 -> PWM
//   GPIO 22 -> INA
//   GPIO 23 -> INB
// 3: Back Left
//   GPIO 11 -> PWM
//   GPIO 10 -> INA
//   GPIO 9  -> INB
// 4: Back Right
//   GPIO 7  -> PWM
//   GPIO 25 -> INA
//   GPIO 8  -> INB
//
// Controls:
//   1 / 2 / 3 / 4      : select motor (the previously selected motor is stopped)
//   Up arrow   (or W)  : increase duty cycle of the selected motor
//   Down arrow (or S)  : decrease duty cycle of the selected motor
//   Space              : selected motor ON / OFF (OFF = brake)
//   R                  : toggle direction of selected motor (only while it is OFF)
//   X                  : stop ALL motors immediately
//   Q (or Ctrl+C)      : quit
//
// All motors start OFF. Only one motor runs at a time: selecting a different
// motor brakes the one you were on. Each motor remembers its own duty and
// direction settings.
//
// Install: sudo apt install liblgpio-dev g++
// Build:   g++ -o vnh5019_teleop vnh5019_teleop.cpp -llgpio
// Run:     ./vnh5019_teleop   (run it in a real terminal; use sudo if needed)

#include <lgpio.h>
#include <termios.h>
#include <unistd.h>

#include <csignal>
#include <iostream>

// ---- Settings (adjust to taste) ----
constexpr float PWM_FREQ_HZ = 1000.0f;
constexpr int INITIAL_DUTY = 2;  // percent
constexpr int MAX_DUTY = 30;     // safety cap in percent; raise it when you're ready
constexpr int DUTY_STEP = 1;     // percent per key press

// ---- Motors ----
struct Motor {
    const char* name;
    int pwm;
    int ina;
    int inb;
    bool running = false;
    bool forward = true;
    int duty = INITIAL_DUTY;
};

constexpr int NUM_MOTORS = 4;

static Motor g_motors[NUM_MOTORS] = {
    {"Front Left",  27, 4,  17},  // {name, PWM, INA, INB}
    {"Front Right", 24, 22, 23},
    {"Back Left",   11, 10, 9},
    {"Back Right",  7,  25, 8},
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

enum class Key { None, Up, Down, Space, Reverse, StopAll, Quit,
                 Select1, Select2, Select3, Select4 };

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
        case 'x': case 'X': return Key::StopAll;
        case 'q': case 'Q': return Key::Quit;
        case '1':           return Key::Select1;
        case '2':           return Key::Select2;
        case '3':           return Key::Select3;
        case '4':           return Key::Select4;
        default:            return Key::None;
    }
}

// ---- Motor helpers ----
static void applyDuty(const Motor& m) {
    // pwmCycles = 0 means run until changed
    lgTxPwm(g_h, m.pwm, PWM_FREQ_HZ, static_cast<float>(m.duty), 0, 0);
}

// PWM off, INA = INB = 0 (brake to ground).
static void brake(Motor& m) {
    m.running = false;
    lgTxPwm(g_h, m.pwm, PWM_FREQ_HZ, 0.0f, 0, 0);
    lgGpioWrite(g_h, m.pwm, 0);
    lgGpioWrite(g_h, m.ina, 0);
    lgGpioWrite(g_h, m.inb, 0);
}

static void brakeAll() {
    for (Motor& m : g_motors) brake(m);
}

static void applyDirection(const Motor& m) {
    lgGpioWrite(g_h, m.ina, m.forward ? 1 : 0);
    lgGpioWrite(g_h, m.inb, m.forward ? 0 : 1);
}

static void printStatus(int sel, const char* msg = "") {
    const Motor& m = g_motors[sel];
    std::cout << "\r\033[K"
              << "Motor " << (sel + 1) << " (" << m.name << ")  "
              << (m.running ? "[ ON ] " : "[ OFF] ")
              << (m.forward ? "FORWARD  " : "REVERSE  ")
              << "duty = " << m.duty << "% (max " << MAX_DUTY << "%)  "
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

    std::cout << "VNH5019 4-motor keyboard teleop\n\n";
    for (int i = 0; i < NUM_MOTORS; ++i) {
        const Motor& m = g_motors[i];
        std::cout << "  " << (i + 1) << " : " << m.name
                  << "  (PWM=GPIO" << m.pwm << " INA=GPIO" << m.ina
                  << " INB=GPIO" << m.inb << ")\n";
    }
    std::cout << "\n"
              << "  1-4     : select motor (stops the previous one)\n"
              << "  Up/W    : duty +" << DUTY_STEP << "%\n"
              << "  Down/S  : duty -" << DUTY_STEP << "%\n"
              << "  Space   : selected motor ON/OFF\n"
              << "  R       : toggle direction (motor must be OFF)\n"
              << "  X       : stop all motors\n"
              << "  Q       : quit\n\n";

    RawTerminal term;
    term.enable();

    int sel = 0;  // currently selected motor index
    brakeAll();
    printStatus(sel);

    while (!g_stop) {
        Key key = readKey();
        if (key == Key::None) continue;
        if (key == Key::Quit) break;

        const char* msg = "";
        Motor& m = g_motors[sel];

        switch (key) {
            case Key::Select1:
            case Key::Select2:
            case Key::Select3:
            case Key::Select4: {
                int newSel = static_cast<int>(key) - static_cast<int>(Key::Select1);
                if (newSel != sel) {
                    if (m.running) {
                        brake(m);
                        msg = "(previous motor stopped)";
                    }
                    sel = newSel;
                }
                break;
            }

            case Key::Up:
                if (m.duty < MAX_DUTY) m.duty += DUTY_STEP;
                else msg = "(at max)";
                if (m.duty > MAX_DUTY) m.duty = MAX_DUTY;
                if (m.running) applyDuty(m);
                break;

            case Key::Down:
                if (m.duty > 0) m.duty -= DUTY_STEP;
                else msg = "(at min)";
                if (m.duty < 0) m.duty = 0;
                if (m.running) applyDuty(m);
                break;

            case Key::Space:
                if (!m.running) {
                    m.running = true;
                    applyDirection(m);
                    applyDuty(m);
                } else {
                    brake(m);
                }
                break;

            case Key::Reverse:
                if (m.running) {
                    msg = "(stop the motor first)";
                } else {
                    m.forward = !m.forward;
                }
                break;

            case Key::StopAll:
                brakeAll();
                msg = "(all motors stopped)";
                break;

            default:
                break;
        }

        printStatus(sel, msg);
    }

    brakeAll();
    lgGpiochipClose(g_h);
    std::cout << "\nAll motors stopped. Bye.\n";
    return 0;
}
