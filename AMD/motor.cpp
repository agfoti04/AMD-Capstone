// VNH5019 motor test - Raspberry Pi 3 / C++ / lgpio
//
// BCM GPIO numbering:
//
//   GPIO21 -> VNH5019 PWM
//   GPIO22 -> VNH5019 INA
//   GPIO23 -> VNH5019 INB
//
// Controls:
//   W / Up      increase duty
//   S / Down    decrease duty
//   Space       motor ON/OFF
//   R           reverse direction (only while OFF)
//   T           TEST: 100% duty for 2 seconds
//   Q           quit
//
// Build:
//   g++ -Wall -Wextra -o vnh5019_test vnh5019_test.cpp -llgpio
//
// Run:
//   sudo ./vnh5019_test

#include <lgpio.h>
#include <termios.h>
#include <unistd.h>

#include <csignal>
#include <iostream>
#include <cstring>

// ============================================================
// GPIO CONFIGURATION
// ============================================================

constexpr int PWM_PIN = 23;
constexpr int INA_PIN = 21;
constexpr int INB_PIN = 22;

constexpr float PWM_FREQ_HZ = 1000.0f;

constexpr int INITIAL_DUTY = 0;
constexpr int MAX_DUTY     = 30;
constexpr int DUTY_STEP    = 1;

static int g_h = -1;
static volatile std::sig_atomic_t g_stop = 0;

static void onSignal(int)
{
    g_stop = 1;
}


// ============================================================
// TERMINAL
// ============================================================

struct RawTerminal
{
    termios original{};
    bool active = false;

    void enable()
    {
        if (tcgetattr(STDIN_FILENO, &original) != 0)
            return;

        termios raw = original;

        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN]  = 0;
        raw.c_cc[VTIME] = 1;

        tcsetattr(STDIN_FILENO, TCSANOW, &raw);

        active = true;
    }

    ~RawTerminal()
    {
        if (active)
            tcsetattr(STDIN_FILENO, TCSANOW, &original);
    }
};


enum class Key
{
    None,
    Up,
    Down,
    Space,
    Reverse,
    Test,
    Quit
};


static Key readKey()
{
    char c;

    if (read(STDIN_FILENO, &c, 1) <= 0)
        return Key::None;

    // Arrow keys: ESC [ A / ESC [ B
    if (c == 27)
    {
        char seq[2];

        if (read(STDIN_FILENO, &seq[0], 1) <= 0)
            return Key::None;

        if (read(STDIN_FILENO, &seq[1], 1) <= 0)
            return Key::None;

        if (seq[0] == '[')
        {
            if (seq[1] == 'A')
                return Key::Up;

            if (seq[1] == 'B')
                return Key::Down;
        }

        return Key::None;
    }

    switch (c)
    {
        case 'w':
        case 'W':
            return Key::Up;

        case 's':
        case 'S':
            return Key::Down;

        case ' ':
            return Key::Space;

        case 'r':
        case 'R':
            return Key::Reverse;

        case 't':
        case 'T':
            return Key::Test;

        case 'q':
        case 'Q':
            return Key::Quit;

        default:
            return Key::None;
    }
}


// ============================================================
// GPIO HELPERS
// ============================================================

static bool gpioWriteChecked(int gpio, int value)
{
    int rc = lgGpioWrite(g_h, gpio, value);

    if (rc < 0)
    {
        std::cerr
            << "\nERROR: lgGpioWrite(GPIO"
            << gpio
            << ", "
            << value
            << ") failed: "
            << rc
            << "\n";

        return false;
    }

    return true;
}


static bool pwmChecked(float duty)
{
    int rc = lgTxPwm(
        g_h,
        PWM_PIN,
        PWM_FREQ_HZ,
        duty,
        0,
        0
    );

    if (rc < 0)
    {
        std::cerr
            << "\nERROR: lgTxPwm failed.\n"
            << "  GPIO     = " << PWM_PIN << "\n"
            << "  Frequency = " << PWM_FREQ_HZ << " Hz\n"
            << "  Duty      = " << duty << "%\n"
            << "  Error     = " << rc << "\n";

        return false;
    }

    return true;
}


// ============================================================
// MOTOR CONTROL
// ============================================================

static void stopMotor()
{
    //
    // PWM = 0 disables the H-bridge outputs.
    //
    // This is the safest "OFF" state.
    //
    pwmChecked(0.0f);

    gpioWriteChecked(INA_PIN, 0);
    gpioWriteChecked(INB_PIN, 0);
}


static void setDirection(bool forward)
{
    if (forward)
    {
        // INA = 1
        // INB = 0
        //
        // With PWM high:
        // OUTA = HIGH
        // OUTB = LOW
        //
        // Motor drives forward.

        gpioWriteChecked(INA_PIN, 1);
        gpioWriteChecked(INB_PIN, 0);
    }
    else
    {
        // Reverse

        gpioWriteChecked(INA_PIN, 0);
        gpioWriteChecked(INB_PIN, 1);
    }
}


static void setDuty(int duty)
{
    if (duty < 0)
        duty = 0;

    if (duty > MAX_DUTY)
        duty = MAX_DUTY;

    pwmChecked(static_cast<float>(duty));
}


// ============================================================
// STATUS
// ============================================================

static void printStatus(
    bool running,
    bool forward,
    int duty,
    const char* message = "")
{
    std::cout
        << "\r\033[K"
        << (running ? "[ ON ] " : "[ OFF] ")
        << (forward ? "FORWARD " : "REVERSE ")
        << "duty=" << duty << "%   "
        << message
        << std::flush;
}


// ============================================================
// 100% TEST
// ============================================================

static void fullPowerTest(bool forward)
{
    std::cout
        << "\n\n========================================\n"
        << "100% DUTY TEST\n"
        << "========================================\n";

    std::cout
        << "Direction: "
        << (forward ? "FORWARD" : "REVERSE")
        << "\n";

    std::cout
        << "Setting INA/INB...\n";

    setDirection(forward);

    usleep(100000);

    std::cout
        << "Applying 100% PWM for 2 seconds...\n"
        << "WATCH MOTOR + POWER SUPPLY CURRENT\n";

    if (!pwmChecked(100.0f))
    {
        std::cerr << "PWM command failed!\n";
        stopMotor();
        return;
    }

    usleep(2000000);

    std::cout
        << "Stopping motor.\n";

    stopMotor();

    std::cout
        << "========================================\n\n";
}


// ============================================================
// MAIN
// ============================================================

int main()
{
    std::cout
        << "========================================\n"
        << " VNH5019 Raspberry Pi Motor Test\n"
        << "========================================\n\n";

    std::cout
        << "PWM = GPIO" << PWM_PIN << "\n"
        << "INA = GPIO" << INA_PIN << "\n"
        << "INB = GPIO" << INB_PIN << "\n\n";

    // --------------------------------------------------------
    // Open GPIO chip
    // --------------------------------------------------------

    g_h = lgGpiochipOpen(0);

    if (g_h < 0)
    {
        std::cerr
            << "ERROR: Could not open gpiochip0.\n"
            << "Error code: "
            << g_h
            << "\n";

        return 1;
    }

    // --------------------------------------------------------
    // Signals
    // --------------------------------------------------------

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    // --------------------------------------------------------
    // Claim GPIOs
    // --------------------------------------------------------

    int rc;

    rc = lgGpioClaimOutput(g_h, 0, PWM_PIN, 0);

    if (rc < 0)
    {
        std::cerr
            << "ERROR: Could not claim PWM GPIO "
            << PWM_PIN
            << ". Error = "
            << rc
            << "\n";

        lgGpiochipClose(g_h);
        return 1;
    }

    rc = lgGpioClaimOutput(g_h, 0, INA_PIN, 0);

    if (rc < 0)
    {
        std::cerr
            << "ERROR: Could not claim INA GPIO "
            << INA_PIN
            << ". Error = "
            << rc
            << "\n";

        lgGpiochipClose(g_h);
        return 1;
    }

    rc = lgGpioClaimOutput(g_h, 0, INB_PIN, 0);

    if (rc < 0)
    {
        std::cerr
            << "ERROR: Could not claim INB GPIO "
            << INB_PIN
            << ". Error = "
            << rc
            << "\n";

        lgGpiochipClose(g_h);
        return 1;
    }

    // --------------------------------------------------------
    // Initial safe state
    // --------------------------------------------------------

    stopMotor();

    bool running = false;
    bool forward = true;
    int duty = INITIAL_DUTY;

    // --------------------------------------------------------
    // Terminal
    // --------------------------------------------------------

    RawTerminal term;
    term.enable();

    std::cout
        << "\nControls:\n"
        << "  W / Up       duty +1%\n"
        << "  S / Down     duty -1%\n"
        << "  Space        ON / OFF\n"
        << "  R            reverse direction\n"
        << "  T            100% duty test for 2 sec\n"
        << "  Q            quit\n\n";

    std::cout
        << "IMPORTANT:\n"
        << "  T = direct 100% motor test.\n"
        << "  If T produces NO current draw, the problem is\n"
        << "  almost certainly wiring / power / enable / GPIO,\n"
        << "  not the duty-cycle ramp.\n\n";

    printStatus(running, forward, duty);

    // --------------------------------------------------------
    // Main loop
    // --------------------------------------------------------

    while (!g_stop)
    {
        Key key = readKey();

        if (key == Key::None)
            continue;

        if (key == Key::Quit)
            break;

        const char* msg = "";

        switch (key)
        {
            // --------------------------------------------
            // Increase duty
            // --------------------------------------------

            case Key::Up:

                if (duty < MAX_DUTY)
                {
                    duty += DUTY_STEP;
                }
                else
                {
                    msg = "(MAX)";
                }

                if (running)
                    setDuty(duty);

                break;


            // --------------------------------------------
            // Decrease duty
            // --------------------------------------------

            case Key::Down:

                if (duty > 0)
                {
                    duty -= DUTY_STEP;
                }
                else
                {
                    msg = "(MIN)";
                }

                if (running)
                    setDuty(duty);

                break;


            // --------------------------------------------
            // ON / OFF
            // --------------------------------------------

            case Key::Space:

                if (!running)
                {
                    //
                    // Direction FIRST.
                    //
                    // Then PWM.
                    //
                    setDirection(forward);

                    usleep(10000);

                    if (setDuty, true)
                    {
                        if (!pwmChecked(static_cast<float>(duty)))
                        {
                            msg = "(PWM ERROR)";
                            running = false;
                            stopMotor();
                            break;
                        }
                    }

                    running = true;
                }
                else
                {
                    stopMotor();
                    running = false;
                }

                break;


            // --------------------------------------------
            // Reverse
            // --------------------------------------------

            case Key::Reverse:

                if (running)
                {
                    msg = "(STOP FIRST)";
                }
                else
                {
                    forward = !forward;
                }

                break;


            // --------------------------------------------
            // Full-power diagnostic test
            // --------------------------------------------

            case Key::Test:

                if (running)
                {
                    msg = "(STOP FIRST)";
                }
                else
                {
                    fullPowerTest(forward);
                }

                break;


            default:
                break;
        }

        printStatus(running, forward, duty, msg);
    }

    // --------------------------------------------------------
    // Shutdown
    // --------------------------------------------------------

    stopMotor();

    lgGpiochipClose(g_h);

    std::cout
        << "\n\nMotor stopped. GPIO released.\n";

    return 0;
}
