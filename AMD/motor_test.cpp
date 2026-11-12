// Basic VNH5019 motor driver test for Raspberry Pi 3 (C++ / lgpio)
//
// Pins are BCM / GPIO numbers (NOT physical header pin numbers):
//   GPIO21 -> PWM
//   GPIO22 -> INA
//   GPIO23 -> INB
//
// Sequence: slow ramp forward -> stop -> slow ramp reverse -> stop.
// Press Ctrl+C at any time to stop the motor immediately.
//
// Install: sudo apt install liblgpio-dev g++
// Build:   g++ -o vnh5019_test vnh5019_test_lgpio.cpp -llgpio
// Run:     ./vnh5019_test        (try sudo ./vnh5019_test if you get a permission error)

#include <lgpio.h>

#include <csignal>
#include <iostream>
#include <string>

// ---- Pins (BCM numbering) ----
constexpr int PWM_PIN = 23;
constexpr int INA_PIN = 21;
constexpr int INB_PIN = 22;

// ---- Test settings (adjust to taste) ----
constexpr float PWM_FREQ_HZ = 1000.0f;
constexpr int START_DUTY = 2;       // percent
constexpr int MAX_DUTY = 15;        // never exceed 15% in this basic test
constexpr int DUTY_STEP = 1;        // percent per step
constexpr double STEP_TIME_S = 1.5; // time at each step

static volatile std::sig_atomic_t g_stop = 0;
static int g_h = -1;  // lgpio chip handle

static void onSignal(int) { g_stop = 1; }

// Sleep in small chunks so Ctrl+C is noticed quickly.
static void sleepSec(double sec) {
    while (sec > 0 && !g_stop) {
        double chunk = sec > 0.05 ? 0.05 : sec;
        lguSleep(chunk);
        sec -= chunk;
    }
}

static void setDuty(int percent) {
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

static void setDirection(bool forward) {
    lgGpioWrite(g_h, INA_PIN, forward ? 1 : 0);
    lgGpioWrite(g_h, INB_PIN, forward ? 0 : 1);
}

static void ramp(bool forward) {
    const char* name = forward ? "FORWARD" : "REVERSE";
    setDirection(forward);

    // Ramp up slowly
    for (int duty = START_DUTY; duty <= MAX_DUTY && !g_stop; duty += DUTY_STEP) {
        setDuty(duty);
        std::cout << name << ": duty = " << duty << "%\n";
        sleepSec(STEP_TIME_S);
    }

    // Ramp back down before stopping or reversing
    for (int duty = MAX_DUTY; duty >= START_DUTY && !g_stop; duty -= DUTY_STEP) {
        setDuty(duty);
        std::cout << name << ": duty = " << duty << "% (slowing)\n";
        sleepSec(STEP_TIME_S / 2);
    }

    brake();
    if (!g_stop) {
        std::cout << name << " done, braking.\n\n";
        sleepSec(2.0);  // let the motor fully stop before changing direction
    }
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

    // Claim pins as outputs, all starting low
    if (lgGpioClaimOutput(g_h, 0, PWM_PIN, 0) < 0 ||
        lgGpioClaimOutput(g_h, 0, INA_PIN, 0) < 0 ||
        lgGpioClaimOutput(g_h, 0, INB_PIN, 0) < 0) {
        std::cerr << "Failed to claim GPIO pins. Is another program using them?\n";
        lgGpiochipClose(g_h);
        return 1;
    }

    std::cout << "VNH5019 basic test\n";
    std::cout << "PWM=GPIO" << PWM_PIN << "  INA=GPIO" << INA_PIN
              << "  INB=GPIO" << INB_PIN << "\n";
    std::cout << "Duty range: " << START_DUTY << "% to " << MAX_DUTY << "%\n";
    std::cout << "Check wiring, keep clear of the motor, then press Enter to start...";
    std::string dummy;
    std::getline(std::cin, dummy);

    if (!g_stop) ramp(true);
    if (!g_stop) ramp(false);

    if (g_stop) {
        std::cout << "\nInterrupted, stopping motor.\n";
    } else {
        std::cout << "Test complete.\n";
    }

    brake();
    lgGpiochipClose(g_h);
    return 0;
}
