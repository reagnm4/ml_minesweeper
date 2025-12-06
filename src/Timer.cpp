#include "Timer.h"

Timer::Timer() {}

void Timer::start() {
    running = true;
    paused = false;
    startTime = std::chrono::high_resolution_clock::now();
}

void Timer::pause() {
    if (!paused) {
        pausedTime = getSeconds();
        paused = true;
    }
}

void Timer::resume() {
    if (paused) {
        paused = false;
        startTime = std::chrono::high_resolution_clock::now();
    }
}

void Timer::reset() {
    pausedTime = 0;
    paused = false;
    running = false;
}

int Timer::getSeconds() const {
    if (!running) return 0;
    if (paused) return pausedTime;

    auto now = std::chrono::high_resolution_clock::now();
    return pausedTime + std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
}

int Timer::getMinutes() const {
    return getSeconds() / 60;
}

