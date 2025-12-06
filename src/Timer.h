#pragma once
#include <SFML/Graphics.hpp>
#include <chrono>

class Timer {
public:
    Timer();
    void start();
    void pause();
    void resume();
    void reset();
    int getSeconds() const;
    int getMinutes() const;

private:
    bool running = false;
    bool paused = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    int pausedTime = 0; // seconds
};
