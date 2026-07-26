#pragma once
#include <SFML/Graphics.hpp>
#include <string>
#include "Board.h"
#include "Button.h"
#include "Timer.h"
#include "Number.h"
#include "Leaderboard.h"
#include "Textures.h"
#include "Config.h"

enum class GameState {
    Playing,
    Won,
    Lost
};

class GameWindow {
public:
    GameWindow(unsigned int w, unsigned int h, const std::string& playerName, const Config& cfg);
    void run();

private:
    void processEvents();
    void update();
    void draw();
    void resetGame();
    bool loadTexture(sf::Texture &tex, const std::string &path);
    void positionFaceButton();
    void drawMineCounter();
    void drawTimer();

    unsigned int width;
    unsigned int height;
    std::string player;
    Config config;
    TextureManager texManager;

    sf::RenderWindow window;

    std::unique_ptr<Board> board;
    Timer timer;

    Button faceButton;
    Button debugButton;
    Button pauseButton;
    Button leaderboardButton;

    Number mineCounter;
    Number timeCounter;

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    int elapsedSeconds = 0;
    bool timerRunning = false;

    bool paused = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> pauseStart;
    int pausedTime = 0;

    int flagsPlaced = 0;
    int mineCountRemaining = 0;

    bool debugMode = false;

    std::unique_ptr<Leaderboard> leaderboard;
    GameState state = GameState::Playing;

};


