#include "Game.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iostream>

GameWindow::GameWindow(unsigned int w, unsigned int h, const std::string& playerName, const Config& cfg) : width(w), height(h), player(playerName), config(cfg), window(sf::VideoMode({w, h}), "Minesweeper", sf::Style::Close) {
    window.setFramerateLimit(60);

    texManager.load("hidden", "files/images/tile_hidden.png");
    texManager.load("revealed", "files/images/tile_revealed.png");
    texManager.load("mine", "files/images/mine.png");
    texManager.load("flag", "files/images/flag.png");
    texManager.load("digits", "files/images/digits.png");
    texManager.load("num1", "files/images/number_1.png");
    texManager.load("num2", "files/images/number_2.png");
    texManager.load("num3", "files/images/number_3.png");
    texManager.load("num4", "files/images/number_4.png");
    texManager.load("num5", "files/images/number_5.png");
    texManager.load("num6", "files/images/number_6.png");
    texManager.load("num7", "files/images/number_7.png");
    texManager.load("num8", "files/images/number_8.png");
    texManager.load("face_happy", "files/images/face_happy.png");
    texManager.load("face_win", "files/images/face_win.png");
    texManager.load("face_lose", "files/images/face_lose.png");
    texManager.load("debug", "files/images/debug.png");
    texManager.load("pause", "files/images/pause.png");
    texManager.load("play", "files/images/play.png");
    texManager.load("leaderboard", "files/images/leaderboard.png");

    board = std::make_unique<Board>(config.columns, config.rows, config.mines, &texManager);
    mineCountRemaining = config.mines;
    flagsPlaced = 0;
    leaderboard = std::make_unique<Leaderboard>("files/leaderboard.txt");

    faceButton.setTexture(texManager.get("face_happy"));
    positionFaceButton();
    debugButton.setTexture(texManager.get("debug"));
    debugButton.setPosition((config.columns * 32.f) - 304.f, 32.f * (config.rows + 0.5f));
    pauseButton.setTexture(texManager.get("pause"));
    pauseButton.setPosition((config.columns * 32.f) - 240.f, 32.f * (config.rows + 0.5f));
    leaderboardButton.setTexture(texManager.get("leaderboard"));
    leaderboardButton.setPosition((config.columns * 32.f) - 176.f, 32.f * (config.rows + 0.5f));
}

bool GameWindow::loadTexture(sf::Texture &tex, const std::string &path) {
    if (!tex.loadFromFile(path)) {
        std::cerr << "Failed to load texture: " << path << std::endl;
        return false;
    }
    return true;
}


void GameWindow::run() {
    elapsedSeconds = 0;
    startTime = std::chrono::high_resolution_clock::now();
    timerRunning = true;

    while (window.isOpen()) {
        processEvents();
        update();
        draw();
    }
}

void GameWindow::resetGame() {
    board->reset();
    state = GameState::Playing;
    faceButton.setTexture(texManager.get("face_happy"));
    positionFaceButton();

    startTime = std::chrono::high_resolution_clock::now();
    elapsedSeconds = 0;
    timerRunning = true;

    flagsPlaced = 0;
    mineCountRemaining = config.mines;
}

void GameWindow::drawMineCounter() {
    float x = 33;
    float y = 32 * (config.rows + 0.5f) + 16;
    int display = mineCountRemaining;
    sf::Texture* digits = texManager.get("digits");

    if (display < 0) {
        sf::Sprite minus(*digits);
        minus.setTextureRect(sf::IntRect({10 * 21, 0}, {21, 32}));
        minus.setPosition({x, y});
        window.draw(minus);

        display = -display;
        x += 21;
    }

    int hundreds = (display / 100) % 10;
    int tens     = (display / 10)  % 10;
    int ones     = display % 10;

    int arr[3] = { hundreds, tens, ones };

    for (int d : arr) {
        sf::Sprite digit(*digits);
        digit.setTextureRect(sf::IntRect({d * 21, 0}, {21, 32}));
        digit.setPosition({x, y});
        window.draw(digit);
        x += 21;
    }
}

void GameWindow::drawTimer() {
    sf::Texture* digits = texManager.get("digits");

    int minutes = (elapsedSeconds / 60) % 100;
    int seconds = elapsedSeconds % 60;
    float y = 32.f * (config.rows + 0.5f) + 16.f;
    float minX = (config.columns * 32.f) - 97.f;
    int minTens = (minutes / 10) % 10;
    int minOnes = minutes % 10;

    {
        sf::Sprite s(*digits);
        s.setTextureRect(sf::IntRect({minTens * 21, 0}, {21, 32}));
        s.setPosition({minX, y});
        window.draw(s);
    }
    {
        sf::Sprite s(*digits);
        s.setTextureRect(sf::IntRect({minOnes * 21, 0}, {21, 32}));
        s.setPosition({minX + 21, y});
        window.draw(s);
    }
    float secX = (config.columns * 32.f) - 54.f;
    int secTens = (seconds / 10) % 10;
    int secOnes = seconds % 10;
    {
        sf::Sprite s(*digits);
        s.setTextureRect(sf::IntRect({secTens * 21, 0}, {21, 32}));
        s.setPosition({secX, y});
        window.draw(s);
    }
    {
        sf::Sprite s(*digits);
        s.setTextureRect(sf::IntRect({secOnes * 21, 0}, {21, 32}));
        s.setPosition({secX + 21, y});
        window.draw(s);
    }
}

void GameWindow::positionFaceButton() {
    float x = (config.columns * 32.f) / 2.f - 32.f;
    float y = 32.f * (config.rows + 0.5f);
    faceButton.setPosition(x, y);
}

void GameWindow::processEvents() {
    while (auto event = window.pollEvent()) {

        if (event->is<sf::Event::Closed>()) {
            window.close();
        }

        if (auto mouse = event->getIf<sf::Event::MouseButtonPressed>()) {

            sf::Vector2i pos = sf::Mouse::getPosition(window);

            // Integer division truncates toward zero, so x in [-31, -1] used to
            // map to column 0 -- clicking just off the left/top edge revealed a
            // tile. Anything negative is off the board, full stop.
            int tileX = pos.x >= 0 ? pos.x / 32 : -1;
            int tileY = pos.y >= 0 ? pos.y / 32 : -1;

            if (!board) return;

            if (faceButton.wasClicked(pos)) {
                resetGame();
            }

            if (debugButton.wasClicked(pos)) {
                if (state != GameState::Playing)
                    return;
                debugMode = !debugMode;
            }

            if (pauseButton.wasClicked(pos)) {
                if (state != GameState::Playing)
                    return;
                paused = !paused;
                if (paused) {
                    sf::Vector2f pos = pauseButton.getSprite().getPosition();
                    pauseButton.setTexture(texManager.get("play"));
                    pauseButton.setPosition(pos.x, pos.y);
                    pauseStart = std::chrono::high_resolution_clock::now();
                    timerRunning = false;
                }
                else {
                    sf::Vector2f pos = pauseButton.getSprite().getPosition();
                    pauseButton.setTexture(texManager.get("pause"));
                    pauseButton.setPosition(pos.x, pos.y);
                    auto now = std::chrono::high_resolution_clock::now();
                    auto pauseDuration = now - pauseStart;
                    startTime += pauseDuration;
                    timerRunning = true;
                    paused = false;
                }
            }

            if (leaderboardButton.wasClicked(pos)) {
                paused = true;
                timerRunning = false;
                sf::Vector2f pos = pauseButton.getSprite().getPosition();
                pauseButton.setTexture(texManager.get("play"));
                pauseButton.setPosition(pos.x, pos.y);
                leaderboard->openWindow();
                return;
            }

            if (state == GameState::Playing) {
                if (mouse->button == sf::Mouse::Button::Left) {
                    board->revealTile(tileX, tileY);
                }
                if (mouse->button == sf::Mouse::Button::Right) {
                    if (tileX < 0 || tileY < 0 || tileX >= (int)config.columns || tileY >= (int)config.rows)
                        return;

                    Tile& tile = board->getTile(tileX, tileY);
                    bool wasFlagged = tile.isFlagged();
                    board->flagTile(tileX, tileY);
                    bool isFlagged = tile.isFlagged();

                    if (!wasFlagged && isFlagged) flagsPlaced++;
                    else if (wasFlagged && !isFlagged) flagsPlaced--;

                    mineCountRemaining = config.mines - flagsPlaced;
                }
            }
        }
    }
}

void GameWindow::update() {
    if (!board) return;

    if (state == GameState::Playing) {
        if (board->checkLoss()) {
            state = GameState::Lost;
            faceButton.setTexture(texManager.get("face_lose"));
            timerRunning = false;
        }
        else if (board->checkWin()) {
            state = GameState::Won;
            faceButton.setTexture(texManager.get("face_win"));
            timerRunning = false;
            leaderboard->addScore(player, elapsedSeconds);
        }
    }

    if (timerRunning && !paused && state == GameState::Playing) {
        auto now = std::chrono::high_resolution_clock::now();
        elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
    }

    positionFaceButton();
}

void GameWindow::draw() {
    window.clear(sf::Color::White);
    board->draw(window, debugMode, paused);
    window.draw(faceButton.getSprite());
    window.draw(debugButton.getSprite());
    window.draw(pauseButton.getSprite());
    window.draw(leaderboardButton.getSprite());
    drawMineCounter();
    drawTimer();
    window.display();
}
