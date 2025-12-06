#include "Board.h"
#include <random>
#include <ctime>

Board::Board(int cols, int rows, int mines, TextureManager* tex) : columns(cols), rows(rows), mineCount(mines), texManager(tex) {
    grid.resize(rows);
    for (int y = 0; y < rows; y++) {
        grid[y].resize(columns);
    }
    reset();
}

void Board::reset() {
    lost = false;
    won = false;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            grid[y][x].setTexture(texManager->get("hidden"));
            grid[y][x].setPosition(x * 32.f, y * 32.f);
            grid[y][x].setMine(false);
            grid[y][x].setRevealed(false);
            grid[y][x].setFlagged(false);
            grid[y][x].setAdjacentMines(0);
            grid[y][x].setCoords(x, y);
        }
    }

    randomizeMines();
    linkNeighbors();
    calculateAdjacency();
}

void Board::randomizeMines() {
    std::mt19937 rng(static_cast<unsigned long>(std::time(nullptr)));
    std::uniform_int_distribution<int> distCol(0, columns - 1);
    std::uniform_int_distribution<int> distRow(0, rows - 1);

    int placed = 0;

    while (placed < mineCount) {
        int x = distCol(rng);
        int y = distRow(rng);

        if (!grid[y][x].isMine()) {
            grid[y][x].setMine(true);
            placed++;
        }
    }
}

void Board::linkNeighbors() {
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {

            Tile& current = grid[y][x];
            current.getNeighbors().clear();

            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];

                if (isInBounds(nx, ny)) {
                    current.addNeighbor(&grid[ny][nx]);
                }
            }
        }
    }
}

void Board::calculateAdjacency() {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {

            Tile& t = grid[y][x];

            if (t.isMine()) {
                t.setAdjacentMines(-1);
                continue;
            }

            int count = 0;
            for (Tile* n : t.getNeighbors()) {
                if (n->isMine())
                    count++;
            }

            t.setAdjacentMines(count);
        }
    }
}

Tile& Board::getTile(int x, int y) {
    return grid[y][x];
}

bool Board::isInBounds(int x, int y) const {
    return x >= 0 && x < columns && y >= 0 && y < rows;
}

void Board::revealTile(int x, int y) {
    if (!isInBounds(x, y)) return;
    Tile& t = grid[y][x];
    if (t.isRevealed() || t.isFlagged()) return;
    t.setRevealed(true);

    if (t.isMine()) {
        lost = true;
        return;
    }

    if (t.getAdjacentMines() == 0) {
        for (Tile* n : t.getNeighbors()) {
            revealTile(n->getX(), n->getY());
        }
    }
}

void Board::flagTile(int x, int y) {
    if (!isInBounds(x, y)) return;
    Tile& t = grid[y][x];
    if (t.isRevealed()) return;
    t.setFlagged(!t.isFlagged());
}


bool Board::checkWin() const {
    if (lost) return false;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            const Tile& t = grid[y][x];

            if (!t.isMine() && !t.isRevealed()) {
                return false;
            }
        }
    }

    return true;
}

bool Board::checkLoss() const {
    return lost;
}

void Board::draw(sf::RenderWindow& window, bool debugMode, bool paused) {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            Tile& t = grid[y][x];

            if (paused) {
                sf::Sprite base(*texManager->get("revealed"));
                base.setPosition({x * 32.f, y * 32.f});
                window.draw(base);
                continue;
            }

            if (!t.isRevealed()) {
                sf::Sprite hidden(*texManager->get("hidden"));
                hidden.setPosition({x * 32.f, y * 32.f});
                window.draw(hidden);
                if (t.isFlagged()) {
                    sf::Sprite flag(*texManager->get("flag"));
                    flag.setPosition({x * 32.f, y * 32.f});
                    window.draw(flag);
                }
                if (debugMode && t.isMine()) {
                    sf::Sprite mine(*texManager->get("mine"));
                    mine.setPosition({x * 32.f, y * 32.f});
                    window.draw(mine);
                }
                continue;
            }

            sf::Sprite base(*texManager->get("revealed"));
            base.setPosition({x * 32.f, y * 32.f});
            window.draw(base);

            if (t.isMine()) {
                sf::Sprite mine(*texManager->get("mine"));
                mine.setPosition({x * 32.f, y * 32.f});
                window.draw(mine);
                continue;
            }

            int adj = t.getAdjacentMines();
            if (adj > 0) {
                std::string key = "num" + std::to_string(adj);
                sf::Sprite num(*texManager->get(key));
                num.setPosition({x * 32.f, y * 32.f});
                window.draw(num);
            }
        }
    }
}

