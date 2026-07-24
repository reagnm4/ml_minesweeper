#include "Board.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>

Board::Board(int cols, int rows, int mines, TextureManager* tex) : columns(cols), rows(rows), mineCount(mines), texManager(tex) {
    // One seed per Board, drawn from the OS entropy source.
    std::random_device rd;
    rng.seed(rd());

    // A config file asking for more mines than there are cells used to spin
    // forever inside the placement loop. Clamp instead of hanging.
    if (columns < 0) columns = 0;
    if (rows < 0) rows = 0;
    mineCount = std::max(0, std::min(mineCount, columns * rows));

    grid.resize(rows);
    for (int y = 0; y < rows; y++) {
        grid[y].resize(columns);
    }
    reset();
}

void Board::reset() {
    lost = false;
    won = false;
    minesPlaced = false;

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

    linkNeighbors();
    // Mines are placed by the first revealTile() call -- see placeMines().
}

void Board::placeMines(int safeX, int safeY) {
    // Everything except the clicked cell and its neighbours is fair game. That
    // is what makes the first click safe, and it guarantees the first click
    // opens a zero so the player gets a region instead of a lone number.
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(columns) * rows);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            if (std::abs(x - safeX) <= 1 && std::abs(y - safeY) <= 1) continue;
            candidates.push_back(y * columns + x);
        }
    }

    // Too crowded to keep the whole neighbourhood clear: protect just the
    // clicked cell, then give up on that too rather than place fewer mines.
    if (static_cast<int>(candidates.size()) < mineCount) {
        candidates.clear();
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < columns; x++)
                if (x != safeX || y != safeY) candidates.push_back(y * columns + x);
    }
    if (static_cast<int>(candidates.size()) < mineCount) {
        candidates.resize(static_cast<size_t>(columns) * rows);
        std::iota(candidates.begin(), candidates.end(), 0);
    }

    // Partial Fisher-Yates: uniform over layouts and always terminates. The
    // old rejection-sampling loop did neither at high mine densities.
    const int k = std::min<int>(mineCount, static_cast<int>(candidates.size()));
    for (int i = 0; i < k; i++) {
        std::uniform_int_distribution<int> pick(i, static_cast<int>(candidates.size()) - 1);
        std::swap(candidates[i], candidates[pick(rng)]);
        const int cell = candidates[i];
        grid[cell / columns][cell % columns].setMine(true);
    }

    minesPlaced = true;
    calculateAdjacency();
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
    if (lost || won) return;

    Tile& first = grid[y][x];
    if (first.isRevealed() || first.isFlagged()) return;

    if (!minesPlaced) placeMines(x, y);

    if (first.isMine()) {
        first.setRevealed(true);
        lost = true;
        return;
    }

    // Iterative flood fill. The recursive version overflowed the stack on
    // boards of roughly 300x300 and larger.
    std::vector<Tile*> stack;
    stack.push_back(&first);

    while (!stack.empty()) {
        Tile* t = stack.back();
        stack.pop_back();

        if (t->isRevealed() || t->isFlagged() || t->isMine()) continue;
        t->setRevealed(true);

        if (t->getAdjacentMines() != 0) continue;

        for (Tile* n : t->getNeighbors()) {
            if (!n->isRevealed() && !n->isFlagged()) stack.push_back(n);
        }
    }
}

void Board::flagTile(int x, int y) {
    if (!isInBounds(x, y)) return;
    if (lost || won) return;
    Tile& t = grid[y][x];
    if (t.isRevealed()) return;
    t.setFlagged(!t.isFlagged());
}


bool Board::checkWin() const {
    if (lost) return false;
    if (!minesPlaced) return false;  // nothing has been revealed yet

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
    // Textures that failed to load come back as nullptr; dereferencing them to
    // build a sprite is an immediate segfault, which is what happened when the
    // game was launched from a directory without files/images/.
    sf::Texture* hiddenTex   = texManager->get("hidden");
    sf::Texture* revealedTex = texManager->get("revealed");
    sf::Texture* mineTex     = texManager->get("mine");
    sf::Texture* flagTex     = texManager->get("flag");

    auto blit = [&window](sf::Texture* tex, float px, float py) {
        if (!tex) return;
        sf::Sprite sprite(*tex);
        sprite.setPosition({px, py});
        window.draw(sprite);
    };

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            Tile& t = grid[y][x];
            const float px = x * 32.f;
            const float py = y * 32.f;

            if (paused) {
                blit(revealedTex, px, py);
                continue;
            }

            if (!t.isRevealed()) {
                blit(hiddenTex, px, py);
                if (t.isFlagged()) blit(flagTex, px, py);
                if (debugMode && t.isMine()) blit(mineTex, px, py);
                continue;
            }

            blit(revealedTex, px, py);

            if (t.isMine()) {
                blit(mineTex, px, py);
                continue;
            }

            int adj = t.getAdjacentMines();
            if (adj > 0) {
                blit(texManager->get("num" + std::to_string(adj)), px, py);
            }
        }
    }
}
