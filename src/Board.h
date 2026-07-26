#pragma once
#include <SFML/Graphics.hpp>
#include <random>
#include <vector>
#include <string>
#include "Tile.h"
#include "Textures.h"

// NOTE: this is the board used by the SFML game. The ML side of the project
// uses the headless engine in engine/Minesweeper.h instead -- it has no
// graphics dependency and can be copied. The rules implemented here are kept
// deliberately identical.
class Board {
public:
    Board(int cols, int rows, int mines, TextureManager* tex);

    void reset();
    void revealTile(int x, int y);
    void flagTile(int x, int y);

    bool isInBounds(int x, int y) const;
    bool checkWin() const;
    bool checkLoss() const;

    void draw(sf::RenderWindow& window, bool debugMode, bool paused);

    // Caller must check isInBounds(x, y) first.
    Tile& getTile(int x, int y);

private:
    // Mines are placed on the first reveal of a game, not at reset(), so that
    // the first cell the player clicks is guaranteed to be safe.
    void placeMines(int safeX, int safeY);
    void calculateAdjacency();
    void linkNeighbors();

    int columns;
    int rows;
    int mineCount;

    bool lost = false;
    bool won = false;
    bool minesPlaced = false;

    // Seeded once per Board. Re-seeding a fresh mt19937 from time(nullptr) on
    // every call, as this class used to do, hands out the same board to every
    // reset() that happens within the same second.
    std::mt19937 rng;

    std::vector<std::vector<Tile>> grid;
    TextureManager* texManager;
};
