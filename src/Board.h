#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include "Tile.h"
#include "Textures.h"

class Board {
public:
    Board(int cols, int rows, int mines, TextureManager* tex);

    void reset();
    void randomizeMines();
    void calculateAdjacency();
    void linkNeighbors();
    void revealTile(int x, int y);
    void flagTile(int x, int y);

    bool isInBounds(int x, int y) const;
    bool checkWin() const;
    bool checkLoss() const;

    void draw(sf::RenderWindow& window, bool debugMode, bool paused);

    Tile& getTile(int x, int y);

private:
    int columns;
    int rows;
    int mineCount;

    bool lost = false;
    bool won = false;

    std::vector<std::vector<Tile>> grid;
    TextureManager* texManager;
};