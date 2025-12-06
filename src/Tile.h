#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <memory>

class Tile {
public:
    Tile();
    void setMine(bool m);
    void setRevealed(bool r);
    void setFlagged(bool f);
    bool isMine() const;
    bool isRevealed() const;
    bool isFlagged() const;
    void setAdjacentMines(int n);
    int getAdjacentMines() const;
    void addNeighbor(Tile* t);
    std::vector<Tile*>& getNeighbors();
    void setCoords(int x, int y) { tx = x; ty = y; }
    void setTexture(sf::Texture* tex);
    void setNumberTexture(sf::Texture* digits, int number);
    sf::Sprite& getSprite();
    void setPosition(float x, float y);
    int getX() const { return tx; }
    int getY() const { return ty; }

private:
    bool mine = false;
    bool revealed = false;
    bool flagged = false;
    int adjacentMines = 0;

    std::unique_ptr<sf::Sprite> sprite;
    sf::Texture* texturePtr = nullptr;

    std::vector<Tile*> neighbors;

    int tx;
    int ty;
};
