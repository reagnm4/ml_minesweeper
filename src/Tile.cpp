#include "Tile.h"

Tile::Tile()
    : sprite(nullptr)
{}

void Tile::setMine(bool m) { mine = m; }
void Tile::setRevealed(bool r) { revealed = r; }
void Tile::setFlagged(bool f) { flagged = f; }

bool Tile::isMine() const { return mine; }
bool Tile::isRevealed() const { return revealed; }
bool Tile::isFlagged() const { return flagged; }

void Tile::setAdjacentMines(int n) { adjacentMines = n; }
int Tile::getAdjacentMines() const { return adjacentMines; }

void Tile::addNeighbor(Tile* t) { neighbors.push_back(t); }
std::vector<Tile*>& Tile::getNeighbors() { return neighbors; }

void Tile::setTexture(sf::Texture* tex) {
    if (!tex) return;

    sf::Vector2f pos = sprite ? sprite->getPosition() : sf::Vector2f(0.f, 0.f);

    sprite = std::make_unique<sf::Sprite>(*tex);
    sprite->setPosition(pos);
}

void Tile::setNumberTexture(sf::Texture* digits, int number) {
    if (!digits) return;

    sf::Vector2f pos = sprite ? sprite->getPosition() : sf::Vector2f(0.f, 0.f);
    int digitWidth = 21;
    int index = number + 1;

    sf::IntRect rect(sf::Vector2i(index * digitWidth, 0), sf::Vector2i(digitWidth, 32));

    sprite = std::make_unique<sf::Sprite>(*digits, rect);
    sprite->setPosition(pos);
}


sf::Sprite& Tile::getSprite() { return *sprite; }

void Tile::setPosition(float x, float y) {
    if (sprite) sprite->setPosition({x, y});
}