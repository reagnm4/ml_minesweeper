#pragma once
#include <SFML/Graphics.hpp>
#include <vector>

class Number {
public:
    Number();

    void setTexture(const sf::Texture& digits);
    void setValue(int value);

    void draw(sf::RenderWindow& window);

private:
    const sf::Texture* digitsPtr = nullptr;
    std::vector<sf::Sprite> sprites;
    int value = 0;
};
