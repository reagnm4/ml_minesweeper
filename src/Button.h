#pragma once
#include <SFML/Graphics.hpp>
#include <memory>

class Button {
public:
    Button();
    Button(sf::Texture* tex);
    void setTexture(sf::Texture* tex);
    void setPosition(float x, float y);
    bool wasClicked(sf::Vector2i mousePos) const;

    sf::Sprite& getSprite();

private:
    std::unique_ptr<sf::Sprite> sprite;
    sf::Texture* texturePtr = nullptr;
};
