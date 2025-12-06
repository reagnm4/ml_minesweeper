#include "Button.h"

Button::Button() : sprite(nullptr) {}

Button::Button(sf::Texture* tex) : texturePtr(tex) {
    sprite = std::make_unique<sf::Sprite>(*texturePtr);
}

void Button::setTexture(sf::Texture* tex) {
    texturePtr = tex;
    sprite = std::make_unique<sf::Sprite>(*texturePtr);
}

void Button::setPosition(float x, float y) {
    if (sprite)
        sprite->setPosition({x, y});
}

bool Button::wasClicked(sf::Vector2i mousePos) const {
    sf::Vector2f p{(float)mousePos.x, (float)mousePos.y};
    return sprite->getGlobalBounds().contains(p);
}

sf::Sprite& Button::getSprite() {
    return *sprite;
}
