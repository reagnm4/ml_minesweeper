#include "Number.h"

Number::Number() {}

void Number::setTexture(const sf::Texture& digits) {
    digitsPtr = &digits;
}

void Number::setValue(int v) {
    value = v;
}

void Number::draw(sf::RenderWindow& window) {
    for (auto& s : sprites)
        window.draw(s);
}
