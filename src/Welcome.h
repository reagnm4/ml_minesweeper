#pragma once
#include <SFML/Graphics.hpp>
#include <string>

class WelcomeWindow {
public:
    WelcomeWindow(unsigned int w, unsigned int h);
    bool run();
    std::string getName() const;

private:
    void processEvents();
    void handleKeyPressed(const sf::Event::KeyPressed* key);
    void handleTextEntered(const sf::Event::TextEntered* text);
    void setTextCentered(sf::Text& text, float x, float y);
    void setupText();

    unsigned int width;
    unsigned int height;

    sf::RenderWindow window;
    sf::Font font;

    sf::Text title;
    sf::Text prompt;
    sf::Text nameText;

    std::string name;
    bool enterPressed = false;
};