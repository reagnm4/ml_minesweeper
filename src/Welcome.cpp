#include "Welcome.h"
#include "Game.h"
#include <iostream>
#include <cctype>

WelcomeWindow::WelcomeWindow(unsigned int w, unsigned int h) : width(w), height(h), title(font), prompt(font), nameText(font), window(sf::VideoMode({w, h}), "Welcome to Minesweeper", sf::Style::Close) {
    window.setFramerateLimit(60);

    if (!font.openFromFile("files/font.ttf")) {
        std::cerr << "Failed to load font\n";
    }

    setupText();
}

void WelcomeWindow::setupText() {
    title = sf::Text(font, "WELCOME TO MINESWEEPER !", 24);
    title.setStyle(sf::Text::Bold | sf::Text::Underlined);
    title.setFillColor(sf::Color::White);
    setTextCentered(title, width / 2.f, height / 2.f - 150);

    prompt = sf::Text(font, "Enter your name:", 20);
    prompt.setStyle(sf::Text::Bold);
    prompt.setFillColor(sf::Color::White);
    setTextCentered(prompt, width / 2.f, height / 2.f - 75);

    nameText = sf::Text(font, "|", 18);
    nameText.setStyle(sf::Text::Bold);
    nameText.setFillColor(sf::Color::Yellow);
    setTextCentered(nameText, width / 2.f, height / 2.f - 45);
}

void WelcomeWindow::setTextCentered(sf::Text& text, float x, float y) {
    sf::FloatRect r = text.getLocalBounds();
    text.setOrigin({r.position.x + r.size.x / 2.f, r.position.y + r.size.y / 2.f});
    text.setPosition({x, y});
}

std::string WelcomeWindow::getName() const {
    return name;
}

bool WelcomeWindow::run() {
    while (window.isOpen() && !enterPressed) {
        processEvents();

        window.clear(sf::Color::Blue);
        window.draw(title);
        window.draw(prompt);
        window.draw(nameText);
        window.display();
    }

    window.close();

    return enterPressed;
}

void WelcomeWindow::processEvents() {
    while (auto event = window.pollEvent()) {

        if (event->is<sf::Event::Closed>()) {
            window.close();
        }

        if (auto key = event->getIf<sf::Event::KeyPressed>()) {
            handleKeyPressed(key);
        }

        if (auto text = event->getIf<sf::Event::TextEntered>()) {
            handleTextEntered(text);
        }
    }
}

void WelcomeWindow::handleKeyPressed(const sf::Event::KeyPressed* key) {
    if (key->code == sf::Keyboard::Key::Enter && !name.empty()) {
        enterPressed = true;
    }
}

void WelcomeWindow::handleTextEntered(const sf::Event::TextEntered* text) {
    char c = static_cast<char>(text->unicode);

    if (c == 8 && !name.empty()) {
        name.pop_back();
    }
    else if (std::isalpha((unsigned char)c) && name.size() < 10) {
        if (name.empty()) name += std::toupper(c);
        else name += std::tolower(c);
    }

    nameText.setString(name + "|");
    setTextCentered(nameText, width / 2.f, height / 2.f - 45);
}