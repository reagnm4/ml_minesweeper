#include "Leaderboard.h"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstdio>

Leaderboard::Leaderboard(const std::string& filePath) : path(filePath) {
    font.openFromFile("files/font.ttf");
    loadScores();
}

void Leaderboard::addScore(const std::string& name, int timeSeconds) {
    scores.emplace_back(name, timeSeconds);
    sortScores();
    for (int i = 0; i < scores.size(); i++) {
        if (scores[i].first == name && scores[i].second == timeSeconds) {
            lastInsertedIndex = i;
            break;
        }
    }
    saveScores();
}

void Leaderboard::loadScores()
{
    scores.clear();

    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        size_t commaPos = line.find(',');
        if (commaPos == std::string::npos) continue;

        std::string timeStr = line.substr(0, commaPos);
        std::string name     = line.substr(commaPos + 1);

        if (!name.empty() && name[0] == ' ') {
            name = name.substr(1);
        }

        int m = 0, s = 0;
        sscanf(timeStr.c_str(), "%d:%d", &m, &s);

        int totalSec = m * 60 + s;

        scores.emplace_back(name, totalSec);
    }

    sortScores();
}

void Leaderboard::saveScores() {
    std::ofstream file(path, std::ios::trunc);

    for (auto& entry : scores) {
        int m = entry.second / 60;
        int s = entry.second % 60;
        char buf[10];
        sprintf(buf, "%02d:%02d", m, s);

        file << buf << ", " << entry.first << "\n";
    }
}

void Leaderboard::sortScores() {
    std::sort(scores.begin(), scores.end(), [](auto& a, auto& b) {return a.second < b.second;});
}

void Leaderboard::drawTitle(sf::RenderWindow& window) {
    sf::Text title(font, "LEADERBOARD", 20);
    title.setFillColor(sf::Color::White);
    title.setStyle(sf::Text::Bold | sf::Text::Underlined);

    float x = window.getSize().x / 2.f;
    float y = 60.f;

    sf::FloatRect r = title.getLocalBounds();
    title.setOrigin({r.position.x + r.size.x / 2.f, r.position.y + r.size.y / 2.f});
    title.setPosition({x, y});
    window.draw(title);
}

void Leaderboard::drawScores(sf::RenderWindow& window) {
    float startX = window.getSize().x / 2.f;
    float startY = 140.f;

    float colIndex = startX - 130.f;
    float colTime  = startX;
    float colName  = startX + 130.f;

    for (int i = 0; i < scores.size() && i < 5; i++) {
        int minutes = scores[i].second / 60;
        int seconds = scores[i].second % 60;

        char timeBuf[10];
        sprintf(timeBuf, "%02d:%02d", minutes, seconds);

        sf::Text idx(font, std::to_string(i + 1) + ".", 18);
        idx.setFillColor(sf::Color::White);
        idx.setStyle(sf::Text::Bold);

        sf::FloatRect r1 = idx.getLocalBounds();
        idx.setOrigin({r1.size.x / 2.f, r1.size.y / 2.f});
        idx.setPosition({colIndex, startY + i * 40.f});
        window.draw(idx);

        sf::Text t(font, timeBuf, 18);
        t.setFillColor(sf::Color::White);
        t.setStyle(sf::Text::Bold);

        sf::FloatRect r2 = t.getLocalBounds();
        t.setOrigin({r2.size.x / 2.f, r2.size.y / 2.f});
        t.setPosition({colTime, startY + i * 40.f});
        window.draw(t);

        std::string displayName = scores[i].first;

        if (i == lastInsertedIndex)
            displayName += "*";

        sf::Text n(font, displayName, 18);
        n.setFillColor(sf::Color::White);
        n.setStyle(sf::Text::Bold);

        sf::FloatRect r3 = n.getLocalBounds();
        n.setOrigin({r3.size.x / 2.f, r3.size.y / 2.f});
        n.setPosition({colName, startY + i * 40.f});
        window.draw(n);
    }
}

void Leaderboard::openWindow() {
    sf::RenderWindow window(
        sf::VideoMode({400, 400}),
        "Leaderboard",
        sf::Style::Close
    );

    window.setFramerateLimit(60);

    while (window.isOpen())
    {
        while (auto event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear(sf::Color::Blue);
        drawTitle(window);
        drawScores(window);
        window.display();
    }
}
