#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <string>

class Leaderboard {
public:
    Leaderboard(const std::string& filePath);
    void addScore(const std::string& name, int timeSeconds);
    void openWindow();

private:
    void loadScores();
    void saveScores();
    void sortScores();
    void drawTitle(sf::RenderWindow& window);
    void drawScores(sf::RenderWindow& window);

    std::string path;
    std::vector<std::pair<std::string, int>> scores;
    sf::Font font;
    int lastInsertedIndex = -1;
};


