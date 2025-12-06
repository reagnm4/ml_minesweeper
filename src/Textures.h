#pragma once
#include <SFML/Graphics.hpp>
#include <unordered_map>
#include <string>
#include <memory>

class TextureManager {
public:
    TextureManager() = default;
    bool load(const std::string& key, const std::string& filepath);
    sf::Texture* get(const std::string& key);

private:
    std::unordered_map<std::string, std::unique_ptr<sf::Texture>> textures;
};
