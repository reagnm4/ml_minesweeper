#include "Textures.h"
#include <iostream>

bool TextureManager::load(const std::string& key, const std::string& filepath) {
    auto tex = std::make_unique<sf::Texture>();

    if (!tex->loadFromFile(filepath)) {
        std::cerr << "Failed to load texture: " << filepath << std::endl;
        return false;
    }

    textures[key] = std::move(tex);
    return true;
}

sf::Texture* TextureManager::get(const std::string& key) {
    auto it = textures.find(key);
    if (it != textures.end())
        return it->second.get();
    return nullptr;
}
