#pragma once
#include <string>

struct Config {
    int columns = 0;
    int rows = 0;
    int mines = 0;

    bool load(const std::string& filename);
};
