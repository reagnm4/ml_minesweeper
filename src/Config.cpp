#include "Config.h"
#include <fstream>
#include <iostream>

bool Config::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot read " << filename << "\n";
        return false;
    }

    if (!(file >> columns >> rows >> mines)) {
        std::cerr << "ERROR: Invalid config file format.\n";
        return false;
    }

    return true;
}
