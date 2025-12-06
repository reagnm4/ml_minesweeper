#include "Config.h"
#include "Welcome.h"
#include "Game.h"

int main() {
    Config cfg;
    cfg.load("files/config.cfg");

    int width  = cfg.columns * 32;
    int height = cfg.rows * 32 + 100;

    WelcomeWindow welcome(width, height);
    if (!welcome.run()) return 0;

    std::string playerName = welcome.getName();

    GameWindow game(width, height, playerName, cfg);
    game.run();

    return 0;
}
