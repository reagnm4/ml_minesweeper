cmake_minimum_required(VERSION 3.28)
project(moore.reagan.project3 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(SFML
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG 3.0.2
        GIT_SHALLOW ON
        EXCLUDE_FROM_ALL
        SYSTEM
)
FetchContent_MakeAvailable(SFML)

add_executable(moore.reagan.project3 src/main.cpp
        src/Welcome.cpp
        src/Welcome.h
        src/Game.h
        src/Game.cpp
        src/Tile.h
        src/Tile.cpp
        src/Board.h
        src/Board.cpp
        src/Button.h
        src/Button.cpp
        src/Timer.h
        src/Timer.cpp
        src/Number.h
        src/Number.cpp
        src/Leaderboard.h
        src/Leaderboard.cpp
        src/Textures.h
        src/Textures.cpp
        src/config.h
        src/config.cpp)
target_compile_features(moore.reagan.project3 PRIVATE cxx_std_17)
target_link_libraries(moore.reagan.project3 PRIVATE SFML::Graphics)