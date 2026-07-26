cmake_minimum_required(VERSION 3.28)
project(ml_minesweeper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The SFML game and the ML framework are built independently. Turn this off to
# build the headless framework with no network access and no graphics stack --
# that is how the tests and gates are run.
option(MS_BUILD_GUI "Build the SFML Minesweeper game in src/" ON)

# ---------------------------------------------------------------------------
# Headless framework (no third-party dependencies, by design -- see spec S2)
# ---------------------------------------------------------------------------

add_library(ms_engine engine/Minesweeper.cpp)
target_include_directories(ms_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(ms_solver solver/Solver.cpp)
target_link_libraries(ms_solver PUBLIC ms_engine)

add_library(ms_tensor tensor/Matrix.cpp)
target_include_directories(ms_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(ms_autograd autograd/Autograd.cpp autograd/GradCheck.cpp)
target_link_libraries(ms_autograd PUBLIC ms_tensor)

add_library(ms_nn nn/Layers.cpp optim/Optim.cpp)
target_link_libraries(ms_nn PUBLIC ms_autograd)

add_library(ms_train train/Dataset.cpp)
target_link_libraries(ms_train PUBLIC ms_nn ms_solver)

add_library(ms_rl rl/Env.cpp rl/Dqn.cpp)
target_link_libraries(ms_rl PUBLIC ms_nn ms_engine)

enable_testing()

# Each gate is its own executable so it can be run and read in isolation.
function(ms_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${ARGN})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

ms_add_test(test_engine      tests/test_engine.cpp      ms_engine)
ms_add_test(gate0_baseline   tests/gate0_baseline.cpp   ms_solver)
ms_add_test(gate1_gradcheck  tests/gate1_gradcheck.cpp  ms_autograd)
ms_add_test(gate2a_xor       tests/gate2a_xor.cpp       ms_nn)
ms_add_test(gate2b_mines     tests/gate2b_mines.cpp     ms_train)
ms_add_test(test_rl_components tests/test_rl_components.cpp ms_rl)
ms_add_test(gate3_dqn        tests/gate3_dqn.cpp        ms_rl ms_solver)

# Dumps gradients for tests/oracle/compare_oracle.sh to diff against NumPy.
# Not a test itself -- it prints numbers, the shell script judges them.
add_executable(oracle_dump tests/oracle_dump.cpp)
target_link_libraries(oracle_dump PRIVATE ms_autograd)
add_test(NAME oracle_numpy_crosscheck
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/compare_oracle.sh
                 $<TARGET_FILE:oracle_dump>)

# ---------------------------------------------------------------------------
# SFML game (unchanged in structure; see README for the bugs fixed in Board)
# ---------------------------------------------------------------------------

if(MS_BUILD_GUI)
    include(FetchContent)
    FetchContent_Declare(SFML
            GIT_REPOSITORY https://github.com/SFML/SFML.git
            GIT_TAG 3.0.2
            GIT_SHALLOW ON
            EXCLUDE_FROM_ALL
            SYSTEM
    )
    FetchContent_MakeAvailable(SFML)

    add_executable(moore.reagan.project3
            src/main.cpp
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
            src/Config.h
            src/Config.cpp)
    target_compile_features(moore.reagan.project3 PRIVATE cxx_std_17)
    target_link_libraries(moore.reagan.project3 PRIVATE SFML::Graphics)
endif()
