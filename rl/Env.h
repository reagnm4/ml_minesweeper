// Stage 3, component 1 and 2: state encoding and the reward function.
//
// This is the whole interface between Minesweeper and the Q-network. It is
// deliberately the first thing built and the first thing sanity-checked,
// because every later component consumes it: a bug here would look exactly
// like "the DQN does not learn".
#pragma once

#include <cstdint>
#include <vector>

#include "../engine/Minesweeper.h"
#include "../tensor/Matrix.h"

namespace ms {

// One-hot channel layout per cell:
//   0..8      revealed, carrying that many adjacent mines
//   kRlUnknown  not revealed
//
// There is no flag channel and no off-board channel. Flags are absent because
// the agent has no flag action -- its only move is to reveal -- so a flag
// could never appear on a board it played. Off-board is absent because the
// encoding covers the whole board rather than a window, so nothing hangs off
// the edge. Both differ from the Stage-2 patch encoding for those reasons.
constexpr int kRlUnknown = 9;
constexpr int kRlChannels = 10;

inline int state_size(int rows, int cols) { return rows * cols * kRlChannels; }

// Write the full-board encoding into `out`, which must hold state_size()
// doubles. Uses only Board::state(), never ground truth.
void encode_state(const Board& b, double* out);

// Same, as a 1 x state_size row -- the shape the network's forward pass wants.
nn::Matrix encode_state_row(const Board& b);

// Flat indices of every cell that can legally be revealed. The action space is
// every cell, but only hidden cells are ever legal; masking illegal actions is
// what stops the agent from wasting its entire budget re-clicking open cells.
std::vector<int> legal_actions(const Board& b);

// Reward shaping. The spec warns that reward-shaping choices drive most of the
// reward jitter reported for Deep-Q on Minesweeper, so this is kept as small
// and as explicit as possible, and every value lives here rather than being
// scattered through the training loop.
struct RewardConfig {
    double win = 1.0;
    double mine = -1.0;
    // Per *move* that opens at least one cell, not per cell opened. Rewarding
    // per cell would make a lucky flood fill worth more than a correct
    // deduction, which is the opposite of what we want the agent to learn.
    double safe_reveal = 0.1;
};

struct Step {
    double reward = 0.0;
    bool done = false;
    bool won = false;
    bool hit_mine = false;
    int newly_revealed = 0;
};

// Apply `action` (a flat cell index) to `b` and report what happened.
Step step(Board& b, int action, const RewardConfig& cfg);

}  // namespace ms
