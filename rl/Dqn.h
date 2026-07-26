// Stage 3, components 3, 5 and 6: epsilon-greedy exploration, the target
// network, and the Q-update.
//
// The Q-network is an ordinary MLP from nn/, trained by the same autograd and
// the same Adam that Gates 1 and 2 already proved correct. Nothing in
// tensor/, autograd/, nn/ or optim/ was added or changed for this stage --
// that is the point of the build order. If the agent underperforms, the fault
// is here.
#pragma once

#include <random>
#include <vector>

#include "../nn/Layers.h"
#include "../optim/Optim.h"
#include "Env.h"
#include "Replay.h"

namespace ms {

struct DqnConfig {
    int rows = 6;
    int cols = 6;
    int mines = 5;

    int hidden = 128;

    double lr = 1e-3;
    // Discount. Minesweeper episodes are short (a handful of moves), so a
    // discount close to 1 is fine: there is no risk of an unbounded return.
    double gamma = 0.95;

    int batch = 64;
    int replay_capacity = 50000;
    // Transitions collected before any training happens, so the first batches
    // are not 64 copies of the same opening position.
    int warmup = 2000;
    // How often the target network is refreshed from the online network.
    int target_sync = 500;

    double eps_start = 1.0;
    double eps_end = 0.05;
    int eps_decay_steps = 15000;

    RewardConfig reward;
    uint64_t seed = 1;
};

// Linear decay from eps_start to eps_end over eps_decay_steps, flat after.
double epsilon_at(const DqnConfig& cfg, int step);

class DqnAgent {
public:
    explicit DqnAgent(const DqnConfig& cfg);

    // Epsilon-greedy over *legal* actions only. Returns a flat cell index, or
    // -1 if the board has no legal move.
    int select_action(const Board& b, double epsilon, std::mt19937_64& rng);

    // Pure exploitation, used for evaluation.
    int greedy_action(const Board& b) { return select_action(b, 0.0, eval_rng_); }

    // One gradient step on a uniformly sampled batch. Returns the loss, or -1
    // if the buffer does not hold a full batch yet.
    double train_step(std::mt19937_64& rng);

    void sync_target();

    ReplayBuffer& replay() { return replay_; }
    const nn::MLP& online() const { return online_; }
    const nn::MLP& target() const { return target_; }

    int actions() const { return cfg_.rows * cfg_.cols; }

private:
    // Q-values for one board, as a plain vector of length actions().
    std::vector<double> q_values(const nn::MLP& net, const Board& b);

    // Declaration order is initialisation order: init_rng_ must exist before
    // the two networks, which draw their initial weights from it.
    DqnConfig cfg_;
    std::mt19937_64 init_rng_;
    nn::MLP online_;
    nn::MLP target_;
    nn::Adam opt_;
    ReplayBuffer replay_;
    std::mt19937_64 eval_rng_;
};

// Copy every parameter value from `src` into `dst`. Both must have identical
// architecture. This is a plain value copy, not a graph operation -- the
// target network must not be differentiated through.
void copy_parameters(nn::MLP& dst, const nn::MLP& src);

struct EvalResult {
    double win_rate = 0.0;
    double mean_revealed = 0.0;
    double mean_moves = 0.0;
};

// Play `games` greedily (epsilon = 0) and report how it went.
EvalResult evaluate(DqnAgent& agent, const DqnConfig& cfg, int games, uint64_t seed);

}  // namespace ms
