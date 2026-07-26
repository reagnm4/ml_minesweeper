#include "Dqn.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ms {

double epsilon_at(const DqnConfig& cfg, int step) {
    if (step >= cfg.eps_decay_steps) return cfg.eps_end;
    const double t = static_cast<double>(step) / cfg.eps_decay_steps;
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start);
}

void copy_parameters(nn::MLP& dst, const nn::MLP& src) {
    std::vector<nn::Tensor> d = dst.parameters();
    std::vector<nn::Tensor> s = src.parameters();
    for (size_t i = 0; i < d.size(); ++i)
        d[i].value().data() = s[i].value().data();
}

namespace {

// Expand a stored board view into the one-hot row the network expects.
void expand(const std::vector<int8_t>& view, double* out) {
    const int n = static_cast<int>(view.size()) * kRlChannels;
    std::fill(out, out + n, 0.0);
    for (size_t i = 0; i < view.size(); ++i) {
        const int channel = view[i] >= 0 ? static_cast<int>(view[i]) : kRlUnknown;
        out[i * kRlChannels + channel] = 1.0;
    }
}

}  // namespace

DqnAgent::DqnAgent(const DqnConfig& cfg)
    : cfg_(cfg),
      init_rng_(cfg.seed),
      // state -> hidden -> one Q-value per cell. The output layer has no
      // activation: a Q-value is an unbounded expected return, so squashing it
      // through a sigmoid or a ReLU would make some returns unrepresentable.
      online_({state_size(cfg.rows, cfg.cols), cfg.hidden, cfg.rows * cfg.cols},
              nn::Activation::ReLU, nn::Activation::None, init_rng_),
      target_({state_size(cfg.rows, cfg.cols), cfg.hidden, cfg.rows * cfg.cols},
              nn::Activation::ReLU, nn::Activation::None, init_rng_),
      opt_(online_.parameters(), cfg.lr),
      replay_(cfg.replay_capacity),
      eval_rng_(cfg.seed ^ 0x5EEDu) {
    // Both networks start life identical.
    copy_parameters(target_, online_);
}

std::vector<double> DqnAgent::q_values(const nn::MLP& net, const Board& b) {
    nn::Tensor q = net.forward(encode_state_row(b));
    std::vector<double> out(actions());
    for (int a = 0; a < actions(); ++a) out[a] = q.value()(0, a);
    return out;
}

int DqnAgent::select_action(const Board& b, double epsilon, std::mt19937_64& rng) {
    const std::vector<int> legal = legal_actions(b);
    if (legal.empty()) return -1;

    // Explore. Note this samples among *legal* actions, not all 36 cells:
    // exploring by clicking an already-open square teaches nothing and would
    // burn most of the exploration budget on no-ops.
    if (epsilon > 0.0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < epsilon)
        return legal[std::uniform_int_distribution<int>(0, static_cast<int>(legal.size()) - 1)(rng)];

    // Exploit: the highest-valued legal action.
    const std::vector<double> q = q_values(online_, b);
    int best = legal[0];
    double best_q = q[best];
    for (int a : legal)
        if (q[a] > best_q) { best_q = q[a]; best = a; }
    return best;
}

void DqnAgent::sync_target() { copy_parameters(target_, online_); }

double DqnAgent::train_step(std::mt19937_64& rng) {
    if (!replay_.ready(cfg_.batch)) return -1.0;

    const int B = cfg_.batch;
    const int A = actions();
    const int S = state_size(cfg_.rows, cfg_.cols);

    std::vector<int> idx;
    replay_.sample(B, rng, idx);

    nn::Matrix x(B, S), x_next(B, S);
    for (int i = 0; i < B; ++i) {
        const Transition& t = replay_.at(idx[i]);
        expand(t.state, &x.data()[static_cast<size_t>(i) * S]);
        expand(t.next_state, &x_next.data()[static_cast<size_t>(i) * S]);
    }

    // The target network's job. Using the online network here instead would
    // mean the value we regress toward moves every single update, because it
    // is produced by the very weights being changed -- the regression target
    // chases its own tail and training oscillates or diverges. Freezing a copy
    // for target_sync steps gives a stationary target to aim at.
    nn::Tensor q_next = target_.forward(x_next);

    nn::Tensor q = online_.forward(x);

    // Build the regression target. It starts as an exact copy of the network's
    // current output, and only the entry for the action actually taken is
    // overwritten:
    //
    //     y = r                                    if the episode ended
    //     y = r + gamma * max_{legal a'} Q_t(s',a')  otherwise
    //
    // Every other entry is left equal to the prediction, so (pred - target) is
    // exactly zero there and no gradient flows. That is how "update only the
    // action you took" is expressed using nothing but the ops Gate 1 already
    // verified -- no gather op was added to the framework for this stage.
    nn::Matrix targets = q.value();

    for (int i = 0; i < B; ++i) {
        const Transition& t = replay_.at(idx[i]);
        double y = t.reward;

        if (!t.done) {
            // Maximise over the legal actions of the *next* state. Including
            // illegal ones would let the bootstrapped value come from clicking
            // an already-open cell, which the agent can never actually do.
            double best = -std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < t.next_state.size(); ++c)
                if (t.next_state[c] == kHidden)
                    best = std::max(best, q_next.value()(i, static_cast<int>(c)));
            if (std::isfinite(best)) y += cfg_.gamma * best;
        }

        targets(i, t.action) = y;
    }

    // mse_loss averages over all B*A entries, but only B of them are non-zero,
    // so the gradient is 1/A of a per-action mean squared error. With Adam the
    // constant is irrelevant: the update is m_hat / (sqrt(v_hat) + eps), which
    // is invariant to any fixed rescaling of the gradient.
    opt_.zero_grad();
    nn::Tensor loss = nn::mse_loss(q, targets);
    loss.backward();
    opt_.step();

    return loss.item();
}

EvalResult evaluate(DqnAgent& agent, const DqnConfig& cfg, int games, uint64_t seed) {
    // A fixed seed means every evaluation plays the *same* set of boards, so
    // movement in the win rate reflects the policy changing rather than the
    // draw changing.
    Board b(cfg.rows, cfg.cols, cfg.mines, seed);
    EvalResult r;
    long long revealed = 0, moves = 0;

    for (int g = 0; g < games; ++g) {
        b.reset();
        while (!b.is_over()) {
            const int a = agent.greedy_action(b);
            if (a < 0) break;
            step(b, a, cfg.reward);
            ++moves;
        }
        r.win_rate += b.is_won() ? 1.0 : 0.0;
        revealed += b.revealed_count();
    }

    r.win_rate = 100.0 * r.win_rate / games;
    r.mean_revealed = static_cast<double>(revealed) / games;
    r.mean_moves = static_cast<double>(moves) / games;
    return r;
}

}  // namespace ms
