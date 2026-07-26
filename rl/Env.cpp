#include "Env.h"

#include <algorithm>

namespace ms {

void encode_state(const Board& b, double* out) {
    const int n = state_size(b.rows(), b.cols());
    std::fill(out, out + n, 0.0);

    for (int i = 0; i < b.cells(); ++i) {
        const int8_t v = b.state()[i];
        // kHidden and kFlagged both land on kRlUnknown. The agent cannot flag,
        // so kFlagged should never appear here, but collapsing it costs
        // nothing and keeps the encoding total.
        const int channel = v >= 0 ? static_cast<int>(v) : kRlUnknown;
        out[i * kRlChannels + channel] = 1.0;
    }
}

nn::Matrix encode_state_row(const Board& b) {
    nn::Matrix m(1, state_size(b.rows(), b.cols()));
    encode_state(b, m.data().data());
    return m;
}

std::vector<int> legal_actions(const Board& b) {
    std::vector<int> out;
    if (b.is_over()) return out;
    out.reserve(b.cells());
    for (int i = 0; i < b.cells(); ++i)
        if (b.state()[i] == kHidden) out.push_back(i);
    return out;
}

Step step(Board& b, int action, const RewardConfig& cfg) {
    Step s;
    const int before = b.revealed_count();

    const Move m = b.reveal(action / b.cols(), action % b.cols());
    s.newly_revealed = b.revealed_count() - before;

    switch (m) {
        case Move::HitMine:
            s.reward = cfg.mine;
            s.done = true;
            s.hit_mine = true;
            break;

        case Move::Revealed:
            if (b.is_won()) {
                s.reward = cfg.win;
                s.done = true;
                s.won = true;
            } else {
                s.reward = cfg.safe_reveal;
            }
            break;

        case Move::NoOp:
        case Move::Invalid:
            // The caller picked an illegal action. Nothing changes and no
            // reward is given; the training loop only ever selects from
            // legal_actions(), so reaching here means a bug upstream.
            s.done = b.is_over();
            break;
    }

    return s;
}

}  // namespace ms
