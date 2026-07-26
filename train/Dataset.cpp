#include "Dataset.h"

#include <algorithm>
#include <numeric>
#include <random>

#include "../solver/Solver.h"

namespace ms {

double Dataset::majority_baseline() const {
    if (samples() == 0) return 0.0;
    const int negatives = samples() - positives;
    return static_cast<double>(std::max(positives, negatives)) / samples();
}

int features_for(int patch_radius) {
    const int side = 2 * patch_radius + 1;
    return side * side * kChannels;
}

void encode_patch(const Board& b, int r, int c, int patch_radius, double* out) {
    const int n = features_for(patch_radius);
    std::fill(out, out + n, 0.0);

    int cell = 0;
    for (int dr = -patch_radius; dr <= patch_radius; ++dr) {
        for (int dc = -patch_radius; dc <= patch_radius; ++dc, ++cell) {
            const int rr = r + dr, cc = c + dc;
            int channel;

            if (!b.in_bounds(rr, cc)) {
                channel = kOffBoard;
            } else {
                const int8_t v = b.at(rr, cc);
                // kHidden and kFlagged both collapse to kUnknown. This is the
                // line that stops the solver's (always correct) flags from
                // leaking the label into the input.
                channel = v >= 0 ? static_cast<int>(v) : kUnknown;
            }

            out[cell * kChannels + channel] = 1.0;
        }
    }
}

namespace {

// A cell worth asking about: still unknown, and touching something revealed.
// Cells in the middle of an untouched region carry no local information, so
// including them would just teach the network the base rate.
bool on_frontier(const Board& b, int r, int c, bool include_flagged) {
    const int8_t v = b.at(r, c);
    if (v == kFlagged && !include_flagged) return false;
    if (v != kHidden && v != kFlagged) return false;

    int nb[8];
    const int n = b.neighbors(r, c, nb);
    for (int i = 0; i < n; ++i)
        if (b.state()[nb[i]] >= 0) return true;
    return false;
}

}  // namespace

Dataset build_dataset(const DatasetConfig& cfg) {
    const int n_features = features_for(cfg.patch_radius);

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(static_cast<size_t>(cfg.max_samples) * n_features);
    ys.reserve(cfg.max_samples);

    std::mt19937_64 rng(cfg.seed);
    Board board(cfg.rows, cfg.cols, cfg.mines, cfg.seed);

    std::vector<double> scratch(n_features);
    std::vector<int> frontier;
    int positives = 0;
    int games = 0;

    for (; games < cfg.max_games && static_cast<int>(ys.size()) < cfg.max_samples; ++games) {
        board.reset();

        play(board, [&](const Board& b) {
            if (static_cast<int>(ys.size()) >= cfg.max_samples) return;

            frontier.clear();
            for (int r = 0; r < b.rows(); ++r)
                for (int c = 0; c < b.cols(); ++c)
                    if (on_frontier(b, r, c, cfg.include_flagged))
                        frontier.push_back(b.idx(r, c));
            if (frontier.empty()) return;

            // Take a few at random rather than all of them, so one long game
            // cannot dominate the dataset.
            const int take = std::min<int>(cfg.samples_per_turn,
                                           static_cast<int>(frontier.size()));
            for (int k = 0; k < take; ++k) {
                std::uniform_int_distribution<int> pick(k, static_cast<int>(frontier.size()) - 1);
                std::swap(frontier[k], frontier[pick(rng)]);

                const int idx = frontier[k];
                const int r = idx / b.cols(), c = idx % b.cols();

                encode_patch(b, r, c, cfg.patch_radius, scratch.data());
                xs.insert(xs.end(), scratch.begin(), scratch.end());

                const double label = b.is_mine(r, c) ? 1.0 : 0.0;
                ys.push_back(label);
                positives += static_cast<int>(label);

                if (static_cast<int>(ys.size()) >= cfg.max_samples) return;
            }
        });
    }

    // Shuffle so mini-batches are not correlated by game.
    const int n = static_cast<int>(ys.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    Dataset out;
    out.x = nn::Matrix(n, n_features);
    out.y = nn::Matrix(n, 1);
    out.games_used = games;
    out.positives = positives;

    for (int i = 0; i < n; ++i) {
        const int src = order[i];
        std::copy(xs.begin() + static_cast<size_t>(src) * n_features,
                  xs.begin() + static_cast<size_t>(src + 1) * n_features,
                  out.x.data().begin() + static_cast<size_t>(i) * n_features);
        out.y(i, 0) = ys[src];
    }
    return out;
}

nn::Matrix take_rows(const nn::Matrix& src, const std::vector<int>& order, int begin, int count) {
    nn::Matrix out(count, src.cols());
    for (int i = 0; i < count; ++i) {
        const int row = order[begin + i];
        std::copy(src.data().begin() + static_cast<size_t>(row) * src.cols(),
                  src.data().begin() + static_cast<size_t>(row + 1) * src.cols(),
                  out.data().begin() + static_cast<size_t>(i) * src.cols());
    }
    return out;
}

}  // namespace ms
