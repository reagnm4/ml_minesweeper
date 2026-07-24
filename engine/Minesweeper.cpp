#include "Minesweeper.h"

#include <algorithm>
#include <numeric>

namespace ms {

namespace {
constexpr int kDr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr int kDc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
}  // namespace

Board::Board(int rows, int cols, int mines, uint64_t seed, FirstClick policy)
    : rows_(std::max(rows, 0)),
      cols_(std::max(cols, 0)),
      mines_(std::max(mines, 0)),
      policy_(policy),
      rng_(seed) {
    // Clamping rather than asserting: the old engine looped forever on
    // mines > cells, and a config typo should never hang the program.
    mines_ = std::min(mines_, rows_ * cols_);
    mine_.resize(static_cast<size_t>(rows_) * cols_);
    adj_.resize(mine_.size());
    view_.resize(mine_.size());
    reset();
}

void Board::reset() {
    std::fill(mine_.begin(), mine_.end(), 0);
    std::fill(adj_.begin(), adj_.end(), 0);
    std::fill(view_.begin(), view_.end(), kHidden);
    placed_ = false;
    won_ = false;
    lost_ = false;
    revealed_ = 0;
    flags_ = 0;
    // A board with no safe cells at all is won the moment it starts; there is
    // nothing to reveal.
    if (mines_ == cells()) won_ = true;
}

void Board::reseed(uint64_t seed) {
    rng_.seed(seed);
    reset();
}

int Board::neighbors(int r, int c, int* out) const {
    int n = 0;
    for (int k = 0; k < 8; ++k) {
        const int nr = r + kDr[k], nc = c + kDc[k];
        if (in_bounds(nr, nc)) out[n++] = idx(nr, nc);
    }
    return n;
}

void Board::place_mines(int safe_r, int safe_c) {
    // Build the list of cells a mine is allowed to occupy, then partially
    // shuffle it (Fisher-Yates) and take the first `mines_`. This is uniform
    // over all valid layouts and, unlike rejection sampling, always
    // terminates -- including at 100% mine density.
    std::vector<uint8_t> forbidden(mine_.size(), 0);
    forbidden[idx(safe_r, safe_c)] = 1;
    if (policy_ == FirstClick::SafeNeighborhood) {
        int nb[8];
        const int n = neighbors(safe_r, safe_c, nb);
        for (int i = 0; i < n; ++i) forbidden[nb[i]] = 1;
    }

    std::vector<int> candidates;
    candidates.reserve(mine_.size());
    for (int i = 0; i < static_cast<int>(mine_.size()); ++i)
        if (!forbidden[i]) candidates.push_back(i);

    // If the board is too crowded to honour the full safe neighbourhood, fall
    // back to protecting only the clicked cell, and past that place mines
    // everywhere else. The clicked cell stays safe as long as mines < cells.
    if (static_cast<int>(candidates.size()) < mines_) {
        candidates.clear();
        for (int i = 0; i < static_cast<int>(mine_.size()); ++i)
            if (i != idx(safe_r, safe_c)) candidates.push_back(i);
    }
    if (static_cast<int>(candidates.size()) < mines_) {
        candidates.clear();
        candidates.resize(mine_.size());
        std::iota(candidates.begin(), candidates.end(), 0);
    }

    const int k = std::min<int>(mines_, static_cast<int>(candidates.size()));
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> pick(i, static_cast<int>(candidates.size()) - 1);
        std::swap(candidates[i], candidates[pick(rng_)]);
        mine_[candidates[i]] = 1;
    }

    placed_ = true;
    compute_adjacency();
}

void Board::compute_adjacency() {
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            int nb[8];
            const int n = neighbors(r, c, nb);
            int count = 0;
            for (int i = 0; i < n; ++i) count += mine_[nb[i]];
            adj_[idx(r, c)] = static_cast<int8_t>(count);
        }
    }
}

void Board::flood_reveal(int start) {
    // Iterative: the recursive version in the SFML engine blows the stack on
    // boards around 300x300 and larger.
    std::vector<int> stack;
    stack.push_back(start);

    while (!stack.empty()) {
        const int cur = stack.back();
        stack.pop_back();
        if (view_[cur] != kHidden) continue;  // already revealed, or flagged

        view_[cur] = adj_[cur];
        ++revealed_;
        if (adj_[cur] != 0) continue;

        const int r = cur / cols_, c = cur % cols_;
        int nb[8];
        const int n = neighbors(r, c, nb);
        for (int i = 0; i < n; ++i)
            if (view_[nb[i]] == kHidden) stack.push_back(nb[i]);
    }
}

Move Board::reveal(int r, int c) {
    if (is_over() || !in_bounds(r, c)) return Move::Invalid;

    const int i = idx(r, c);
    if (view_[i] != kHidden) return Move::NoOp;  // revealed or flagged

    if (!placed_) place_mines(r, c);

    if (mine_[i]) {
        view_[i] = 0;  // the code is irrelevant once the game is lost
        lost_ = true;
        return Move::HitMine;
    }

    flood_reveal(i);
    if (revealed_ == cells() - mines_) won_ = true;
    return Move::Revealed;
}

bool Board::toggle_flag(int r, int c) {
    if (is_over() || !in_bounds(r, c)) return false;
    const int i = idx(r, c);
    if (view_[i] == kHidden) {
        view_[i] = kFlagged;
        ++flags_;
        return true;
    }
    if (view_[i] == kFlagged) {
        view_[i] = kHidden;
        --flags_;
        return true;
    }
    return false;  // already revealed
}

}  // namespace ms
