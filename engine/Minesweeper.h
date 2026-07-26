// Headless Minesweeper engine.
//
// This is the single source of truth for game rules on the ML side of the
// project. It deliberately has no dependency on SFML, no I/O and no global
// state, so that it can be copied, snapshotted and run a million times in a
// tight loop.
//
// Coordinates are (row, col) throughout, matching the spec. The SFML game in
// src/ uses (x, y) = (col, row); do not mix them up.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace ms {

// Cell codes used by state()/at(). Values 0..8 mean "revealed, and this many
// of my eight neighbours are mines". The two negatives are the states an
// observer cannot see through.
constexpr int8_t kHidden = -1;
constexpr int8_t kFlagged = -2;

// What the engine guarantees about the very first reveal of a game.
enum class FirstClick {
    // Only the clicked cell is guaranteed to be mine-free.
    SafeCell,
    // The clicked cell and its eight neighbours are mine-free, so the first
    // reveal always opens a zero and triggers a flood fill. This is what
    // modern Minesweeper implementations do, and it is the default because it
    // is what the Stage-0 baseline should be measured against.
    SafeNeighborhood,
};

enum class Move {
    Invalid,   // out of bounds, or the game is already over
    NoOp,      // legal coordinates, but the cell was already revealed/flagged
    Revealed,  // one or more cells were revealed
    HitMine,   // revealed a mine; the game is now lost
};

class Board {
public:
    // `seed` fixes the whole sequence of boards this object will ever produce,
    // so an entire experiment is reproducible from one number.
    //
    // `mines` is clamped to what the board can actually hold; a request for
    // more mines than cells is a caller bug, but it must not hang.
    Board(int rows, int cols, int mines, uint64_t seed,
          FirstClick policy = FirstClick::SafeNeighborhood);

    // Start a fresh game. The RNG stream continues where it left off, so
    // repeated reset() calls give *different* boards -- deterministically.
    void reset();

    // Restart the RNG stream from a known point, then reset().
    void reseed(uint64_t seed);

    // Reveal a cell. Mines are placed lazily on the first reveal of a game,
    // which is how first-click safety is implemented: the engine simply
    // refuses to put a mine where the player is about to click.
    Move reveal(int r, int c);

    // Toggle a flag. Returns true if the flag state changed. Flags are a
    // bookkeeping aid for the solver; they never affect win/loss detection,
    // but they do block reveal() the way a real game does.
    bool toggle_flag(int r, int c);

    // --- observation (what an agent is allowed to see) --------------------
    // Row-major, size rows()*cols(). Kept up to date incrementally.
    const std::vector<int8_t>& state() const { return view_; }
    int8_t at(int r, int c) const { return view_[idx(r, c)]; }
    bool is_hidden(int r, int c) const { return at(r, c) == kHidden; }
    bool is_flagged(int r, int c) const { return at(r, c) == kFlagged; }
    bool is_revealed(int r, int c) const { return at(r, c) >= 0; }

    bool is_won() const { return won_; }
    bool is_lost() const { return lost_; }
    bool is_over() const { return won_ || lost_; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int cells() const { return rows_ * cols_; }
    int mines() const { return mines_; }
    int flags_placed() const { return flags_; }
    int revealed_count() const { return revealed_; }
    // Mines the player has not yet accounted for with a flag. May go negative
    // if the player over-flags; that is the player's problem, not a bug.
    int mines_remaining() const { return mines_ - flags_; }

    bool in_bounds(int r, int c) const {
        return r >= 0 && r < rows_ && c >= 0 && c < cols_;
    }
    int idx(int r, int c) const { return r * cols_ + c; }

    // Neighbour offsets, exposed so callers do not each rewrite the dr/dc loop.
    // Writes up to 8 flat indices into `out` and returns how many were written.
    int neighbors(int r, int c, int* out) const;

    // --- ground truth (NOT part of the observation) -----------------------
    // Only for dataset labelling, tests and debug rendering. An agent that
    // calls these is cheating.
    bool is_mine(int r, int c) const { return mine_[idx(r, c)]; }
    int adjacent_mines(int r, int c) const { return adj_[idx(r, c)]; }
    bool mines_placed() const { return placed_; }

private:
    void place_mines(int safe_r, int safe_c);
    void compute_adjacency();
    void flood_reveal(int start);  // iterative; never recurses

    int rows_, cols_, mines_;
    FirstClick policy_;

    std::mt19937_64 rng_;

    std::vector<uint8_t> mine_;  // ground truth
    std::vector<int8_t> adj_;    // ground truth adjacency counts
    std::vector<int8_t> view_;   // what the player sees

    bool placed_ = false;
    bool won_ = false;
    bool lost_ = false;
    int revealed_ = 0;
    int flags_ = 0;
};

}  // namespace ms
