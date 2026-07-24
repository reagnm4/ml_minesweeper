// Deterministic single-cell constraint solver (Stage 0).
//
// This is the honest baseline every learned agent in this project is measured
// against, and the thing that generates board states for the Stage-2 dataset.
// It only ever reasons about one revealed cell at a time -- no subset logic,
// no global mine-count reasoning, no enumeration. That is deliberate: it keeps
// the baseline simple enough to be obviously correct.
#pragma once

#include <functional>

#include "../engine/Minesweeper.h"

namespace ms {

struct SolveStats {
    bool won = false;
    int reveals = 0;        // reveal() calls that actually opened something
    int flags = 0;          // cells flagged by deduction
    int guesses = 0;        // moves made with no deterministic option
    bool died_on_guess = false;  // false + !won means a deduction was wrong (a bug)
};

// Apply single-cell logic until no further deduction is available.
// Returns the number of deductions applied (reveals + flags).
int solve_to_fixpoint(Board& b, SolveStats* stats = nullptr);

// Flat index of the hidden cell least likely to be a mine, or -1 if there is
// no hidden cell left. Only meaningful once solve_to_fixpoint() is stuck.
int best_guess(const Board& b);

// Called with the board at the start of every turn, before the solver acts on
// it. Used by the Stage-2 dataset builder to capture realistic mid-game
// states; ignored otherwise.
using StateHook = std::function<void(const Board&)>;

// Play one board to completion.
SolveStats play(Board& b, const StateHook& on_turn = nullptr);

}  // namespace ms
