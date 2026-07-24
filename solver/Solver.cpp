#include "Solver.h"

#include <limits>
#include <vector>

namespace ms {

int solve_to_fixpoint(Board& b, SolveStats* stats) {
    int total = 0;
    bool changed = true;

    while (changed && !b.is_over()) {
        changed = false;

        for (int r = 0; r < b.rows(); ++r) {
            for (int c = 0; c < b.cols(); ++c) {
                const int8_t n = b.at(r, c);
                if (n <= 0) continue;  // hidden, flagged, or a revealed zero

                int nb[8];
                const int cnt = b.neighbors(r, c, nb);

                int flagged = 0;
                int hidden[8];
                int nhidden = 0;
                for (int i = 0; i < cnt; ++i) {
                    const int8_t v = b.state()[nb[i]];
                    if (v == kFlagged) ++flagged;
                    else if (v == kHidden) hidden[nhidden++] = nb[i];
                }
                if (nhidden == 0) continue;

                // Rule 1: the number equals the flags already placed plus every
                // remaining unknown -> all the unknowns must be mines.
                if (n == flagged + nhidden) {
                    for (int i = 0; i < nhidden; ++i) {
                        b.toggle_flag(hidden[i] / b.cols(), hidden[i] % b.cols());
                        if (stats) ++stats->flags;
                        ++total;
                    }
                    changed = true;
                    continue;
                }

                // Rule 2: the number is already fully accounted for by flags ->
                // every remaining unknown is safe.
                if (n == flagged) {
                    for (int i = 0; i < nhidden; ++i) {
                        // A flood fill from an earlier iteration of this loop
                        // may already have opened this cell; reveal() no-ops.
                        const Move m = b.reveal(hidden[i] / b.cols(), hidden[i] % b.cols());
                        if (m == Move::Revealed) {
                            if (stats) ++stats->reveals;
                            ++total;
                            changed = true;
                        } else if (m == Move::HitMine) {
                            // Unreachable unless the deduction logic is wrong.
                            if (stats) ++stats->reveals;
                            return total;
                        }
                    }
                }
            }
        }
    }
    return total;
}

int best_guess(const Board& b) {
    const int n = b.cells();

    // Baseline probability for a cell no revealed number touches: the mines
    // still unaccounted for, spread evenly over the cells still in play.
    int unknown_total = 0;
    for (int i = 0; i < n; ++i)
        if (b.state()[i] == kHidden) ++unknown_total;
    if (unknown_total == 0) return -1;

    const double density =
        unknown_total > 0
            ? static_cast<double>(b.mines_remaining()) / static_cast<double>(unknown_total)
            : 1.0;

    std::vector<double> p(n, density);

    // Each revealed number k with f flagged and u unknown neighbours says
    // "(k - f) mines are hidden among these u cells", i.e. each of them is a
    // mine with probability (k - f)/u *under that constraint alone*. A cell
    // covered by several constraints gets the worst (highest) of them, which
    // is a conservative estimate -- the exact joint probability needs
    // enumeration, and the spec explicitly allows a simple heuristic here.
    for (int r = 0; r < b.rows(); ++r) {
        for (int c = 0; c < b.cols(); ++c) {
            const int8_t k = b.at(r, c);
            if (k < 0) continue;

            int nb[8];
            const int cnt = b.neighbors(r, c, nb);
            int flagged = 0, unknown = 0;
            for (int i = 0; i < cnt; ++i) {
                const int8_t v = b.state()[nb[i]];
                if (v == kFlagged) ++flagged;
                else if (v == kHidden) ++unknown;
            }
            if (unknown == 0) continue;

            const double local = static_cast<double>(k - flagged) / unknown;
            for (int i = 0; i < cnt; ++i)
                if (b.state()[nb[i]] == kHidden && local > p[nb[i]]) p[nb[i]] = local;
        }
    }

    int best = -1;
    double best_p = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        if (b.state()[i] != kHidden) continue;
        if (p[i] < best_p) {
            best_p = p[i];
            best = i;
        }
    }
    return best;
}

SolveStats play(Board& b, const StateHook& on_turn) {
    SolveStats s;

    while (!b.is_over()) {
        if (on_turn) on_turn(b);

        if (solve_to_fixpoint(b, &s) > 0) continue;
        if (b.is_over()) break;

        const int g = best_guess(b);
        if (g < 0) break;  // nothing left to click; is_over() should already be true

        ++s.guesses;
        const Move m = b.reveal(g / b.cols(), g % b.cols());
        if (m == Move::HitMine) s.died_on_guess = true;
        else if (m == Move::Revealed) ++s.reveals;
        else break;  // Invalid/NoOp here would mean best_guess returned a bad cell
    }

    s.won = b.is_won();
    return s;
}

}  // namespace ms
