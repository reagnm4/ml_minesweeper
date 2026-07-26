// Stage 0 engine tests.
//
// Several of these exist specifically because the original SFML engine failed
// them: reset() returning identical boards, no first-click safety, an infinite
// loop on mines > cells, and a recursive flood fill that overflowed the stack.
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "../engine/Minesweeper.h"
#include "TestUtil.h"

using namespace ms;
using test::check;
using test::note;
using test::section;

namespace {

std::string layout(const Board& b) {
    std::string s;
    for (int r = 0; r < b.rows(); ++r)
        for (int c = 0; c < b.cols(); ++c) s += b.is_mine(r, c) ? '*' : '.';
    return s;
}

int count_mines(const Board& b) {
    int n = 0;
    for (int r = 0; r < b.rows(); ++r)
        for (int c = 0; c < b.cols(); ++c) n += b.is_mine(r, c);
    return n;
}

}  // namespace

int main() {
    constexpr int R = 16, C = 26, M = 50;

    section("randomness and determinism");
    {
        Board b(R, C, M, /*seed=*/1234);
        std::set<std::string> seen;
        for (int i = 0; i < 200; ++i) {
            b.reset();
            b.reveal(0, 0);  // force mine placement
            seen.insert(layout(b));
        }
        check(seen.size() == 200, "200 resets produce 200 distinct layouts",
              "got " + std::to_string(seen.size()) + " distinct");
    }
    {
        // Same seed => same sequence of boards. This is what makes a 1000-game
        // baseline reproducible.
        Board a(R, C, M, 99), b(R, C, M, 99);
        bool same = true;
        for (int i = 0; i < 50 && same; ++i) {
            a.reveal(3, 3);
            b.reveal(3, 3);
            if (layout(a) != layout(b)) same = false;
            a.reset();
            b.reset();
        }
        check(same, "identical seeds replay an identical sequence of boards");
    }
    {
        Board a(R, C, M, 1), b(R, C, M, 2);
        a.reveal(0, 0);
        b.reveal(0, 0);
        check(layout(a) != layout(b), "different seeds give different boards");
    }

    section("mine placement");
    {
        bool ok = true;
        Board b(R, C, M, 7);
        for (int i = 0; i < 300 && ok; ++i) {
            b.reset();
            b.reveal(i % R, i % C);
            if (count_mines(b) != M) ok = false;
        }
        check(ok, "every board holds exactly `mines` mines");
    }
    {
        // The old engine died on the first click at roughly the mine density.
        int deaths = 0;
        constexpr int kTrials = 2000;
        Board b(R, C, M, 4242);
        for (int i = 0; i < kTrials; ++i) {
            b.reset();
            const int r = static_cast<int>(i * 7919u % R);
            const int c = static_cast<int>(i * 104729u % C);
            if (b.reveal(r, c) == Move::HitMine) ++deaths;
        }
        check(deaths == 0, "first click is never a mine",
              std::to_string(deaths) + "/" + std::to_string(kTrials) + " deaths");
    }
    {
        // SafeNeighborhood should also guarantee the first click opens a zero.
        bool always_zero = true, nbrs_clear = true;
        Board b(R, C, M, 31337);
        for (int i = 0; i < 500; ++i) {
            b.reset();
            const int r = static_cast<int>(i * 7919u % R);
            const int c = static_cast<int>(i * 104729u % C);
            b.reveal(r, c);
            if (b.adjacent_mines(r, c) != 0) always_zero = false;
            int nb[8];
            const int n = b.neighbors(r, c, nb);
            for (int k = 0; k < n; ++k)
                if (b.is_mine(nb[k] / C, nb[k] % C)) nbrs_clear = false;
        }
        check(nbrs_clear, "SafeNeighborhood keeps all 8 neighbours mine-free");
        check(always_zero, "first click always opens a zero (so a region opens)");
    }
    {
        Board b(R, C, M, 5, FirstClick::SafeCell);
        int deaths = 0, nonzero = 0;
        for (int i = 0; i < 500; ++i) {
            b.reset();
            if (b.reveal(8, 13) == Move::HitMine) ++deaths;
            if (b.adjacent_mines(8, 13) != 0) ++nonzero;
        }
        check(deaths == 0, "SafeCell policy: clicked cell is still never a mine");
        check(nonzero > 0, "SafeCell policy: neighbours may be mines (as documented)",
              std::to_string(nonzero) + "/500 first clicks landed on a number");
    }

    section("adjacency and reveal rules");
    {
        Board b(R, C, M, 11);
        b.reveal(5, 5);
        bool ok = true;
        for (int r = 0; r < R && ok; ++r)
            for (int c = 0; c < C && ok; ++c) {
                int n = 0;
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (!dr && !dc) continue;
                        if (!b.in_bounds(r + dr, c + dc)) continue;
                        n += b.is_mine(r + dr, c + dc);
                    }
                if (b.adjacent_mines(r, c) != n) ok = false;
            }
        check(ok, "adjacency counts match an independent recount");
    }
    {
        // Reveal only cells known to be safe; a mine must never surface, and
        // every revealed zero must have all its neighbours open.
        bool no_mine_shown = true, closed = true;
        Board b(R, C, M, 808);
        for (int t = 0; t < 200; ++t) {
            b.reset();
            b.reveal(t % R, (t * 3) % C);
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c)
                    if (!b.is_mine(r, c)) b.reveal(r, c);

            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) {
                    if (b.is_revealed(r, c) && b.is_mine(r, c)) no_mine_shown = false;
                    if (!b.is_revealed(r, c) || b.at(r, c) != 0) continue;
                    int nb[8];
                    const int n = b.neighbors(r, c, nb);
                    for (int k = 0; k < n; ++k)
                        if (b.state()[nb[k]] == kHidden) closed = false;
                }
            if (!b.is_won()) closed = false;
        }
        check(no_mine_shown, "revealing safe cells never exposes a mine");
        check(closed, "zero-cell flood fill is closed, and clearing all safe cells wins");
    }
    {
        Board b(R, C, M, 12);
        b.reveal(0, 0);
        int mine_r = -1, mine_c = -1;
        for (int r = 0; r < R && mine_r < 0; ++r)
            for (int c = 0; c < C && mine_r < 0; ++c)
                if (b.is_mine(r, c)) { mine_r = r; mine_c = c; }
        check(b.reveal(mine_r, mine_c) == Move::HitMine && b.is_lost() && !b.is_won(),
              "revealing a mine loses the game");
        check(b.reveal(0, 1) == Move::Invalid, "moves after game over are rejected");
    }

    section("flags");
    {
        Board b(R, C, M, 13);
        b.reveal(0, 0);
        int hr = -1, hc = -1;
        for (int r = 0; r < R && hr < 0; ++r)
            for (int c = 0; c < C && hr < 0; ++c)
                if (b.is_hidden(r, c)) { hr = r; hc = c; }

        check(b.toggle_flag(hr, hc) && b.is_flagged(hr, hc) && b.flags_placed() == 1,
              "flagging a hidden cell works");
        check(b.reveal(hr, hc) == Move::NoOp && b.is_flagged(hr, hc),
              "a flagged cell cannot be revealed");
        check(b.toggle_flag(hr, hc) && b.is_hidden(hr, hc) && b.flags_placed() == 0,
              "unflagging restores the hidden state");
        check(!b.toggle_flag(0, 0), "an already-revealed cell cannot be flagged");
        check(b.mines_remaining() == M, "mines_remaining tracks flags");
    }
    {
        // Flags must not be able to fake a win.
        Board b(R, C, M, 14);
        b.reveal(0, 0);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                if (b.is_hidden(r, c)) b.toggle_flag(r, c);
        check(!b.is_won(), "flagging everything does not win the game");
    }

    section("observation hygiene");
    {
        Board b(R, C, M, 15);
        b.reveal(4, 4);
        int hidden_or_flagged = 0, leaks = 0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                const int8_t v = b.at(r, c);
                if (v < 0) { ++hidden_or_flagged; continue; }
                if (b.is_mine(r, c)) ++leaks;                 // a mine shown as a number
                if (v != b.adjacent_mines(r, c)) ++leaks;     // wrong number shown
            }
        check(leaks == 0 && hidden_or_flagged > 0,
              "state() never reveals ground truth about hidden cells");
    }

    section("copyability and snapshots");
    {
        Board a(R, C, M, 16);
        a.reveal(2, 2);
        Board snapshot = a;                       // must compile and deep-copy
        const std::string before = layout(snapshot);
        const int revealed_before = snapshot.revealed_count();

        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                if (!a.is_mine(r, c)) a.reveal(r, c);

        check(layout(snapshot) == before && snapshot.revealed_count() == revealed_before,
              "a copied Board is independent of the original");
        check(snapshot.state() != a.state(), "the snapshot kept the older view");
    }

    section("degenerate configurations (the old engine hung on these)");
    {
        Board b(5, 5, 30, 17);  // more mines than cells
        check(b.mines() == 25, "mines > cells is clamped instead of looping forever",
              "requested 30, got " + std::to_string(b.mines()));
        check(b.is_won(), "a board that is entirely mines is already won");
    }
    {
        Board b(5, 5, 24, 18);  // one safe cell, SafeNeighborhood impossible
        check(b.reveal(2, 2) != Move::HitMine,
              "an over-crowded board still honours the clicked cell");
        check(b.is_won() && count_mines(b) == 24, "the single safe cell wins immediately");
    }
    {
        Board b(0, 0, 0, 19);
        check(b.cells() == 0 && b.reveal(0, 0) == Move::Invalid, "a zero-size board is inert");
    }
    {
        Board b(8, 8, 0, 20);
        check(b.reveal(4, 4) == Move::Revealed && b.is_won(),
              "a board with no mines is won by one click");
    }
    {
        Board b(4, 4, 5, 21);
        check(b.reveal(-1, 0) == Move::Invalid && b.reveal(0, 99) == Move::Invalid &&
                  !b.toggle_flag(99, 99),
              "out-of-bounds coordinates are rejected, not clamped");
    }

    section("large boards (the old engine segfaulted at 300x300)");
    {
        Board b(600, 600, 1, 22);
        b.reveal(300, 300);
        check(b.revealed_count() > 300000, "600x600 flood fill completes without recursing",
              std::to_string(b.revealed_count()) + " cells revealed");
    }

    return test::summary("engine");
}
