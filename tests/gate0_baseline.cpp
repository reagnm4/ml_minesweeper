// VERIFICATION GATE 0 (spec section 3, Stage 0).
//
//   "run the solver over 1000 random games; log win rate. This number is the
//    baseline. Record it."
//
// It also checks the one property that makes the baseline trustworthy: the
// solver must only ever die on a *guess*. If a deterministic deduction ever
// reveals a mine, either the solver's logic or the engine's bookkeeping is
// wrong, and the recorded win rate is meaningless.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "../engine/Minesweeper.h"
#include "../solver/Solver.h"
#include "TestUtil.h"

using namespace ms;

namespace {

struct Difficulty {
    const char* name;
    int rows, cols, mines;
};

struct Result {
    int wins = 0;
    int games = 0;
    long long guesses = 0;
    int unsound = 0;  // games lost without ever guessing == a logic bug
    double seconds = 0.0;
};

Result run(const Difficulty& d, int games, uint64_t seed) {
    Result out;
    out.games = games;
    Board b(d.rows, d.cols, d.mines, seed);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < games; ++i) {
        b.reset();
        const SolveStats s = play(b);
        out.wins += s.won ? 1 : 0;
        out.guesses += s.guesses;
        // Lost, but not on a guess => a deduction revealed a mine.
        if (!s.won && !s.died_on_guess) ++out.unsound;
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

std::string pct(int n, int d) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f%%", d ? 100.0 * n / d : 0.0);
    return buf;
}

}  // namespace

int main() {
    constexpr int kGames = 1000;
    constexpr uint64_t kSeed = 20240724;

    const std::vector<Difficulty> difficulties = {
        {"beginner      9x9   /10", 9, 9, 10},
        {"intermediate 16x16  /40", 16, 16, 40},
        {"expert       16x30  /99", 16, 30, 99},
        {"repo config  16x26  /50", 16, 26, 50},
    };

    std::printf("Gate 0 baseline -- deterministic single-cell constraint solver\n");
    std::printf("%d games per difficulty, seed %llu\n\n", kGames,
                static_cast<unsigned long long>(kSeed));
    std::printf("%-26s %8s %10s %14s %8s\n", "difficulty", "wins", "win rate", "guesses/game",
                "time");
    std::printf("%s\n", std::string(72, '-').c_str());

    std::vector<Result> results;
    for (const auto& d : difficulties) {
        const Result r = run(d, kGames, kSeed);
        results.push_back(r);
        std::printf("%-26s %8d %10s %14.2f %7.2fs\n", d.name, r.wins, pct(r.wins, r.games).c_str(),
                    static_cast<double>(r.guesses) / r.games, r.seconds);
        std::fflush(stdout);
    }
    std::printf("\n");

    test::section("gate 0 checks");
    int unsound_total = 0;
    for (const auto& r : results) unsound_total += r.unsound;
    test::check(unsound_total == 0,
                "the solver never dies on a deterministic move (only on guesses)",
                std::to_string(unsound_total) + " unsound losses");

    // A win rate of exactly 0 would mean the solver is not working at all; a
    // win rate of 100% on expert would mean the engine is leaking ground truth.
    bool plausible = true;
    for (size_t i = 0; i < results.size(); ++i)
        if (results[i].wins == 0 || results[i].wins == results[i].games) plausible = false;
    test::check(plausible, "win rates are neither 0% nor 100% on every difficulty");

    // Reproducibility: the whole point of seeding the engine.
    const Result again = run(difficulties[0], kGames, kSeed);
    test::check(again.wins == results[0].wins && again.guesses == results[0].guesses,
                "re-running with the same seed reproduces the baseline exactly");

    const Result other_seed = run(difficulties[0], kGames, kSeed + 1);
    test::check(other_seed.wins != results[0].wins || other_seed.guesses != results[0].guesses,
                "a different seed gives a genuinely different run");

    std::printf("\nBASELINE TO RECORD:\n");
    for (size_t i = 0; i < results.size(); ++i)
        std::printf("  %-26s %s\n", difficulties[i].name, pct(results[i].wins, results[i].games).c_str());

    return test::summary("gate 0");
}
