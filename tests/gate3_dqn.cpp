// VERIFICATION GATE 3 (spec section 3, Stage 3) -- optional, and last.
//
//   "DQN win rate on the small board rises meaningfully above random over
//    training. Compare against the Stage-0 baseline honestly -- matching or
//    approaching it is a strong result; beating a good CSP solver is *not*
//    expected and is not the success criterion."
//
// The threshold is fixed here, before any run, so it cannot be tuned to
// whatever the agent happens to score: five times the measured random
// baseline. Both baselines below were measured, not assumed -- see the table
// printed at startup.
//
// Everything this file exercises below the RL layer has already been proven:
// Gate 1 verified the gradients, Gate 2 verified that the whole training loop
// learns. So if this gate fails, the fault is in rl/, which is exactly the
// property the spec's build order was designed to buy.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../rl/Dqn.h"
#include "../solver/Solver.h"
#include "TestUtil.h"

using namespace ms;
using test::check;
using test::note;
using test::section;

namespace {

// 1000 games rather than 500: at a ~10% win rate a 500-game evaluation has a
// standard error of 1.3 points, so two consecutive readings can differ by 5
// points on an unchanged policy. That is enough noise to flip a threshold.
constexpr int kEvalGames = 1000;

// The gate is judged on the mean of the last few evaluations rather than the
// single final one, for the same reason. This tightens the *measurement*; the
// 5x threshold itself is unchanged.
constexpr int kFinalWindow = 3;
constexpr uint64_t kEvalSeed = 987654321;

// "Meaningfully above random", fixed in advance.
constexpr double kMeaningfulMultiple = 5.0;

double random_baseline(const DqnConfig& cfg, int games, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Board b(cfg.rows, cfg.cols, cfg.mines, seed);
    int wins = 0;
    for (int g = 0; g < games; ++g) {
        b.reset();
        while (!b.is_over()) {
            const std::vector<int> legal = legal_actions(b);
            if (legal.empty()) break;
            const int a = legal[rng() % legal.size()];  // one draw, not two
            b.reveal(a / cfg.cols, a % cfg.cols);
        }
        wins += b.is_won();
    }
    return 100.0 * wins / games;
}

double solver_baseline(const DqnConfig& cfg, int games, uint64_t seed) {
    Board b(cfg.rows, cfg.cols, cfg.mines, seed);
    int wins = 0;
    for (int g = 0; g < games; ++g) {
        b.reset();
        wins += play(b).won;
    }
    return 100.0 * wins / games;
}

}  // namespace

int main(int argc, char** argv) {
    DqnConfig cfg;
    cfg.rows = 6;
    cfg.cols = 6;
    cfg.mines = 5;
    cfg.hidden = 128;
    cfg.lr = 5e-4;
    cfg.gamma = 0.95;
    cfg.batch = 64;
    cfg.warmup = 2000;
    cfg.target_sync = 500;
    cfg.eps_decay_steps = 30000;
    cfg.seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 20240724;

    const int train_steps = argc > 1 ? std::atoi(argv[1]) : 100000;
    const int eval_every = std::max(1, train_steps / 15);

    section("baselines (measured, not assumed)");
    const double rnd = random_baseline(cfg, 20000, 12345);
    const double sol = solver_baseline(cfg, 20000, 12345);
    std::printf("  board                 %dx%d with %d mines (%d safe cells)\n", cfg.rows,
                cfg.cols, cfg.mines, cfg.rows * cfg.cols - cfg.mines);
    std::printf("  random policy         %.2f%%\n", rnd);
    std::printf("  Stage-0 solver        %.2f%%   <- the honest ceiling, not the target\n", sol);
    std::printf("  gate threshold        %.2f%%   (%.0fx random, fixed in advance)\n\n",
                rnd * kMeaningfulMultiple, kMeaningfulMultiple);

    section("training");
    std::printf("  %d steps, hidden %d, lr %g, gamma %g, batch %d, target sync %d\n\n",
                train_steps, cfg.hidden, cfg.lr, cfg.gamma, cfg.batch, cfg.target_sync);

    DqnAgent agent(cfg);
    std::mt19937_64 rng(cfg.seed);
    Board board(cfg.rows, cfg.cols, cfg.mines, cfg.seed);

    const EvalResult initial = evaluate(agent, cfg, kEvalGames, kEvalSeed);
    std::printf("  %8s %10s %8s %10s %12s\n", "step", "epsilon", "loss", "win rate", "revealed");
    std::printf("  %s\n", std::string(56, '-').c_str());
    std::printf("  %8d %10.3f %8s %9.2f%% %12.2f\n", 0, cfg.eps_start, "-", initial.win_rate,
                initial.mean_revealed);
    std::fflush(stdout);

    std::vector<double> curve;
    curve.push_back(initial.win_rate);
    double best_win = initial.win_rate;
    double running_loss = 0.0;
    int loss_count = 0;

    const auto t0 = std::chrono::steady_clock::now();

    for (int stepi = 1; stepi <= train_steps; ++stepi) {
        if (board.is_over()) board.reset();

        const double eps = epsilon_at(cfg, stepi);
        const int action = agent.select_action(board, eps, rng);
        if (action < 0) { board.reset(); continue; }

        Transition t;
        t.state = board.state();
        t.action = action;
        const Step s = step(board, action, cfg.reward);
        t.next_state = board.state();
        t.reward = s.reward;
        t.done = s.done;
        agent.replay().push(std::move(t));

        if (stepi >= cfg.warmup) {
            const double loss = agent.train_step(rng);
            if (loss >= 0.0) { running_loss += loss; ++loss_count; }
        }
        if (stepi % cfg.target_sync == 0) agent.sync_target();

        if (stepi % eval_every == 0) {
            const EvalResult e = evaluate(agent, cfg, kEvalGames, kEvalSeed);
            curve.push_back(e.win_rate);
            best_win = std::max(best_win, e.win_rate);
            std::printf("  %8d %10.3f %8.4f %9.2f%% %12.2f\n", stepi, eps,
                        loss_count ? running_loss / loss_count : 0.0, e.win_rate,
                        e.mean_revealed);
            std::fflush(stdout);
            running_loss = 0.0;
            loss_count = 0;
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const EvalResult final_eval = evaluate(agent, cfg, kEvalGames, kEvalSeed);

    // The statistic the gate is judged on: the mean of the last kFinalWindow
    // evaluations, i.e. 3000 games of a settled policy rather than 1000 games
    // of whatever the last update happened to produce.
    const int window = std::min<int>(kFinalWindow, static_cast<int>(curve.size()));
    double settled = 0.0;
    for (int i = 0; i < window; ++i) settled += curve[curve.size() - 1 - i];
    settled /= window;

    std::printf("\n  trained in %.1fs (%.0f steps/s)\n", secs, train_steps / secs);
    std::printf("  settled win rate %.2f%%  (mean of last %d evals, %d games each)\n", settled,
                window, kEvalGames);
    std::printf("  final eval %.2f%%, best eval %.2f%%\n", final_eval.win_rate, best_win);
    std::printf("  mean cells revealed %.2f of %d safe\n", final_eval.mean_revealed,
                cfg.rows * cfg.cols - cfg.mines);

    section("gate 3 checks");
    {
        char buf[240];

        std::snprintf(buf, sizeof buf, "settled %.2f%% vs random %.2f%% (%.1fx, need %.0fx)",
                      settled, rnd, rnd > 0 ? settled / rnd : 0.0, kMeaningfulMultiple);
        check(settled >= rnd * kMeaningfulMultiple,
              "GATE 3: win rate rises meaningfully above random", buf);

        std::snprintf(buf, sizeof buf, "%.2f%% -> %.2f%%", initial.win_rate, settled);
        check(settled > initial.win_rate,
              "the trained policy beats the untrained network", buf);

        // Guard against a lucky final evaluation: the second half of the curve
        // should sit above the first half, not just its last point.
        const size_t half = curve.size() / 2;
        double first = 0.0, second = 0.0;
        for (size_t i = 0; i < half; ++i) first += curve[i];
        for (size_t i = half; i < curve.size(); ++i) second += curve[i];
        first /= half;
        second /= (curve.size() - half);
        std::snprintf(buf, sizeof buf, "first half mean %.2f%%, second half mean %.2f%%", first,
                      second);
        check(second > first, "the learning curve trends upward, not just its last point", buf);

        std::snprintf(buf, sizeof buf, "%.2f cells of %d", final_eval.mean_revealed,
                      cfg.rows * cfg.cols - cfg.mines);
        check(final_eval.mean_revealed > 1.0, "the agent survives past its first move", buf);
    }

    section("honest comparison against Stage 0");
    {
        std::printf("  DQN            %.2f%%\n", settled);
        std::printf("  random         %.2f%%\n", rnd);
        std::printf("  Stage-0 solver %.2f%%\n\n", sol);
        note("The spec is explicit that beating a constraint solver is not expected and is");
        note("not the success criterion. Minesweeper is NP-complete, and single-cell logic");
        note("already extracts most of the locally-available information. The DQN's job");
        note("here is to demonstrate learning, which is what the checks above measure.");
    }

    return test::summary("gate 3");
}
