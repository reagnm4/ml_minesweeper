// VERIFICATION GATE 2 (spec section 3, Stage 2b).
//
//   "using boards solved by the Stage-0 solver, build a dataset of (local
//    board patch) -> (is this cell a mine, 0/1). Train the MLP with binary
//    cross-entropy. Gate 2: training loss decreases monotonically (smoothed)
//    and validation accuracy beats a predict-the-majority-class baseline by a
//    clear margin."
//
// Two things this test is careful about, because getting either wrong would
// turn the gate into a lie:
//
//   1. Train and validation boards come from disjoint sets of games. Samples
//      drawn from the same game share a mine layout, so splitting by sample
//      would leak the answer across the split.
//   2. The solver's flags are always correct, so the encoding maps flagged and
//      hidden cells to the same channel. There is an explicit check below that
//      this holds -- otherwise the network could score 100% by reading flags.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "../nn/Layers.h"
#include "../optim/Optim.h"
#include "../solver/Solver.h"
#include "../train/Dataset.h"
#include "TestUtil.h"

using namespace nn;
using test::check;
using test::note;
using test::section;

namespace {

constexpr int kEpochs = 25;
constexpr int kBatch = 256;
constexpr int kHidden = 48;
constexpr double kLr = 1e-3;

// "Beats the majority-class baseline by a clear margin": ten percentage
// points, fixed in advance so the threshold cannot be tuned to the result.
constexpr double kClearMargin = 0.10;

struct Eval {
    double loss = 0.0;
    double accuracy = 0.0;
    double mine_recall = 0.0;     // of the true mines, how many were caught
    double mine_precision = 0.0;  // of the cells called mines, how many were
};

Eval evaluate(const MLP& net, const Matrix& x, const Matrix& y) {
    Tensor logits = net.forward(x);
    Eval e;
    e.loss = bce_with_logits_loss(logits, y).item();

    int correct = 0, tp = 0, predicted_pos = 0, actual_pos = 0;
    for (int i = 0; i < y.rows(); ++i) {
        const bool said_mine = logits.value()(i, 0) > 0.0;  // logit > 0 <=> p > 0.5
        const bool is_mine = y(i, 0) > 0.5;
        if (said_mine == is_mine) ++correct;
        if (said_mine) ++predicted_pos;
        if (is_mine) ++actual_pos;
        if (said_mine && is_mine) ++tp;
    }
    e.accuracy = static_cast<double>(correct) / y.rows();
    e.mine_recall = actual_pos ? static_cast<double>(tp) / actual_pos : 0.0;
    e.mine_precision = predicted_pos ? static_cast<double>(tp) / predicted_pos : 0.0;
    return e;
}

}  // namespace

int main() {
    section("dataset");

    ms::DatasetConfig train_cfg;
    train_cfg.seed = 1;
    train_cfg.max_samples = 30000;

    ms::DatasetConfig val_cfg = train_cfg;
    val_cfg.seed = 99991;  // a different seed means a disjoint set of games
    val_cfg.max_samples = 8000;

    const auto t0 = std::chrono::steady_clock::now();
    const ms::Dataset train = ms::build_dataset(train_cfg);
    const ms::Dataset val = ms::build_dataset(val_cfg);
    const double build_secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("  train %d samples from %d games, %.1f%% mines\n", train.samples(),
                train.games_used, 100.0 * train.positives / train.samples());
    std::printf("  val   %d samples from %d games, %.1f%% mines\n", val.samples(),
                val.games_used, 100.0 * val.positives / val.samples());
    std::printf("  %d features per sample (5x5 patch x 11 channels), built in %.1fs\n",
                train.features(), build_secs);

    const double baseline = val.majority_baseline();
    std::printf("  majority-class baseline on validation: %.2f%%\n\n", 100.0 * baseline);

    check(train.samples() > 20000 && val.samples() > 5000, "datasets are large enough to train on");
    check(train.positives > 0 && train.positives < train.samples(),
          "the dataset contains both mines and safe cells");
    check(train.games_used > 100 && val.games_used > 50,
          "samples are spread over many games, not a handful",
          std::to_string(train.games_used) + " train / " + std::to_string(val.games_used) +
              " val games");

    section("encoding hygiene (does the input leak the label?)");
    {
        // Build a position, flag a cell the solver has proven to be a mine,
        // and confirm the encoding is byte-identical to leaving it hidden.
        ms::Board b(16, 16, 40, 4242);
        b.reveal(8, 8);
        ms::solve_to_fixpoint(b);

        int flagged_r = -1, flagged_c = -1;
        for (int r = 0; r < b.rows() && flagged_r < 0; ++r)
            for (int c = 0; c < b.cols() && flagged_r < 0; ++c)
                if (b.is_flagged(r, c)) { flagged_r = r; flagged_c = c; }

        if (flagged_r < 0) {
            check(false, "found a flagged cell to test with");
        } else {
            const int n = ms::features_for(2);
            std::vector<double> with_flag(n), without_flag(n);

            // Encode a patch that contains the flagged cell.
            const int cr = flagged_r, cc = std::max(0, flagged_c - 1);
            ms::encode_patch(b, cr, cc, 2, with_flag.data());
            b.toggle_flag(flagged_r, flagged_c);  // back to plain hidden
            ms::encode_patch(b, cr, cc, 2, without_flag.data());

            check(with_flag == without_flag,
                  "a flagged cell encodes identically to a hidden one",
                  "otherwise the solver's always-correct flags would be the answer key");
        }
    }
    {
        // Every cell of the patch must set exactly one channel.
        ms::Board b(16, 16, 40, 77);
        b.reveal(0, 0);
        const int n = ms::features_for(2);
        std::vector<double> f(n);
        ms::encode_patch(b, 1, 1, 2, f.data());

        bool one_hot = true;
        for (int cell = 0; cell < 25; ++cell) {
            double s = 0.0;
            for (int ch = 0; ch < ms::kChannels; ++ch) s += f[cell * ms::kChannels + ch];
            if (std::fabs(s - 1.0) > 1e-12) one_hot = false;
        }
        check(one_hot, "every patch cell activates exactly one channel");
    }

    section("training");

    std::mt19937_64 rng(20240724);
    // Logits out, no output activation: bce_with_logits_loss applies the
    // sigmoid internally in its numerically stable form.
    MLP net({train.features(), kHidden, 1}, Activation::ReLU, Activation::None, rng);
    Adam opt(net.parameters(), kLr);

    const Eval untrained = evaluate(net, val.x, val.y);
    std::printf("  before training: val accuracy %.2f%%, loss %.4f\n", 100.0 * untrained.accuracy,
                untrained.loss);

    std::vector<int> order(train.samples());
    std::iota(order.begin(), order.end(), 0);

    std::vector<double> epoch_losses;
    double smoothed = -1.0;
    bool monotone = true;
    double worst_increase = 0.0;

    const auto train_start = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);

        double total = 0.0;
        int batches = 0;
        for (int start = 0; start + kBatch <= train.samples(); start += kBatch) {
            const Matrix xb = ms::take_rows(train.x, order, start, kBatch);
            const Matrix yb = ms::take_rows(train.y, order, start, kBatch);

            opt.zero_grad();
            Tensor loss = bce_with_logits_loss(net.forward(xb), yb);
            loss.backward();
            opt.step();

            total += loss.item();
            ++batches;
        }

        const double epoch_loss = total / batches;
        epoch_losses.push_back(epoch_loss);

        if (smoothed < 0.0) {
            smoothed = epoch_loss;
        } else {
            const double next = 0.7 * smoothed + 0.3 * epoch_loss;
            if (next > smoothed) worst_increase = std::max(worst_increase, next - smoothed);
            if (next > smoothed + 1e-6) monotone = false;
            smoothed = next;
        }

        if (epoch % 5 == 0 || epoch == kEpochs - 1) {
            const Eval e = evaluate(net, val.x, val.y);
            std::printf("  epoch %2d  train loss %.4f   val loss %.4f  val acc %.2f%%\n", epoch,
                        epoch_loss, e.loss, 100.0 * e.accuracy);
            std::fflush(stdout);
        }
    }
    const double train_secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - train_start).count();

    const Eval final_eval = evaluate(net, val.x, val.y);
    std::printf("\n  trained in %.1fs\n", train_secs);
    std::printf("  final: val accuracy %.2f%%  (baseline %.2f%%)  mine precision %.2f%%  "
                "mine recall %.2f%%\n",
                100.0 * final_eval.accuracy, 100.0 * baseline, 100.0 * final_eval.mine_precision,
                100.0 * final_eval.mine_recall);

    section("gate 2 checks");
    {
        char buf[220];
        std::snprintf(buf, sizeof buf, "%.4f -> %.4f, worst smoothed uptick %.2e",
                      epoch_losses.front(), epoch_losses.back(), worst_increase);
        check(monotone, "GATE 2: smoothed training loss decreases monotonically", buf);

        std::snprintf(buf, sizeof buf, "val accuracy %.2f%% vs baseline %.2f%% (+%.2f points, "
                                       "need +%.0f)",
                      100.0 * final_eval.accuracy, 100.0 * baseline,
                      100.0 * (final_eval.accuracy - baseline), 100.0 * kClearMargin);
        check(final_eval.accuracy >= baseline + kClearMargin,
              "GATE 2: validation accuracy beats the majority baseline by a clear margin", buf);

        std::snprintf(buf, sizeof buf, "%.4f -> %.4f", epoch_losses.front(), epoch_losses.back());
        check(epoch_losses.back() < epoch_losses.front() * 0.7, "training loss fell substantially",
              buf);

        std::snprintf(buf, sizeof buf, "untrained %.2f%% -> trained %.2f%%",
                      100.0 * untrained.accuracy, 100.0 * final_eval.accuracy);
        check(final_eval.accuracy > untrained.accuracy + 0.05,
              "training is what produced the accuracy, not the architecture", buf);

        // A model that never predicts "mine" would score the baseline exactly.
        // Requiring real recall proves it learned the minority class.
        std::snprintf(buf, sizeof buf, "recall %.2f%%, precision %.2f%%",
                      100.0 * final_eval.mine_recall, 100.0 * final_eval.mine_precision);
        check(final_eval.mine_recall > 0.5 && final_eval.mine_precision > 0.5,
              "the model actually identifies mines rather than always saying 'safe'", buf);
    }

    return test::summary("gate 2");
}
