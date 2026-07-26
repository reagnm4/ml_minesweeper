// Stage 2b dataset builder: (local board patch) -> (is the centre cell a mine).
//
// Board states come from letting the Stage-0 solver play real games, so the
// positions are ones an agent would actually meet, not uniformly random noise.
// Labels come from the engine's ground truth.
//
// THE ENCODING MUST NOT LEAK THE LABEL. The solver flags cells it has proven
// to be mines, and every flag it places is correct, so a feature that said
// "this cell is flagged" would be handing over the answer. Flagged cells are
// therefore encoded exactly like hidden ones -- see kUnknown below.
#pragma once

#include <cstdint>
#include <vector>

#include "../engine/Minesweeper.h"
#include "../tensor/Matrix.h"

namespace ms {

struct DatasetConfig {
    int rows = 16;
    int cols = 16;
    int mines = 40;

    // Half-width of the square patch. 2 gives a 5x5 window, which is the
    // smallest that lets the network see a numbered cell *and* that cell's
    // other neighbours -- the information single-cell logic actually uses.
    int patch_radius = 2;

    // Frontier cells sampled per turn. Kept small so the samples come from
    // many different games rather than a handful of heavily-sampled ones.
    int samples_per_turn = 2;

    // Whether to sample cells the solver has already flagged. Those are cells
    // it has *proven* to be mines, and they stay on the frontier for the rest
    // of the game, so including them re-samples the same proven mine every
    // turn and pushes the positive rate far above the true mine density.
    // Off by default: the interesting question is the cells still genuinely in
    // doubt.
    bool include_flagged = false;

    int max_games = 20000;
    int max_samples = 30000;
    uint64_t seed = 1;
};

struct Dataset {
    nn::Matrix x;  // (samples, features)
    nn::Matrix y;  // (samples, 1), 1.0 where the centre cell is a mine

    int games_used = 0;
    int positives = 0;

    int samples() const { return y.rows(); }
    int features() const { return x.cols(); }
    // Accuracy of always predicting the more common class.
    double majority_baseline() const;
};

// One-hot channel layout for a single cell of the patch.
//   0..8      revealed, carrying that many adjacent mines
//   kUnknown  hidden OR flagged -- deliberately indistinguishable
//   kOffBoard the patch hangs off the edge of the board
constexpr int kUnknown = 9;
constexpr int kOffBoard = 10;
constexpr int kChannels = 11;

int features_for(int patch_radius);

// Encode the patch centred on (r, c) into `out`, which must have
// features_for(patch_radius) elements. Exposed so tests can check the encoding
// directly rather than trusting it.
void encode_patch(const Board& b, int r, int c, int patch_radius, double* out);

Dataset build_dataset(const DatasetConfig& cfg);

// Gather `count` rows of `src`, selected by order[begin .. begin+count). Used
// to cut mini-batches out of a shuffled dataset without copying it.
nn::Matrix take_rows(const nn::Matrix& src, const std::vector<int>& order, int begin, int count);

}  // namespace ms
