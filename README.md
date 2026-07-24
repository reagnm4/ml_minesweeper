Name: Reagan Moore\
Section: 11446\
UFL email: moorereagan@ufl.edu\
System: Windows 11\
Compiler: MinGW GCC\
SFML version: 3.0.2\
IDE: Clion\
Other notes: Base commit of minesweeper in C++, plan to add models that use matrix operations to optimize solutions

---

# Minesweeper NN Framework

A from-scratch C++17 neural-network framework — forward pass, reverse-mode
backpropagation, and gradient-based training implemented by hand with **no ML
libraries** — verified by gradient checking and evaluated on Minesweeper
against a deterministic constraint-solver baseline.

Built to the stage order in `SPEC.md`. Every stage has a blocking verification
gate, and every gate below currently passes.

## Results

| Gate | Requirement | Result |
|---|---|---|
| **0** | Solver win rate over 1000 random games | 79.9% beginner · 46.5% intermediate · 1.7% expert · 75.2% repo config |
| **1** | Analytic vs numerical gradient, max relative error < 1e-6 | **2.13e-07** across every op, activation, loss and 3 end-to-end networks |
| **2a** | XOR loss drops to near zero | **3.96e-30** with plain SGD; 10/10 seeds converge |
| **2b** | Val accuracy beats majority baseline by a clear margin | **76.96% vs 62.32%** (+14.6 points) |

### Gate 0 — constraint-solver baseline

1000 games per difficulty, seed 20240724, single-cell logic only (no subset
reasoning, no global mine-count constraint):

| Difficulty | Win rate | Guesses/game |
|---|---|---|
| beginner 9×9 / 10 | 79.9% | 3.09 |
| intermediate 16×16 / 40 | 46.5% | 4.95 |
| expert 16×30 / 99 | 1.7% | 7.09 |
| repo config 16×26 / 50 | 75.2% | 3.19 |

Expert is brutal for pure single-cell logic — that is the honest number, not a
bug. The gate also asserts the solver *never* dies on a deterministic move,
only on a guess; if a deduction ever revealed a mine, the win rate would be
meaningless.

### Gate 1 — gradient checking

The worst relative error, 2.13e-07, is 3.3e-11 of absolute finite-difference
noise sitting on a gradient of size 1.5e-04 — the same absolute noise as every
other op, only divided by a smaller number. Every other check lands between
1e-09 and 1e-11.

Three independent routes to the same derivatives agree:

1. `backward()` — the symbolic chain rule.
2. Central differences, `(f(x+eps) - f(x-eps)) / (2*eps)`.
3. A throwaway NumPy oracle (`tests/oracle/`) with a separately hand-written
   backward pass — agrees with the C++ to **7.4e-16**.

### Gate 2b — supervised mine prediction

30 000 training samples from 2 780 solver-played games, 8 000 validation
samples from a **disjoint** set of games. Input is a 5×5 patch one-hot encoded
over 11 channels (0–8 = revealed count, unknown, off-board). Network is
275 → 48 → 1, ReLU, BCE-with-logits, Adam.

```
val accuracy 76.96%   baseline 62.32%   mine precision 74.51%   mine recall 59.06%
```

Two details that keep this honest:

- **Flagged cells encode identically to hidden cells.** Every flag the solver
  places is provably correct, so a "is flagged" feature would hand over the
  answer. There is an explicit test for this.
- **Already-flagged cells are excluded from sampling** by default. They stay on
  the frontier for the rest of the game, so sampling them re-samples the same
  proven mine every turn and inflates the positive rate from 38% to 57%. Set
  `DatasetConfig::include_flagged` to see that variant.

Some frontier cells are genuinely undecidable from local information, so there
is an irreducible error floor well below 100%.

## Layout

```
engine/    Minesweeper rules, headless and copyable       (Stage 0)
solver/    Constraint solver + probability guess          (Stage 0)
tensor/    Matrix type: matmul, add, hadamard, transpose  (Stage 1)
autograd/  Reverse-mode graph, backward(), grad checking  (Stage 1)
nn/        Dense layer, activations, MLP container        (Stage 2)
optim/     SGD -> momentum -> Adam                        (Stage 2)
train/     Dataset builder                                (Stage 2)
rl/        (empty until Stage 3)
tests/     Per-stage gate checks
src/       The original SFML game
```

## Building

The framework has no third-party dependencies. The SFML game is a separate,
optional target.

```bash
cmake -S . -B build -DMS_BUILD_GUI=OFF   # framework + gates only
cmake --build build -j
cd build && ctest --output-on-failure
```

`-DMS_BUILD_GUI=ON` (the default) additionally fetches SFML 3.0.2 and builds
the game. Each gate is its own binary in `build/bin/` and can be read on its
own:

```bash
./build/bin/gate0_baseline    ./build/bin/gate1_gradcheck
./build/bin/gate2a_xor        ./build/bin/gate2b_mines
```

## Design notes

**Why `double` everywhere.** Gate 1 demands agreement to better than 1e-6. A
central difference divides by `2*eps = 2e-5`, which would amplify float's ~1e-7
representation error to ~1e-2. Single precision cannot pass this gate.

**Why `add_bias` is its own op.** `x @ W + b` needs one bias row shared down the
batch. Rather than a general broadcasting engine (deferred, spec §6), the
broadcast is a single explicit op whose gradient — a sum over the batch — is
the transpose of the implicit copy. It is gradient-checked like everything else.

**Why the backward pass is iterative.** The topological walk uses an explicit
stack rather than recursion, so network depth never lands on the C++ stack.

**Learning rate.** `tests/gate2a_xor.cpp` deliberately shows the same network,
same seed, converging at lr=0.1 and diverging at lr=0.5 — the O(lr²) curvature
term dropped by the first-order argument in `optim/Optim.cpp` made visible.

## Bugs found and fixed in the original engine

The original `src/Board.cpp` was stress-tested headlessly before any ML work
started (SFML stubbed out, so the real game logic ran under test). Seven bugs
were found and fixed:

1. **Same board every reset.** `randomizeMines()` built a fresh `mt19937`
   seeded from `time(nullptr)` on *every call*, so every `reset()` within the
   same second dealt an identical board — 3 consecutive resets produced
   byte-identical layouts. Now seeded once per `Board` from `random_device`.
2. **No first-click safety.** 11.5% of first clicks hit a mine, matching the
   12% mine density. Mines are now placed on the first reveal, keeping the
   clicked cell and its neighbours clear.
3. **Infinite loop on bad config.** Rejection sampling never terminated when
   `mines > rows*cols`, and `Config::load` validated nothing, so a typo in
   `config.cfg` froze the game at launch with no message. The mine count is
   clamped and placement uses partial Fisher-Yates.
4. **Stack overflow on large boards.** The recursive flood fill segfaulted at
   roughly 300×300. Now iterative; verified at 700×700.
5. **Null-texture dereference.** `Board::draw` did `*texManager->get(key)`
   without a null check, so launching from a directory without `files/images/`
   segfaulted instead of erroring.
6. **Clicks just off the board revealed a tile.** `pos.x / 32` truncates toward
   zero, so mouse x in [-31, -1] mapped to column 0.
7. **Leaked `Leaderboard`.** `new` with no matching `delete`; now a
   `unique_ptr`.

The headless `engine/` was written fresh rather than wrapping the SFML board,
because `Tile` holds a `unique_ptr<sf::Sprite>` which makes `Board` impossible
to copy — and snapshotting game states is needed for training data and, later,
replay buffers.

## Not built yet (deliberately)

Stage 3 (DQN), SIMD, broadcasting/views/strides, serialization, CNNs, and
everything else in spec §6. The spec is explicit that a mediocre DQN win rate
is the *expected* outcome of correct code, which is why the math is proven
first and in isolation.
