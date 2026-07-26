// Stage 3, component 4: the experience replay buffer.
//
// Why this exists at all. Consecutive Minesweeper moves within one game are
// enormously correlated -- the board barely changes -- and training a network
// on a stream of correlated samples makes the gradient estimate biased toward
// whatever situation the agent happens to be in right now. Worse, the network
// being trained is also the one generating the data, so a small change in
// behaviour changes the data distribution, which changes the network. Replay
// breaks both loops: transitions are stored, then drawn uniformly at random,
// so a training batch mixes many games and many stages of learning.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace ms {

struct Transition {
    // Board views (the int8 codes from Board::state()), not the expanded
    // one-hot encoding. A 6x6 view is 36 bytes; its one-hot form is 360
    // doubles, or 2880 bytes. Storing views and expanding at sample time makes
    // a 50k-transition buffer cost about 4 MB instead of about 290 MB.
    std::vector<int8_t> state;
    std::vector<int8_t> next_state;
    int action = -1;
    double reward = 0.0;
    bool done = false;
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity) : capacity_(capacity) { data_.reserve(capacity); }

    void push(Transition t) {
        if (static_cast<int>(data_.size()) < capacity_) {
            data_.push_back(std::move(t));
        } else {
            // Circular overwrite: the oldest transition goes first, so the
            // buffer tracks a moving window of recent behaviour.
            data_[next_] = std::move(t);
        }
        next_ = (next_ + 1) % capacity_;
    }

    int size() const { return static_cast<int>(data_.size()); }
    int capacity() const { return capacity_; }
    bool ready(int batch) const { return size() >= batch; }

    const Transition& at(int i) const { return data_[i]; }

    // Uniform sampling *with* replacement. With a buffer of tens of thousands
    // and a batch of 64, collisions are rare enough not to matter, and this
    // avoids the bookkeeping of sampling without replacement.
    void sample(int batch, std::mt19937_64& rng, std::vector<int>& out) const {
        out.clear();
        out.reserve(batch);
        std::uniform_int_distribution<int> pick(0, size() - 1);
        for (int i = 0; i < batch; ++i) out.push_back(pick(rng));
    }

private:
    int capacity_;
    int next_ = 0;
    std::vector<Transition> data_;
};

}  // namespace ms
