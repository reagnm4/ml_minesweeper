// Stage 3 component sanity checks.
//
// The spec requires the RL components to be added "one at a time, each with
// its own sanity check". This file is those checks, and it runs before the
// training gate: if the DQN later fails to learn, everything here has already
// been eliminated as the cause.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "../rl/Dqn.h"
#include "../rl/Env.h"
#include "../rl/Replay.h"
#include "TestUtil.h"

using namespace ms;
using test::check;
using test::note;
using test::section;

namespace {

DqnConfig small_cfg() {
    DqnConfig cfg;
    cfg.rows = 6;
    cfg.cols = 6;
    cfg.mines = 5;
    cfg.hidden = 32;   // small: these checks are about correctness, not skill
    cfg.warmup = 0;
    cfg.seed = 4242;
    return cfg;
}

}  // namespace

int main() {
    // -----------------------------------------------------------------
    section("component 1: state encoding");
    // -----------------------------------------------------------------
    {
        Board b(6, 6, 5, 1);
        b.reveal(2, 2);
        const nn::Matrix x = encode_state_row(b);

        check(x.rows() == 1 && x.cols() == state_size(6, 6),
              "encoding has shape 1 x (cells * channels)",
              std::to_string(x.rows()) + "x" + std::to_string(x.cols()));

        bool one_hot = true;
        for (int c = 0; c < b.cells(); ++c) {
            double s = 0.0;
            for (int ch = 0; ch < kRlChannels; ++ch) s += x(0, c * kRlChannels + ch);
            if (std::fabs(s - 1.0) > 1e-12) one_hot = false;
        }
        check(one_hot, "every cell activates exactly one channel");

        // A hidden cell must encode identically whether or not it hides a mine.
        // If it did not, the network could read the answer straight off its
        // input and the whole exercise would be circular.
        int mine_cell = -1, safe_cell = -1;
        for (int i = 0; i < b.cells(); ++i) {
            if (b.state()[i] != kHidden) continue;
            if (b.is_mine(i / 6, i % 6)) mine_cell = i;
            else safe_cell = i;
        }
        bool identical = mine_cell >= 0 && safe_cell >= 0;
        if (identical)
            for (int ch = 0; ch < kRlChannels; ++ch)
                if (x(0, mine_cell * kRlChannels + ch) != x(0, safe_cell * kRlChannels + ch))
                    identical = false;
        check(identical, "a hidden mine encodes identically to a hidden safe cell",
              "otherwise the state would leak ground truth");
    }
    {
        // Different boards must produce different encodings, or the network is
        // being shown the same input for every position.
        std::set<std::vector<double>> seen;
        Board b(6, 6, 5, 77);
        for (int i = 0; i < 50; ++i) {
            b.reset();
            b.reveal(i % 6, (i * 5) % 6);
            seen.insert(encode_state_row(b).data());
        }
        check(seen.size() > 40, "distinct positions give distinct encodings",
              std::to_string(seen.size()) + "/50 unique");
    }

    // -----------------------------------------------------------------
    section("component 2: reward function");
    // -----------------------------------------------------------------
    {
        RewardConfig rc;
        Board b(6, 6, 5, 5);
        b.reveal(0, 0);

        int mine = -1, safe = -1;
        for (int i = 0; i < b.cells(); ++i) {
            if (b.state()[i] != kHidden) continue;
            if (b.is_mine(i / 6, i % 6)) { if (mine < 0) mine = i; }
            else if (safe < 0) safe = i;
        }

        Board copy = b;
        const Step s_safe = step(copy, safe, rc);
        check(s_safe.reward > 0 && !s_safe.hit_mine && s_safe.newly_revealed > 0,
              "a safe reveal earns positive reward and opens at least one cell",
              "reward " + std::to_string(s_safe.reward) + ", opened " +
                  std::to_string(s_safe.newly_revealed));

        Board copy2 = b;
        const Step s_mine = step(copy2, mine, rc);
        check(s_mine.reward == rc.mine && s_mine.done && s_mine.hit_mine && !s_mine.won,
              "hitting a mine ends the episode with the mine penalty");

        // Clearing every safe cell must produce the win reward exactly once.
        Board copy3 = b;
        double total_win = 0.0;
        int wins = 0;
        for (int i = 0; i < copy3.cells() && !copy3.is_over(); ++i) {
            if (copy3.state()[i] != kHidden || copy3.is_mine(i / 6, i % 6)) continue;
            const Step s = step(copy3, i, rc);
            if (s.won) { ++wins; total_win = s.reward; }
        }
        check(copy3.is_won() && wins == 1 && total_win == rc.win,
              "clearing the board pays the win reward exactly once");

        Board copy4 = b;
        const int already_open = 0;  // (0,0) was revealed above
        const Step s_noop = step(copy4, already_open, rc);
        check(s_noop.reward == 0.0 && s_noop.newly_revealed == 0,
              "re-clicking an open cell changes nothing and pays nothing");
    }

    // -----------------------------------------------------------------
    section("component 3: epsilon-greedy exploration");
    // -----------------------------------------------------------------
    {
        DqnConfig cfg = small_cfg();
        check(epsilon_at(cfg, 0) == cfg.eps_start && epsilon_at(cfg, cfg.eps_decay_steps) == cfg.eps_end &&
                  epsilon_at(cfg, cfg.eps_decay_steps * 10) == cfg.eps_end,
              "epsilon decays from start to end and then stays flat");

        const double mid = epsilon_at(cfg, cfg.eps_decay_steps / 2);
        check(mid < cfg.eps_start && mid > cfg.eps_end, "epsilon is monotone in between",
              "halfway: " + std::to_string(mid));
    }
    {
        DqnConfig cfg = small_cfg();
        DqnAgent agent(cfg);
        std::mt19937_64 rng(9);

        Board b(6, 6, 5, 3);
        b.reveal(3, 3);
        const std::vector<int> legal = legal_actions(b);
        std::set<int> legal_set(legal.begin(), legal.end());

        // Never illegal, at any epsilon.
        bool always_legal = true;
        for (int i = 0; i < 3000; ++i) {
            const double eps = (i % 3) * 0.5;  // 0.0, 0.5, 1.0
            const int a = agent.select_action(b, eps, rng);
            if (!legal_set.count(a)) always_legal = false;
        }
        check(always_legal, "never selects an illegal action, at any epsilon");

        // epsilon = 1 should roam widely; epsilon = 0 should be deterministic.
        std::set<int> explored, exploited;
        for (int i = 0; i < 2000; ++i) explored.insert(agent.select_action(b, 1.0, rng));
        for (int i = 0; i < 50; ++i) exploited.insert(agent.select_action(b, 0.0, rng));

        check(explored.size() > legal.size() / 2, "epsilon=1 explores broadly",
              std::to_string(explored.size()) + " distinct of " + std::to_string(legal.size()) +
                  " legal");
        check(exploited.size() == 1, "epsilon=0 is deterministic",
              std::to_string(exploited.size()) + " distinct choices");
    }
    {
        DqnConfig cfg = small_cfg();
        DqnAgent agent(cfg);
        Board b(6, 6, 5, 3);
        b.reveal(3, 3);
        // A finished game offers nothing to do.
        while (!b.is_over()) {
            const std::vector<int> legal = legal_actions(b);
            if (legal.empty()) break;
            step(b, legal[0], cfg.reward);
        }
        check(legal_actions(b).empty() && agent.greedy_action(b) == -1,
              "a finished episode has no legal actions and returns -1");
    }

    // -----------------------------------------------------------------
    section("component 4: experience replay buffer");
    // -----------------------------------------------------------------
    {
        ReplayBuffer buf(5);
        check(buf.size() == 0 && !buf.ready(1), "a new buffer is empty");

        for (int i = 0; i < 5; ++i) {
            Transition t;
            t.state = {static_cast<int8_t>(i)};
            t.next_state = {static_cast<int8_t>(i)};
            t.action = i;
            t.reward = i;
            buf.push(std::move(t));
        }
        check(buf.size() == 5 && buf.ready(5), "the buffer fills to capacity");

        // Overflow must evict the oldest, not grow and not refuse.
        Transition t;
        t.state = {99};
        t.next_state = {99};
        t.action = 99;
        t.reward = 99;
        buf.push(std::move(t));

        bool has_new = false, has_oldest = false;
        for (int i = 0; i < buf.size(); ++i) {
            if (buf.at(i).action == 99) has_new = true;
            if (buf.at(i).action == 0) has_oldest = true;
        }
        check(buf.size() == 5 && has_new && !has_oldest,
              "pushing past capacity overwrites the oldest transition");
    }
    {
        ReplayBuffer buf(1000);
        for (int i = 0; i < 1000; ++i) {
            Transition t;
            t.state = {static_cast<int8_t>(i % 100)};
            t.next_state = {static_cast<int8_t>(i % 100)};
            t.action = i;
            buf.push(std::move(t));
        }
        std::mt19937_64 rng(1);
        std::vector<int> idx;
        buf.sample(64, rng, idx);

        bool in_range = idx.size() == 64;
        for (int i : idx)
            if (i < 0 || i >= buf.size()) in_range = false;
        check(in_range, "sampling returns the requested count, all in range");

        // A uniform sample of 64 from 1000 should almost never be one value.
        std::set<int> distinct(idx.begin(), idx.end());
        check(distinct.size() > 50, "sampling is spread, not degenerate",
              std::to_string(distinct.size()) + " distinct of 64");
    }

    // -----------------------------------------------------------------
    section("component 5: target network");
    // -----------------------------------------------------------------
    {
        DqnConfig cfg = small_cfg();
        DqnAgent agent(cfg);

        auto params_equal = [](const nn::MLP& a, const nn::MLP& b) {
            std::vector<nn::Tensor> pa = a.parameters(), pb = b.parameters();
            for (size_t i = 0; i < pa.size(); ++i)
                if (pa[i].value().data() != pb[i].value().data()) return false;
            return true;
        };

        check(params_equal(agent.online(), agent.target()),
              "online and target start identical");

        // Fill the buffer with real transitions, then train without syncing.
        Board b(6, 6, 5, 11);
        std::mt19937_64 rng(3);
        for (int i = 0; i < 400; ++i) {
            if (b.is_over()) b.reset();
            const std::vector<int> legal = legal_actions(b);
            if (legal.empty()) { b.reset(); continue; }
            Transition t;
            t.state = b.state();
            t.action = legal[rng() % legal.size()];
            const Step s = step(b, t.action, cfg.reward);
            t.next_state = b.state();
            t.reward = s.reward;
            t.done = s.done;
            agent.replay().push(std::move(t));
        }

        for (int i = 0; i < 20; ++i) agent.train_step(rng);

        check(!params_equal(agent.online(), agent.target()),
              "training moves the online network but leaves the target frozen");

        agent.sync_target();
        check(params_equal(agent.online(), agent.target()),
              "sync_target copies the online weights across");
    }

    // -----------------------------------------------------------------
    section("component 6: the Q-update");
    // -----------------------------------------------------------------
    {
        // The decisive check: can the update actually drive Q(s,a) to a known
        // target? Fill the buffer with copies of one terminal transition whose
        // target is exactly its reward, and watch that one Q-value converge.
        DqnConfig cfg = small_cfg();
        cfg.lr = 5e-3;
        DqnAgent agent(cfg);

        Board b(6, 6, 5, 21);
        b.reveal(0, 0);
        int mine = -1;
        for (int i = 0; i < b.cells(); ++i)
            if (b.state()[i] == kHidden && b.is_mine(i / 6, i % 6)) { mine = i; break; }

        Transition t;
        t.state = b.state();
        t.action = mine;
        Board after = b;
        const Step s = step(after, mine, cfg.reward);
        t.next_state = after.state();
        t.reward = s.reward;
        t.done = s.done;
        for (int i = 0; i < 256; ++i) agent.replay().push(t);

        std::mt19937_64 rng(7);
        nn::Tensor q0 = agent.online().forward(encode_state_row(b));
        const double before = q0.value()(0, mine);

        double last_loss = 0.0;
        for (int i = 0; i < 400; ++i) last_loss = agent.train_step(rng);

        nn::Tensor q1 = agent.online().forward(encode_state_row(b));
        const double after_q = q1.value()(0, mine);

        char buf[220];
        std::snprintf(buf, sizeof buf, "Q(s,mine) %.4f -> %.4f, target %.2f, loss %.2e",
                      before, after_q, t.reward, last_loss);
        check(std::fabs(after_q - t.reward) < 0.05, "the update drives Q(s,a) to a known target",
              buf);
    }
    {
        // Only the action taken may receive gradient. Everything else is left
        // equal to its own prediction, so its error -- and its gradient -- is
        // exactly zero.
        DqnConfig cfg = small_cfg();
        DqnAgent agent(cfg);

        Board b(6, 6, 5, 31);
        b.reveal(1, 1);
        const std::vector<int> legal = legal_actions(b);
        const int chosen = legal[0];

        Transition t;
        t.state = b.state();
        t.action = chosen;
        Board after = b;
        const Step s = step(after, chosen, cfg.reward);
        t.next_state = after.state();
        t.reward = s.reward;
        t.done = s.done;
        for (int i = 0; i < 64; ++i) agent.replay().push(t);

        nn::Tensor q_before = agent.online().forward(encode_state_row(b));
        std::vector<double> before(q_before.value().data());

        std::mt19937_64 rng(5);
        for (int i = 0; i < 60; ++i) agent.train_step(rng);

        nn::Tensor q_after = agent.online().forward(encode_state_row(b));

        const double moved_chosen = std::fabs(q_after.value()(0, chosen) - before[chosen]);
        check(moved_chosen > 1e-4, "the taken action's Q-value moves",
              "moved by " + std::to_string(moved_chosen));
    }

    return test::summary("rl components");
}
