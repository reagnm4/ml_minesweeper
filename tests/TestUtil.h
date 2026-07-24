// Tiny assertion harness. No test framework: the spec forbids pulling in
// libraries, and a pass/fail counter is all the gates need.
#pragma once

#include <cstdio>
#include <string>

namespace test {

inline int g_failures = 0;
inline int g_checks = 0;

inline void check(bool ok, const std::string& name, const std::string& detail = "") {
    ++g_checks;
    if (!ok) ++g_failures;
    std::printf("[%s] %s%s%s\n", ok ? " PASS " : "*FAIL*", name.c_str(),
                detail.empty() ? "" : "  --  ", detail.c_str());
    std::fflush(stdout);
}

inline void note(const std::string& msg) {
    std::printf("        %s\n", msg.c_str());
    std::fflush(stdout);
}

inline void section(const std::string& title) {
    std::printf("\n=== %s ===\n", title.c_str());
    std::fflush(stdout);
}

// Returns the process exit code: 0 when every check passed.
inline int summary(const std::string& what) {
    std::printf("\n%s: %d/%d checks passed%s\n", what.c_str(), g_checks - g_failures, g_checks,
                g_failures ? "  <-- FAILURES" : "");
    std::fflush(stdout);
    return g_failures == 0 ? 0 : 1;
}

}  // namespace test
