#!/usr/bin/env bash
# Third route to the same gradients: compare the C++ framework against the
# throwaway NumPy oracle, element by element.
#
# Usage: tests/oracle/compare_oracle.sh <path-to-oracle_dump-binary>
set -euo pipefail

BIN="${1:-build/bin/oracle_dump}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "SKIP: numpy is not installed; the C++ gradient check in gate1_gradcheck"
    echo "      is the binding verification. This oracle is the spec's optional extra."
    exit 0
fi

"$BIN" > /tmp/ms_cpp_grads.txt
python3 "$HERE/mlp_oracle.py" > /tmp/ms_numpy_grads.txt

python3 - "$@" <<'PY'
import sys

def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = tuple(parts[:-1])
            out[key] = float(parts[-1])
    return out

cpp = load('/tmp/ms_cpp_grads.txt')
npy = load('/tmp/ms_numpy_grads.txt')

missing = set(cpp) ^ set(npy)
if missing:
    print(f"*FAIL* the two dumps do not cover the same values: {sorted(missing)[:5]}")
    sys.exit(1)

worst, where = 0.0, None
for key in cpp:
    a, b = cpp[key], npy[key]
    scale = max(abs(a), abs(b), 1e-12)
    rel = abs(a - b) / scale
    if rel > worst:
        worst, where = rel, key

print(f"compared {len(cpp)} values (loss + every gradient element)")
print(f"worst relative disagreement: {worst:.3e} at {where}")
if worst < 1e-12:
    print("[ PASS ] the C++ framework and the NumPy oracle agree to double precision")
    sys.exit(0)
print("[*FAIL*] C++ and NumPy disagree")
sys.exit(1)
PY
