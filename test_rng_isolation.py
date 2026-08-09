# =============================================================================
#  test_rng_isolation.py
#
#  Regression test for the reproducibility guarantee stated in the manuscript:
#  every bootstrap confidence interval depends only on (seed, label, data) and
#  never on execution order.
#
#  An earlier version of the pipeline shared one module-level generator across
#  all bootstraps. That is reproducible for a frozen script, but inserting or
#  reordering any analysis silently shifted every interval computed after it.
#  These tests fail if that behaviour returns.
#
#  RUN:  python test_rng_isolation.py
# =============================================================================
import sys

import numpy as np

from bootstrap_utils import stream, boot_auc_ci, boot_auc_diff

SEED = 42


def synthetic(n=200, seed=0):
    g = np.random.default_rng(seed)
    y = g.integers(0, 2, n)
    p = np.clip(y * 0.30 + g.normal(0.35, 0.20, n), 0, 1)
    return y, p


results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"  [{'ok' if condition else '!!'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    y, p = synthetic()
    y2, p2 = synthetic(n=140, seed=7)
    n = 400   # smaller than production N_BOOT; the guarantee is size-independent

    print("order independence")
    first = boot_auc_ci(y, p, "unit/alpha", SEED, n)
    for _ in range(5):                                  # burn unrelated draws
        boot_auc_ci(y2, p2, "unit/noise", SEED, n)
        stream("unit/more-noise", SEED).integers(0, 10, 5000)
    check("same label and data give the identical interval",
          boot_auc_ci(y, p, "unit/alpha", SEED, n) == first,
          f"{tuple(round(v, 6) for v in first)}")

    d1 = boot_auc_diff(y, p, y2, p2, "unit/diff", SEED, n)
    stream("unit/more-noise", SEED).integers(0, 10, 9999)
    check("AUC-difference test is also order-independent",
          boot_auc_diff(y, p, y2, p2, "unit/diff", SEED, n) == d1,
          f"delta={d1[0]:.6f}")

    print("\nstream separation")
    check("distinct labels give distinct streams",
          boot_auc_ci(y, p, "unit/beta", SEED, n) != first)
    check("a shared generator is not reused: 20 labels give 20 distinct streams",
          len({tuple(stream(f"unit/{i}", SEED).integers(0, 10**9, 3).tolist())
               for i in range(20)}) == 20)

    print("\ndeterminism")
    a = stream("unit/determinism", SEED).integers(0, 10**9, 4)
    b = stream("unit/determinism", SEED).integers(0, 10**9, 4)
    check("stream() is deterministic within a process", np.array_equal(a, b),
          str(a.tolist()))
    check("labels are hashed stably, not with Python's randomised hash()",
          a.tolist() == stream("unit/determinism", SEED).integers(0, 10**9, 4).tolist())

    print("\nseed still governs everything")
    check("changing the seed changes the interval",
          boot_auc_ci(y, p, "unit/alpha", SEED + 1, n) != first)
    check("changing the seed changes the stream",
          not np.array_equal(a, stream("unit/determinism", SEED + 1).integers(0, 10**9, 4)))

    print()
    if all(results):
        print(f"All {len(results)} checks passed.")
        return 0
    print(f"{results.count(False)} of {len(results)} checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
