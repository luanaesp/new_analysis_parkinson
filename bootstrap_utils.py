# =============================================================================
#  bootstrap_utils.py
#  Order-independent resampling used by both analysis pipelines.
#
#  WHY THIS EXISTS
#    Every bootstrap in this project draws from its OWN generator, seeded
#    deterministically from the global seed plus a stable textual label. A
#    confidence interval therefore depends only on (seed, label, data) and never
#    on how many random numbers were consumed earlier in the script.
#
#    The earlier implementation shared one module-level generator across all
#    bootstraps. That is reproducible for a fixed script, but fragile: adding,
#    removing or reordering any analysis silently shifted every interval
#    computed after it, so intervals were not stable across revisions of the
#    code. test_rng_isolation.py guards against a regression.
#
#    SHA-256 is used instead of hash() because Python randomises string hashing
#    per process, which would make labels non-reproducible across runs.
# =============================================================================
import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT_DEFAULT = 2000


def stream(label, seed):
    """Return an independent generator for the named analysis."""
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def boot_auc_ci(y, p, label, seed, n=N_BOOT_DEFAULT):
    """Percentile bootstrap 95% CI for a single AUC."""
    g = stream(f"boot_auc_ci/{label}", seed)
    y = np.asarray(y)
    p = np.asarray(p)
    a = []
    for _ in range(n):
        idx = g.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            a.append(roc_auc_score(y[idx], p[idx]))
    if not a:
        return (np.nan, np.nan)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def boot_auc_diff(y1, p1, y2, p2, label, seed, n=N_BOOT_DEFAULT):
    """Unpaired bootstrap of an AUC difference between two independent groups.

    Returns (mean difference, 95% CI, two-sided empirical p-value).
    """
    g = stream(f"boot_auc_diff/{label}", seed)
    y1, p1, y2, p2 = map(np.asarray, (y1, p1, y2, p2))
    diffs = []
    for _ in range(n):
        i1 = g.integers(0, len(y1), len(y1))
        i2 = g.integers(0, len(y2), len(y2))
        if len(np.unique(y1[i1])) > 1 and len(np.unique(y2[i2])) > 1:
            diffs.append(roc_auc_score(y1[i1], p1[i1]) - roc_auc_score(y2[i2], p2[i2]))
    diffs = np.array(diffs)
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return (float(np.mean(diffs)),
            (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            float(min(pval, 1.0)))
