"""Small statistical helpers shared by the analysis scripts and the
experiment suite.

These used to live as private functions inside ``scripts/analyze_agreement.py``
and were imported by ``experiments/common.py`` -- a dependency from the
library/experiment layer onto a CLI script, and onto *private* names. They are
pure functions of plain lists, so their canonical home is here; both former
importers now depend on ``src.stats`` instead.
"""

from __future__ import annotations

import math
from typing import List, Optional


def pearson_r(xs: List[Optional[float]], ys: List[Optional[float]]) -> Optional[float]:
    """Pure-python Pearson correlation coefficient.

    Returns ``None`` if fewer than 2 paired points or either series has zero
    variance. None entries on either side are dropped pairwise, so failed
    baseline rows do not silently corrupt the correlation.
    """
    def finite(value):
        try:
            return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None
    xs2, ys2 = zip(*pairs)
    n = len(xs2)
    mean_x = sum(xs2) / n
    mean_y = sum(ys2) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs2, ys2))
    var_x = sum((x - mean_x) ** 2 for x in xs2)
    var_y = sum((y - mean_y) ** 2 for y in ys2)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def cohens_kappa(a: List[Optional[bool]], b: List[Optional[bool]]) -> Optional[float]:
    """Binary Cohen's kappa (chance-corrected agreement) between two labelers.

    Returns ``None`` when no paired points exist. A 1.0 on a constant pair of
    series is technically a 0/0 convention; it is returned as 1.0 here because
    both labelers agreeing on identical labels *is* perfect agreement.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for x, y in pairs if x == y)
    p_o = agree / n
    # Expected agreement from each labeler's marginal distribution.
    a_true = sum(1 for x, _ in pairs if x)
    b_true = sum(1 for _, y in pairs if y)
    p_e = (a_true / n) * (b_true / n) + (1 - a_true / n) * (1 - b_true / n)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)
