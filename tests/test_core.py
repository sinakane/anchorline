import numpy as np
import pandas as pd

from anchorline._core import STATES, add_pseudotime, rank01, robust_z


def test_rank01():
    r = rank01([5.0, np.nan, 1.0, 3.0])
    assert np.isnan(r[1]) and r[2] == 0 and r[0] == 1 and r[3] == 0.5


def test_robust_z_centres_on_median():
    z = robust_z([1, 2, 3, 4, 100])
    assert z[2] == 0


def test_pseudotime_independent_of_mu_order():
    """regression: reference means stored in an h5ad come back alphabetically; positions must still match states"""
    rng = np.random.default_rng(0)
    S = pd.DataFrame(rng.uniform(0.5, 1, (20, len(STATES))), columns=list(STATES))
    mu, sd = S.mean(), S.std()
    a = add_pseudotime(S, mu, sd).pseudotime
    b = add_pseudotime(S, mu.sort_index(), sd.sort_index()).pseudotime     # alphabetical, like uns
    assert np.allclose(a, b)
    hot = S.copy()
    hot.loc[0, :] = S.mean()
    hot.loc[0, "Mono-like-Top100"] = 5                                       # strongly monocytic sample
    assert add_pseudotime(hot, mu.sort_index(), sd.sort_index()).pseudotime[0] > 3.5
