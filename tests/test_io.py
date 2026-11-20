import anndata as ad
import numpy as np
import pandas as pd
import pytest

import anchorline as al


def test_list_dict_signature():
    q = al.io.read_query(["HOXA9", "MEIS1"])
    assert q.kind == "signature" and q.up == ["HOXA9", "MEIS1"] and q.down == []
    q = al.io.read_query({"up": ["A"], "down": ["B"]}, name="x")
    assert q.name == "x" and q.down == ["B"]


def test_builtin_name():
    q = al.io.read_query("Umeda2024_HOXA")
    assert q.kind == "signature" and len(q.up) > 10


def test_signature_csv(tmp_path):
    p = tmp_path / "sig.csv"
    pd.DataFrame({"gene": ["A", "B", "C"], "direction": ["up", "+", "down"]}).to_csv(p, index=False)
    q = al.io.read_query(p)
    assert q.up == ["A", "B"] and q.down == ["C"]


def test_bad_direction(tmp_path):
    p = tmp_path / "sig.csv"
    pd.DataFrame({"gene": ["A"], "direction": ["sideways"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        al.io.read_query(p)


def _cohort(n_genes=3000, n_samples=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.lognormal(2, 1, (n_genes, n_samples)), index=[f"G{i}" for i in range(n_genes)],
                        columns=[f"S{j}" for j in range(n_samples)])


def test_expression_table_with_background(tmp_path):
    M = _cohort()
    p = tmp_path / "cohort.csv"
    M.to_csv(p)
    q = al.io.read_query(p, sample="S2")
    assert q.kind == "sample" and q.background.shape == (6, 3000)
    assert np.allclose(q.expr.values, M["S2"].values, atol=1e-6)


def test_anndata_sample_and_pseudobulk():
    M = _cohort()
    A = ad.AnnData(np.log2(M.T.values + 1), obs=pd.DataFrame({"grp": list("aabbcc")}, index=M.columns),
                   var=pd.DataFrame(index=M.index))
    q = al.io.read_query(A, sample="S1")
    assert q.kind == "sample" and q.background.shape[0] == 6
    q = al.io.read_query(A, groupby="grp", group="b")
    assert q.meta["n_cells"] == 2 and q.background.shape[0] == 3
    assert np.allclose(q.expr.values, np.log2(M[["S2", "S3"]].values + 1).mean(1), atol=1e-5)
