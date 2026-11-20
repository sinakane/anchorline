"""Turn anything into a Query.

    al.io.read_query(["HOXA9", "MEIS1"])                         gene list -> signature (up genes)
    al.io.read_query({"up": [...], "down": [...]})                 signature with down genes
    al.io.read_query("Umeda2024_KMT2Ar_high")                      built-in signature by name (al.sig.list())
    al.io.read_query("my_signature.csv")                           columns gene + direction (up/down), or a single gene column
    al.io.read_query("cohort.tsv", sample="P01")                   genes x samples table; the other samples are the background
    al.io.read_query(series)                                       one expression profile (gene -> value)
    al.io.read_query(adata, sample="P01")                          AnnData, one observation (others = background)
    al.io.read_query(adata, groupby="leiden", group="3")           AnnData pseudobulk of a group (others = background)

Expression can be on any scale (counts, CPM, TPM, log): the profile is converted to within-sample ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import sig as _sig


@dataclass
class Query:
    """A query for `al.tl.match`: either a signature (`up`/`down` gene lists) or a sample profile (`expr`)."""
    name: str
    kind: str                                   # "signature" or "sample"
    up: list = field(default_factory=list)
    down: list = field(default_factory=list)
    expr: pd.Series = None                      # sample: gene -> expression
    background: pd.DataFrame = None             # sample: cohort, samples x genes (optional)
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        if self.kind == "signature":
            return f"Query('{self.name}', signature: {len(self.up)} up, {len(self.down)} down)"
        bg = "none" if self.background is None else f"{self.background.shape[0]} samples"
        return f"Query('{self.name}', sample: {self.expr.notna().sum()} genes, background: {bg})"


_DIR = {"up": "up", "down": "down", "dn": "down", "+": "up", "-": "down", "1": "up", "-1": "down",
        "positive": "up", "negative": "down", "pos": "up", "neg": "down", "true": "up", "false": "down"}


def _signature_from_frame(df, name):
    cols = {c.lower(): c for c in df.columns}
    gcol = next((cols[c] for c in ("gene", "genes", "symbol", "gene_symbol", "hgnc_symbol") if c in cols), df.columns[0])
    dcol = next((cols[c] for c in ("direction", "dir", "sign", "regulation") if c in cols), None)
    if dcol is None and {"up", "down"} <= set(cols):            # side-by-side up/down columns
        return Query(name, "signature", df[cols["up"]].dropna().astype(str).tolist(),
                     df[cols["down"]].dropna().astype(str).tolist())
    if dcol is None:
        return Query(name, "signature", df[gcol].dropna().astype(str).tolist(), [])
    d = df[dcol].astype(str).str.strip().str.lower().map(_DIR)
    if d.isna().any():
        bad = sorted(set(df[dcol][d.isna()].astype(str)))[:5]
        raise ValueError(f"direction column '{dcol}' has values other than up/down: {bad}")
    g = df[gcol].astype(str)
    return Query(name, "signature", g[d == "up"].tolist(), g[d == "down"].tolist())


def _looks_like_signature(df):
    cols = {c.lower() for c in df.columns}
    if cols & {"direction", "dir", "regulation"} or {"up", "down"} <= cols:
        return True
    num = df.select_dtypes("number").shape[1]
    return num == 0 or (df.shape[1] <= 2 and df.shape[0] < 2000)


def _expression_from_matrix(M, sample, name, background):
    """M: genes x samples or samples x genes; returns Query for `sample` with the rest as background"""
    if sample is None and M.shape[1] == 1:
        sample = M.columns[0]
    if sample is None:
        raise ValueError("a genes x samples table needs sample=<column name>")
    if sample not in M.columns and sample in M.index:
        M = M.T
    if sample not in M.columns:
        raise KeyError(f"sample '{sample}' not found")
    M = M.apply(pd.to_numeric, errors="coerce")
    M = M[~M.index.duplicated()]
    bg = None
    if background is True or (isinstance(background, str) and background == "rest"):
        bg = M.T if M.shape[1] > 1 else None                     # cohort incl. the sample, like the benchmark
    elif isinstance(background, pd.DataFrame):
        bg = background
    return Query(name or str(sample), "sample", expr=M[sample], background=bg)


def _pseudobulk(adata, mask, layer, counts):
    X = adata[mask].layers[layer] if layer else adata[mask].X
    v = np.asarray(X.sum(0) if counts else X.mean(0)).ravel()
    if counts:
        v = np.log2(v / max(v.sum(), 1) * 1e6 + 1)
    return v


def read_query(obj, name=None, sample=None, groupby=None, group=None, background=True, layer=None, counts=None,
               sep=None, index_col=0):
    """Build a Query from a signature, a file, an expression profile or an AnnData.

    Parameters
    ----------
    obj        : list/tuple/set of genes, dict {"up": [...], "down": [...]}, Signature, built-in signature name,
                 path (.csv/.tsv/.txt/.gmt), pandas Series / DataFrame, or AnnData
    name       : label used in titles and file names
    sample     : column (table) or observation (AnnData) to use as the sample
    groupby, group : AnnData only - pseudobulk the observations with obs[groupby] == group
    background : True (default) = centre the sample on its own cohort (other samples / groups in the input);
                 False/None = centre on the atlas reference mean; or a samples x genes DataFrame
    layer      : AnnData layer to use (default .X)
    counts     : AnnData pseudobulk - True if the data are raw counts (summed then log2 CPM); default: guessed
    """
    from .sig import Signature
    if isinstance(obj, Query):
        return obj
    if isinstance(obj, Signature):
        return Query(name or obj.name, "signature", list(obj.up), list(obj.down), meta=dict(source=obj.source))
    if isinstance(obj, dict):
        return Query(name or "signature", "signature", list(obj.get("up", [])), list(obj.get("down", [])))
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Index)):
        return Query(name or "signature", "signature", [str(g) for g in obj], [])
    if isinstance(obj, pd.Series):
        if obj.dtype == object:
            return Query(name or "signature", "signature", obj.dropna().astype(str).tolist(), [])
        bg = background if isinstance(background, pd.DataFrame) else None
        return Query(name or str(obj.name or "sample"), "sample", expr=obj, background=bg)
    if isinstance(obj, pd.DataFrame):
        if _looks_like_signature(obj.reset_index() if obj.index.name else obj):
            return _signature_from_frame(obj.reset_index() if obj.index.name else obj, name or "signature")
        return _expression_from_matrix(obj, sample, name, background)
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            return read_query(_sig.get(str(obj)), name=name)          # built-in signature name
        if p.suffix == ".gmt":
            sigs = _sig.read_gmt(p)
            key = sample or name or next(iter(sigs))
            return read_query(sigs[key], name=name or key)
        if p.suffix in (".h5ad",):
            import anndata as ad
            return read_query(ad.read_h5ad(p), name, sample, groupby, group, background, layer, counts)
        sep = sep or ("\t" if p.suffix in (".tsv", ".txt", ".tab") else ",")
        raw = pd.read_csv(p, sep=sep)
        if _looks_like_signature(raw):
            return _signature_from_frame(raw, name or p.stem)
        M = pd.read_csv(p, sep=sep, index_col=index_col)
        return _expression_from_matrix(M, sample, name, background)
    try:
        import anndata as ad
    except ImportError:
        ad = None
    if ad is not None and isinstance(obj, ad.AnnData):
        A = obj
        genes = pd.Index(A.var_names)
        Xs = A.layers[layer] if layer else A.X
        if counts is None:                                          # integers and large values -> counts
            x = Xs[: min(50, A.n_obs)]
            x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
            counts = bool(np.allclose(x, np.round(x)) and np.nanmax(x) > 50)
        if groupby is not None:
            if group is None:
                raise ValueError("give group=<value of obs[groupby]>")
            labs = A.obs[groupby].astype(str)
            groups = sorted(labs.unique())
            prof = pd.DataFrame({g: _pseudobulk(A, (labs == g).values, layer, counts) for g in groups}, index=genes)
            key = str(group)
            bg = prof.T if background is True else (background if isinstance(background, pd.DataFrame) else None)
            return Query(name or f"{groupby}={key}", "sample", expr=prof[key], background=bg,
                         meta=dict(n_cells=int((labs == key).sum())))
        if sample is None:
            raise ValueError("give sample=<obs name> or groupby=/group= for an AnnData query")
        M = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
        df = pd.DataFrame(M.T, index=genes, columns=A.obs_names)
        if counts:
            df = np.log2(df / df.sum(0).clip(lower=1) * 1e6 + 1)
        return _expression_from_matrix(df, sample, name, background)
    raise TypeError(f"cannot build a query from {type(obj).__name__}")
