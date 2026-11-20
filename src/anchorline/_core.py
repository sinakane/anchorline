"""The Atlas object: reference of untreated/treated samples (within-sample gene ranks) and the drug-vs-control
contrast matrix (log2FC). Low-level maths lives here; user-facing steps are in `tl`, plots in `pl`.

Method (unchanged from the benchmark's query/atlas_query.py):
  reference  every control and treated sample, stored as within-sample gene ranks (0..1) so RNA-seq,
             arrays and pseudobulk compare; ranks are centred on the per-platform gene mean for matching.
  hierarchy  Zeng et al. 2022 Nat Med (doi:10.1038/s41591-022-01819-x) cell-type sets scored as mean rank;
             pseudotime = softmax(z)-weighted position on LSPC-Quiescent(0) - Primed(1) - Cycle(1.5) - GMP(2)
             - ProMono(3) - Mono(4).
  signature  score per untreated sample = mean centred rank of up genes - mean of down genes.
  sample     Spearman correlation on the 2,000 most variable reference genes between the platform-centred
             reference and the query centred on its own cohort (background) or the reference mean.
  cell line  score = mean over its studies (not the max, which would favour lines profiled many times).
"""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ._settings import settings

STATES = {"LSPC-Quiescent": 0.0, "LSPC-Primed-Top100": 1.0, "LSPC-Cycle-Top100": 1.5, "GMP-like-Top100": 2.0,
          "ProMono-like-Top100": 3.0, "Mono-like-Top100": 4.0}
STATE_LABEL = {0.0: "quiescent\nLSPC", 1.0: "primed\nLSPC", 1.5: "cycling\nLSPC", 2.0: "GMP-like",
               3.0: "ProMono-like", 4.0: "Mono-like"}
NON_MYELOID = ("Jurkat", "SEM", "RS4;11", "KOPN-8")      # lymphoid lines, excluded from matching by default
MODELS = ("cell_lines", "patient_derived", "all")


def _resource(name):
    return resources.files("anchorline") / "resources" / name


def _gene_list(name):
    return [l.strip() for l in _resource(name).read_text().splitlines() if l.strip() and not l.startswith("#")]


def rank01(v):
    """within-sample percentile ranks of finite values (0..1), NaN elsewhere"""
    v = np.asarray(v, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() > 1:
        out[ok] = (pd.Series(v[ok]).rank(method="average").values - 1) / (ok.sum() - 1)
    return out


def robust_z(x):
    x = pd.Series(x, dtype=float)
    mad = (x - x.median()).abs().median() * 1.4826
    return (x - x.median()) / (mad if mad > 0 else (x.std() or 1.0))


def state_scores(R, genes, sets):
    gi = {g: i for i, g in enumerate(genes)}
    S = {}
    for k in STATES:
        idx = [gi[g] for g in sets[k] if g in gi]
        S[k] = np.nanmean(R[:, idx].astype(np.float32), axis=1) if idx else np.nan
    return pd.DataFrame(S)


def add_pseudotime(S, ref_mu=None, ref_sd=None, temp=1.0):
    Z = (S - (S.mean() if ref_mu is None else ref_mu)) / (S.std() if ref_sd is None else ref_sd)
    W = np.exp(Z.values / temp)
    W = W / W.sum(1, keepdims=True)
    S = S.copy()
    S["pseudotime"] = W @ np.array(list(STATES.values()))
    S["GMP_to_primitive"] = S[["LSPC-Quiescent", "LSPC-Primed-Top100"]].mean(1) - S["GMP-like-Top100"]
    S["GMP_to_myeloid"] = S[["ProMono-like-Top100", "Mono-like-Top100"]].mean(1) - S["GMP-like-Top100"]
    return S


class Atlas:
    """AML drug-perturbation atlas. Create with `al.load()`.

    Attributes
    ----------
    reference : AnnData, samples x genes within-sample ranks (+ hierarchy scores in .obs)
    contrasts : AnnData, drug-vs-control contrasts x genes log2FC
    controls  : DataFrame, the untreated reference samples with pseudotime and state axes
    """

    def __init__(self, data_dir=None, verbose=True, download=True):
        D = Path(data_dir or settings.data_dir).expanduser()
        if download:
            from . import datasets
            if not datasets.is_available("atlas", D):
                print(f"atlas not found in {D} - downloading it once (~560 MB); "
                      "use al.settings.data_dir or $ANCHORLINE_DATA to keep it elsewhere")
                datasets.download("atlas", D)
        files = {k: D / f for k, f in dict(ref="query_reference.h5ad", con="AML_drug_atlas_bulk.h5ad",
                                           drugs="drugs.tsv", lines="cell_lines.tsv").items()}
        miss = [str(p) for p in files.values() if not p.exists()]
        if miss:
            raise FileNotFoundError("atlas files not found - run al.datasets.download() or set al.settings.data_dir:\n  "
                                    + "\n  ".join(miss))
        if verbose:
            print(f"loading atlas from {D} (~1-2 min, ~1.5 GB RAM)")
        self.data_dir = D
        self.reference = ad.read_h5ad(files["ref"])
        self.genes = np.asarray(self.reference.var_names)
        self.gi = {g: i for i, g in enumerate(self.genes)}
        self.R = self.reference.X
        self.obs = self.reference.obs
        for c in self.obs.columns:
            if str(self.obs[c].dtype) in ("category", "object"):
                self.obs[c] = self.obs[c].astype(str)
        self.mu = pd.Series(self.reference.uns["state_mu"])
        self.sd = pd.Series(self.reference.uns["state_sd"])
        self.sets = {k: list(v) for k, v in self.reference.uns["state_sets"].items()}
        self.contrasts = ad.read_h5ad(files["con"])
        for c in self.contrasts.obs.columns:
            if str(self.contrasts.obs[c].dtype) in ("object", "category"):
                self.contrasts.obs[c] = self.contrasts.obs[c].astype(object).fillna("").astype(str)
        self.ctrl = np.where(self.obs.status.values == "control")[0]
        self.controls = self.obs.iloc[self.ctrl]
        vn = pd.Index(self.contrasts.var_names)
        self.cgi = {g: i for i, g in enumerate(vn)}
        self.pc_idx = np.where(vn.isin(_gene_list("protein_coding_genes.txt")))[0]
        self.surf_idx = np.where(vn.isin(_gene_list("surfaceome_genes.txt")))[0]
        self.surface_genes = set(vn[self.surf_idx])
        self.sid = {s_: i for i, s_ in enumerate(self.reference.obs_names)}
        self.pc_ref = np.array([self.gi.get(g, -1) for g in vn[self.pc_idx]])
        self.surf_ref = np.array([self.gi.get(g, -1) for g in vn[self.surf_idx]])
        pm = pd.DataFrame(self.reference.varm["platform_mean"], columns=self.reference.uns["platforms"])
        self.Rc = self.R[self.ctrl].astype(np.float32) - pm[self.obs.platform.values[self.ctrl]].values.T
        dr = pd.read_csv(files["drugs"], sep="\t", dtype=str).fillna("")
        self.cid = {d.strip().lower(): c for d, c in zip(dr.drug, dr.pubchem_cid) if c}
        cl = pd.read_csv(files["lines"], sep="\t", dtype=str).fillna("")
        self.is_cell_line = set(cl[cl.cellosaurus != ""].cell_line)
        if verbose:
            print(f"  {self.reference.n_obs} reference samples ({len(self.ctrl)} untreated), {len(self.genes)} genes; "
                  f"{self.contrasts.n_obs} drug-vs-control contrasts")

    def __repr__(self):
        o = self.contrasts.obs
        return (f"anchorline Atlas ({self.data_dir})\n  reference: {self.reference.n_obs} samples "
                f"({len(self.ctrl)} untreated) x {len(self.genes)} genes\n  contrasts: {self.contrasts.n_obs} "
                f"({o.accession.nunique()} studies, {o.cell_line.nunique()} models, "
                f"{o.drug_clean.nunique() if 'drug_clean' in o else o.drug.nunique()} drugs)")

    # ------------------------------------------------------------------ lookups
    @property
    def drug_classes(self):
        """table of drug classes in the atlas with contrast / drug counts"""
        o = self.contrasts.obs
        t = o.assign(cls=o.drug_class.replace("", "unclassified").str.split(";")).explode("cls")
        k = "drug_clean" if "drug_clean" in o else "drug"
        return (t.groupby("cls").agg(n_contrasts=("drug", "size"), n_drugs=(k, "nunique"),
                                     n_studies=("accession", "nunique")).sort_values("n_contrasts", ascending=False))

    @property
    def cell_lines(self):
        """untreated reference models with sample/study counts and whether they are cell lines"""
        c = self.controls
        t = c.groupby("cell_line").agg(n_untreated_samples=("status", "size"), n_studies=("accession", "nunique"),
                                       genotype=("genotype", "first"))
        t["is_cell_line"] = t.index.isin(self.is_cell_line)
        return t.drop(index="", errors="ignore").sort_values("n_untreated_samples", ascending=False)

    def clean_drug(self, d):
        """unify combination names (fallback when the harmonised name is missing)"""
        syn = {"aza": "azacitidine", "azacytidine": "azacitidine", "5-aza": "azacitidine", "dac": "decitabine",
               "ara-c": "cytarabine", "arac": "cytarabine", "bay": ""}
        junk = {"simultaneous", "sequential", "combo", "combination", ""}
        d = re.sub(r"\s*\([^)]*\)", "", str(d))
        toks = []
        for t in d.split("+"):
            t = t.split(",")[0]
            t = re.sub(r"\b(2hcl|hcl|dihydrochloride|hydrochloride|mesylate|sodium)\b", "", t, flags=re.I)
            t = syn.get(t.strip().lower(), t.strip().lower())
            if t and t not in junk and t not in toks:
                toks.append(t)
        toks = [t for t in toks if not any(t != u and t in u for u in toks)]
        seen, out = set(), []
        for t in sorted(toks, key=lambda x: (not x.isalpha(), len(x))):
            k = self.cid.get(t, t)
            if k not in seen:
                seen.add(k)
                out.append(t)
        return " + ".join(sorted(out))

    # ------------------------------------------------------------------ matching maths
    def score_signature(self, up, down=()):
        ui = [self.gi[g] for g in up if g in self.gi]
        di = [self.gi[g] for g in down if g in self.gi]
        if not ui:
            raise ValueError("none of the up genes are in the reference")
        sc = np.nanmean(self.Rc[:, ui], 1) - (np.nanmean(self.Rc[:, di], 1) if di else 0.0)
        return sc, [self.genes[i] for i in ui], [self.genes[i] for i in di]

    def score_sample(self, expr, background=None, n_genes=2000, n_sig=100):
        expr = expr[~expr.index.duplicated()]
        common = [g for g in expr.index if g in self.gi]
        if len(common) < 500:
            raise ValueError(f"only {len(common)} query genes are in the reference - are these HGNC symbols?")
        q = pd.Series(rank01(expr[common].values), index=common)
        cols = np.array([self.gi[g] for g in common])
        Rq = self.Rc[:, cols]
        if background is not None:
            bg = background.reindex(columns=common)
            bgR = np.vstack([rank01(r) for r in bg.values])
            qc = q.values - np.nanmean(bgR, 0)
        else:
            qc = q.values - np.nanmean(self.R[self.ctrl][:, cols].astype(np.float32), 0)
        var = np.nanvar(Rq, 0)
        hv = np.argsort(np.nan_to_num(var))[::-1][:n_genes]
        A = np.nan_to_num(Rq[:, hv])
        b = np.nan_to_num(qc[hv])
        rb = pd.Series(b).rank().values
        ra = np.apply_along_axis(lambda x: pd.Series(x).rank().values, 1, A)
        ra = (ra - ra.mean(1, keepdims=True)) / (ra.std(1, keepdims=True) + 1e-9)
        rb = (rb - rb.mean()) / (rb.std() + 1e-9)
        sc = ra @ rb / len(rb)
        qs = pd.Series(qc, index=common).dropna().sort_values()
        Rfull = np.full((1, len(self.genes)), np.nan)
        Rfull[0, cols] = q.values
        state = add_pseudotime(state_scores(Rfull, self.genes, self.sets), self.mu, self.sd).iloc[0]
        return sc, list(qs.index[::-1][:n_sig]), list(qs.index[:n_sig]), state, len(common)

    def sample_table(self, score, models="cell_lines", exclude=NON_MYELOID):
        """per untreated sample: model, study, genotype, score, hierarchy position (filtered by `models`)"""
        if models not in MODELS:
            raise ValueError(f"models must be one of {MODELS}")
        t = self.controls[["cell_line", "accession", "genotype", "platform", "pseudotime", "GMP_to_myeloid",
                           "GMP_to_primitive"]].copy()
        t["score"] = score
        t = t[t.cell_line != ""]
        if models == "cell_lines":
            t = t[t.cell_line.isin(self.is_cell_line)]
        elif models == "patient_derived":
            t = t[~t.cell_line.isin(self.is_cell_line)]
        if exclude:
            t = t[~t.cell_line.isin(list(exclude))]
        return t

    # ------------------------------------------------------------------ contrast helpers
    def sample_state(self, names):
        idx = [self.sid[n] for n in names if n in self.sid]
        if not idx:
            return np.nan, np.nan, np.nan
        s = self.obs.iloc[idx]
        return s.pseudotime.mean(), s.GMP_to_myeloid.mean(), s.GMP_to_primitive.mean()

    def expressed(self, T, ref_cols, thr=0.5):
        """contrasts x genes bool: mean within-sample rank >= thr in control or treated samples"""
        ok = ref_cols >= 0
        out = np.zeros((len(T), len(ref_cols)), bool)
        for k, (t, c) in enumerate(zip(T.treated_samples, T.control_samples)):
            best = None
            for grp in (t, c):
                ix = [self.sid[x] for x in str(grp).split(";") if x in self.sid]
                if ix:
                    mr = np.nanmean(np.asarray(self.R[ix][:, ref_cols[ok]], dtype=np.float32), 0)
                    best = mr if best is None else np.fmax(best, mr)
            if best is not None:
                out[k, ok] = best >= thr
        return out

    def masked_lfc(self, T, cols):
        """contrasts x genes log2FC for contrast columns `cols`: 0 where not expressed in either arm, NaN where
        not measured; also returns the expressed mask"""
        rows = self.contrasts.obs.index.get_indexer(T.index)
        X = np.asarray(self.contrasts.X[rows][:, cols], dtype=np.float32)
        vn = np.asarray(self.contrasts.var_names)[cols]
        E = self.expressed(T, np.array([self.gi.get(g, -1) for g in vn]))
        return np.where(E, X, np.where(np.isnan(X), np.nan, 0.0)), E


def wmean(x, w):
    """weighted mean over rows, ignoring NaN"""
    ww = w[:, None] * ~np.isnan(x)
    return np.nansum(np.nan_to_num(x) * ww, 0) / np.where(ww.sum(0) > 0, ww.sum(0), np.nan)


def load(data_dir=None, verbose=True, download=True):
    """Load the atlas (reference + contrasts). `data_dir` defaults to `al.settings.data_dir`
    (~/.anchorline/data); if the files are missing they are downloaded first (download=False to disable)."""
    return Atlas(data_dir, verbose=verbose, download=download)
