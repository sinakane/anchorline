"""Tools: compute once, then plot / filter as you like.

    mt = al.tl.match(atlas, query)                 # closest untreated cell lines -> MatchResult
    dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)  # drugs tested in them         -> DrugResult
    dr.drugs / dr.contrasts / dr.classes           # tables
    al.tl.gene_changes(dr, ["CLEC12A", "CD84"])    # chosen genes per drug x cell line
    al.tl.top_gene_changes(dr, gene_set="surface") # each drug's top changed genes
    al.tl.signature_changes(dr)                    # how the query's own genes move per drug

Ranking (per contrast in the selected cell lines, robust z-scores capped at +-3, then averaged):
  magnitude  RMS log2FC over protein-coding genes (z within data type)
  reversal   -(mean log2FC of up genes - mean log2FC of down genes): pushes the query signature down
  diff_shift pseudotime(treated) - pseudotime(control): > 0 toward myeloid
Per drug: weighted mean (cell-line match weight x replicate support); score = composite x n/(n+1);
tier T1 = >= 2 studies and >= 75% consistent, T2 = >= 2 contrasts, T3 = single contrast.
Reported, not scored: n_deg_pc (expressed protein-coding genes |log2FC| > 1), surface (SURFY) and protein-coding
net index ((up - down) / expressed) and intensity (mean |log2FC|), top induced surface genes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._core import NON_MYELOID, Atlas, robust_z, wmean
from .io import Query, read_query


# ============================================================================== results
@dataclass
class MatchResult:
    """Output of `al.tl.match`.

    table   : every model ranked by match score (rank, score, best_study, studies, genotype, n_control_samples, weight)
    groups  : model | study pairs ranked by score
    samples : every untreated reference sample used, with its score and hierarchy position
    query   : the Query; up/down = genes used (for a sample query: its most specific genes)
    query_state : hierarchy position of a sample query (None for signatures)
    """
    query: Query
    table: pd.DataFrame
    groups: pd.DataFrame
    samples: pd.DataFrame
    up: list
    down: list
    query_state: pd.Series = None
    models: str = "cell_lines"
    info: dict = field(default_factory=dict)

    @property
    def name(self):
        return self.query.name

    def top(self, n=5):
        """the n best-matching models"""
        return self.table.head(n)

    def select(self, cell_lines=5):
        """resolve `cell_lines` (int = top n, or list of names) to a list of names in this match"""
        if isinstance(cell_lines, (int, np.integer)):
            return list(self.table.index[:cell_lines])
        names = [cell_lines] if isinstance(cell_lines, str) else list(cell_lines)
        low = {c.lower(): c for c in self.table.index}
        out = [low[c.lower()] for c in names if c.lower() in low]
        miss = [c for c in names if c.lower() not in low]
        if miss:
            print(f"not in this match table (excluded by `models`/`exclude`, or no untreated samples): {miss}")
        return out

    def weights(self, cell_lines=5):
        """match weights used to rank drugs: (score - median) / (best selected - median), clipped to [0.05, 1]"""
        sel = self.select(cell_lines)
        s = self.table.score.reindex(sel)
        med = self.table.score.median()
        return ((s - med) / (s.max() - med + 1e-9)).clip(0.05, 1).round(3)

    def __repr__(self):
        t = self.table.head(5)
        top = ", ".join(f"{c} ({s:.2f})" for c, s in zip(t.index, t.score))
        return (f"MatchResult '{self.name}' ({self.query.kind}, models={self.models}): {len(self.table)} models ranked\n"
                f"  top: {top}\n  .table .groups .samples .top(n) .select(...)")


@dataclass
class DrugResult:
    """Output of `al.tl.rank_drugs`.

    drugs     : one row per drug (rank, drug, tier, score, composite, magnitude_z, reversal_z, diff_shift, ...)
    contrasts : one row per drug-vs-control comparison in the selected cell lines
    classes   : one row per drug class (median pseudotime shift, IQR, % toward myeloid, top drugs)
    cell_lines: weights of the selected cell lines
    """
    match: MatchResult
    drugs: pd.DataFrame
    contrasts: pd.DataFrame
    cell_lines: pd.Series
    atlas: Atlas = field(repr=False, default=None)
    _classes: tuple = field(default=None, repr=False)

    @property
    def name(self):
        return self.match.name

    @property
    def classes(self):
        if self._classes is None:
            self._classes = drug_classes(self, min_contrasts=1, _return_contrasts=True)
        D, _ = self._classes
        return D

    def top(self, n=15):
        return self.drugs.head(n)

    def __repr__(self):
        d = self.drugs.head(5)
        top = ", ".join(f"{x} ({t})" for x, t in zip(d.drug_name, d.tier))
        return (f"DrugResult '{self.name}': {len(self.drugs)} drugs from {len(self.contrasts)} contrasts in "
                f"{len(self.cell_lines)} cell lines ({', '.join(self.cell_lines.index)})\n  top: {top}\n"
                f"  .drugs .contrasts .classes .top(n)")


# ============================================================================== 1. matching
def match(atlas, query, models="cell_lines", exclude=NON_MYELOID, n_genes=2000, n_sig=100, name=None, **read_kw):
    """Find the untreated atlas samples / cell lines closest to a query.

    Parameters
    ----------
    atlas   : Atlas from `al.load()`
    query   : anything `al.io.read_query` accepts (gene list, dict, signature name, file, Series, AnnData, Query)
    models  : "cell_lines" (default), "patient_derived" (PDX/primary only) or "all"
    exclude : models left out (default: lymphoid lines Jurkat, SEM, RS4;11, KOPN-8); () keeps everything
    n_genes : sample queries - number of most variable reference genes for the correlation
    n_sig   : sample queries - genes kept as the sample's own up/down signature (used for the reversal term)
    read_kw : passed to `al.io.read_query` (sample=, groupby=, group=, background=, layer=, ...)
    """
    q = read_query(query, name=name, **read_kw)
    if name:
        q.name = name
    info = {}
    if q.kind == "signature":
        sc, up, down = atlas.score_signature(q.up, q.down)
        info.update(n_up_used=len(up), n_down_used=len(down), n_up_given=len(q.up), n_down_given=len(q.down))
        state = None
    else:
        sc, up, down, state, ng = atlas.score_sample(q.expr, q.background, n_genes, n_sig)
        info.update(n_genes_shared=ng)
    t = atlas.sample_table(sc, models, exclude)
    per = (t.groupby(["cell_line", "accession"]).agg(score=("score", "mean"), n_samples=("score", "size"),
                                                     genotype=("genotype", "first")).reset_index())
    groups = per.sort_values("score", ascending=False).reset_index(drop=True)
    cl = (per.groupby("cell_line").agg(score=("score", "mean"), best_study=("score", "max"),
                                       studies=("accession", "nunique")).sort_values("score", ascending=False))
    cl["genotype"] = t.drop_duplicates("cell_line").set_index("cell_line").genotype.reindex(cl.index)
    cl["n_control_samples"] = t.groupby("cell_line").size().reindex(cl.index)
    cl["is_cell_line"] = cl.index.isin(atlas.is_cell_line)
    med = cl.score.median()
    cl["weight"] = ((cl.score - med) / (cl.score.max() - med + 1e-9)).clip(0.05, 1).round(3)
    cl.insert(0, "rank", np.arange(1, len(cl) + 1))
    cl.index.name = "cell_line"
    return MatchResult(q, cl, groups, t, up, down, state, models, info)


# ============================================================================== 2. drug ranking
def _top_surface(g, sg, n=8, min_lfc=1.0, min_frac=0.5):
    Xs = np.vstack(g._surf_lfc.values).astype(float)
    mean = wmean(Xs, np.asarray(g.weight.values, dtype=float))
    frac = (np.nan_to_num(Xs) > min_lfc).mean(0)
    mean = np.where(frac >= min_frac, mean, np.nan)
    o = np.argsort(-np.nan_to_num(mean, nan=-9))[:n]
    return ", ".join(f"{sg[j]} {mean[j]:+.1f}" for j in o if mean[j] > min_lfc)


def rank_drugs(atlas, match_result, cell_lines=5, base_background_only=True, min_contrasts=1):
    """Rank every drug tested in the chosen cell lines.

    Parameters
    ----------
    match_result : MatchResult from `al.tl.match`
    cell_lines   : int (top n of the match, default 5) or list of cell-line names (any from `mt.table`)
    base_background_only : only contrasts in the unmodified line (no knockouts / engineered variants)
    min_contrasts : drop drugs with fewer contrasts
    """
    mt = match_result
    w = mt.weights(cell_lines)
    if w.empty:
        raise ValueError("no cell lines selected")
    o = atlas.contrasts.obs
    sel = o.cell_line.isin(w.index)
    if base_background_only:
        sel &= o.context.str.endswith("|")
    idx = np.where(sel.values)[0]
    if len(idx) == 0:
        raise ValueError(f"no drug contrasts in {list(w.index)}")
    X = atlas.contrasts.X[idx]
    extra = [c for c in ("drug_clean", "drug_display", "epigenetic") if c in o]
    T = o.iloc[idx][["drug", "drug_class"] + extra + ["cell_line", "accession", "dose", "time", "n_treated", "n_control",
                     "value_type", "control_match", "treated_samples", "control_samples"]].copy()
    Xp = X[:, atlas.pc_idx]
    T["magnitude"] = np.sqrt(np.nanmean(Xp ** 2, 1))
    T["magnitude_all_genes"] = np.sqrt(np.nanmean(X ** 2, 1))
    E = atlas.expressed(T, atlas.pc_ref)
    Xe = np.where(E, np.nan_to_num(Xp), 0)
    T["n_expressed_pc"] = E.sum(1)
    T["n_deg_pc"] = (np.abs(Xe) > 1).sum(1)
    T["n_deg_pc_up"] = (Xe > 1).sum(1)
    T["n_deg_pc_down"] = (Xe < -1).sum(1)
    ne = np.maximum(T.n_expressed_pc.values, 1)
    T["pc_net_index"] = (T.n_deg_pc_up - T.n_deg_pc_down).values / ne
    T["pc_intensity"] = np.abs(Xe).sum(1) / ne
    sg = np.asarray(atlas.contrasts.var_names)[atlas.surf_idx]
    Es = atlas.expressed(T, atlas.surf_ref)
    Xr = X[:, atlas.surf_idx]
    Xs = np.where(Es, Xr, np.where(np.isnan(Xr), np.nan, 0.0))
    T["_surf_lfc"] = list(Xs.astype(np.float32))
    T["surface_mean_lfc"] = np.nanmean(Xs, 1)
    T["n_surface_up"] = (np.nan_to_num(Xs) > 1).sum(1)
    T["n_surface_down"] = (np.nan_to_num(Xs) < -1).sum(1)
    nes = np.maximum(Es.sum(1), 1)
    T["n_expressed_surface"] = Es.sum(1)
    T["surface_net_index"] = (T.n_surface_up - T.n_surface_down).values / nes
    T["surface_intensity"] = np.abs(np.nan_to_num(Xs)).sum(1) / nes
    T["surface_up_genes"] = [", ".join(f"{sg[j]} {r[j]:+.1f}" for j in np.argsort(-np.nan_to_num(r, nan=-9))[:8]
                                       if r[j] > 1) for r in Xs]
    dtype = np.where(o.iloc[idx].matrix.str.contains("sciplex"), "sciplex", T.value_type)
    T["data_type"] = dtype
    T["magnitude_z"] = pd.Series(T.magnitude.values).groupby(dtype).transform(robust_z).values
    T["n_deg_pc_z"] = pd.Series(T.n_deg_pc.values, dtype=float).groupby(dtype).transform(robust_z).values
    ui = [atlas.cgi[g] for g in mt.up if g in atlas.cgi]
    di = [atlas.cgi[g] for g in mt.down if g in atlas.cgi]
    if ui:
        eff = np.nanmean(X[:, ui], 1) - (np.nanmean(X[:, di], 1) if di else 0)
        T["signature_change"] = eff
        T["reversal_z"] = robust_z(-eff).values
    pt = [atlas.sample_state(t.split(";")) for t in T.treated_samples]
    pc = [atlas.sample_state(c.split(";")) for c in T.control_samples]
    T["pt_control"], T["pt_treated"] = [p[0] for p in pc], [p[0] for p in pt]
    T["myeloid_control"], T["myeloid_treated"] = [p[1] for p in pc], [p[1] for p in pt]
    T["primitive_control"], T["primitive_treated"] = [p[2] for p in pc], [p[2] for p in pt]
    T["diff_shift"] = T.pt_treated - T.pt_control
    T["diff_shift_z"] = robust_z(T.diff_shift).values
    zc = [c for c in ("magnitude_z", "reversal_z", "diff_shift_z") if c in T]
    T[zc] = T[zc].clip(-3, 3)
    T["composite"] = T[zc].mean(1, skipna=True)
    rep = np.minimum(T.n_treated.astype(int), T.n_control.astype(int))
    T["weight"] = T.cell_line.map(w).values * (rep / (rep + 1))
    T["drug_key"] = (T.drug_clean.where(T.drug_clean != "", T.drug.map(atlas.clean_drug)) if "drug_clean" in T
                     else T.drug.map(atlas.clean_drug))
    T = T[T.drug_key != ""]

    def agg(g):
        ww = g.weight.values
        f = lambda c: np.average(g[c].dropna(), weights=ww[g[c].notna().values]) if g[c].notna().any() else np.nan
        out = dict(drug=(g.drug_display.iloc[0] if "drug_display" in g and g.drug_display.iloc[0] else g.drug_key.iloc[0]),
                   drug_name=g.drug_key.iloc[0],
                   epigenetic=(str(g.epigenetic.iloc[0]) == "True") if "epigenetic" in g else False,
                   drug_class=g.drug_class.iloc[0], composite=f("composite"),
                   magnitude_z=f("magnitude_z"), diff_shift=f("diff_shift"),
                   magnitude=f("magnitude"), n_deg_pc=f("n_deg_pc"), n_deg_pc_up=f("n_deg_pc_up"),
                   n_deg_pc_down=f("n_deg_pc_down"), pc_net_index=f("pc_net_index"), pc_intensity=f("pc_intensity"),
                   consistency=(g.composite > 0).mean(), n_contrasts=len(g), n_studies=g.accession.nunique(),
                   n_cell_lines=g.cell_line.nunique(), cell_lines=", ".join(sorted(g.cell_line.unique())),
                   pt_control=f("pt_control"), pt_treated=f("pt_treated"),
                   myeloid_control=f("myeloid_control"), myeloid_treated=f("myeloid_treated"),
                   primitive_control=f("primitive_control"), primitive_treated=f("primitive_treated"),
                   surface_mean_lfc=f("surface_mean_lfc"), n_surface_up=f("n_surface_up"),
                   n_surface_down=f("n_surface_down"), surface_net_index=f("surface_net_index"),
                   surface_intensity=f("surface_intensity"), surface_up_genes=_top_surface(g, sg))
        if "reversal_z" in g:
            out.update(reversal_z=f("reversal_z"), signature_change=f("signature_change"))
        return pd.Series(out)

    D = T.groupby("drug_key").apply(agg)
    D = D[D.n_contrasts >= min_contrasts]
    D["score"] = D.composite * (1 - 1 / (1 + D.n_contrasts))
    D["tier"] = np.select([(D.n_studies >= 2) & (D.consistency >= 0.75), D.n_contrasts >= 2], ["T1", "T2"], "T3")
    D = D.sort_values("score", ascending=False)
    D.insert(0, "rank", np.arange(1, len(D) + 1))
    first = ["rank", "drug_name", "tier", "score", "composite", "magnitude_z"] + \
            (["reversal_z"] if "reversal_z" in D else []) + ["diff_shift", "consistency", "n_contrasts", "n_studies"]
    D = D[first + [c for c in D.columns if c not in first]]
    return DrugResult(mt, D, T.sort_values("composite", ascending=False), w, atlas)


# ============================================================================== 3. drug classes
def drug_classes(dr, min_contrasts=2, _return_contrasts=False):
    """Per drug class in the selected cell lines: median pseudotime shift (IQR), % of contrasts toward myeloid,
    median change on the GMP->myeloid and GMP->primitive axes, top drugs. Combinations count in each class."""
    T = dr.contrasts.drop(columns=["_surf_lfc"]).copy()
    T["cls"] = T.drug_class.replace("", "unclassified").str.split(";")
    T = T.explode("cls")
    T["d_myeloid"] = T.myeloid_treated - T.myeloid_control
    T["d_primitive"] = T.primitive_treated - T.primitive_control
    rows = []
    for c, g in T.groupby("cls"):
        if len(g) < min_contrasts:
            continue
        q = lambda col, p: float(np.nanpercentile(g[col], p)) if g[col].notna().any() else np.nan
        top = g.groupby("drug_key").composite.mean().sort_values(ascending=False).head(3).index
        rows.append(dict(drug_class=c, n_contrasts=len(g), n_drugs=g.drug_key.nunique(), n_studies=g.accession.nunique(),
                         composite=g.composite.mean(), pt_control=q("pt_control", 50), pt_treated=q("pt_treated", 50),
                         diff_shift=q("diff_shift", 50), diff_shift_q25=q("diff_shift", 25),
                         diff_shift_q75=q("diff_shift", 75), frac_toward_myeloid=(g.diff_shift > 0).mean(),
                         myeloid_control=q("myeloid_control", 50), myeloid_treated=q("myeloid_treated", 50),
                         primitive_control=q("primitive_control", 50), primitive_treated=q("primitive_treated", 50),
                         d_myeloid=q("d_myeloid", 50), d_primitive=q("d_primitive", 50), top_drugs=", ".join(top)))
    D = pd.DataFrame(rows).sort_values("diff_shift", ascending=False).reset_index(drop=True)
    return (D, T) if _return_contrasts else D


# ============================================================================== 4. genes
def _rows(dr, drugs=None, classes=None, n_top=12):
    """row label -> boolean mask over dr.contrasts: named drugs, named classes, or the top-ranked drugs"""
    T, D = dr.contrasts, dr.drugs
    if classes:
        classes = [classes] if isinstance(classes, str) else classes
        cl = T.drug_class.str.split(";")
        out = {c: cl.apply(lambda x: c in x).values for c in classes}
        miss = [c for c, m in out.items() if not m.any()]
        if miss:
            print(f"classes not tested in the selected cell lines: {miss}")
        out = {c: m for c, m in out.items() if m.any()}
    else:
        if drugs:
            drugs = [drugs] if isinstance(drugs, str) else drugs
            low = {str(k).lower(): k for k in D.index}
            low.update({str(v).lower(): k for k, v in D.drug_name.items()})
            keys = [low[d.lower()] for d in drugs if d.lower() in low]
            miss = [d for d in drugs if d.lower() not in low]
            if miss:
                print(f"drugs not tested in the selected cell lines: {miss}")
        else:
            keys = list(D.index[:n_top])
        out = {f"{D.drug_name[k]} ({D.tier[k]})": (T.drug_key == k).values for k in keys}
    if not out:
        raise ValueError("none of the requested drugs/classes was tested in the selected cell lines")
    return out


def gene_changes(dr, genes, drugs=None, classes=None, n_top=12, cell_lines=None):
    """log2FC of chosen genes per row (drug or class) x cell line, plus a 'pooled' column over all selected lines.

    Returns a long table: gene, row, cell_line, mean_log2fc (weighted), n_contrasts, expressed, frac_up, frac_down.
    Genes not expressed in treated or control samples count as 0 (no change)."""
    at = dr.atlas
    genes = [genes] if isinstance(genes, str) else list(genes)
    miss = [g for g in genes if g not in at.cgi]
    if miss:
        print(f"not in the atlas: {miss}")
    genes = [g for g in genes if g in at.cgi]
    if not genes:
        raise ValueError("none of the genes is in the atlas")
    T = dr.contrasts
    rows = _rows(dr, drugs, classes, n_top)
    lines = list(dr.cell_lines.index) if cell_lines is None else list(cell_lines)
    Xg, Eg = at.masked_lfc(T, [at.cgi[g] for g in genes])
    w = T.weight.values.astype(float)
    recs = []
    for k, g in enumerate(genes):
        for rl, mask in rows.items():
            for c in lines + ["pooled"]:
                sel = mask & ((T.cell_line == c).values if c != "pooled" else np.isin(T.cell_line, lines))
                if not sel.any() or np.isnan(Xg[sel, k]).all():
                    continue
                x = Xg[sel, k]
                recs.append(dict(gene=g, row=rl, cell_line=c, mean_log2fc=float(wmean(x[:, None], w[sel])[0]),
                                 n_contrasts=int(sel.sum()), expressed=bool(Eg[sel, k].any()),
                                 frac_up=float((np.nan_to_num(x) > 1).mean()),
                                 frac_down=float((np.nan_to_num(x) < -1).mean())))
    L = pd.DataFrame(recs)
    L.attrs.update(rows=list(rows), cell_lines=lines, genes=genes)
    return L


def top_gene_changes(dr, gene_set="surface", drugs=None, classes=None, n_top=12, min_lfc=1.0, min_frac=0.5):
    """Weighted mean log2FC of every gene in `gene_set` ("surface", "protein_coding" or a list of genes) per row,
    and each row's consistently changed genes (|mean| > min_lfc and changed in >= min_frac of its contrasts).

    Returns (matrix rows x genes, {row: {"up": [(gene, lfc), ...], "down": [...]}})."""
    at = dr.atlas
    T = dr.contrasts
    rows = _rows(dr, drugs, classes, n_top)
    if isinstance(gene_set, str):
        cols = {"surface": at.surf_idx, "protein_coding": at.pc_idx}[gene_set]
    else:
        cols = np.array([at.cgi[g] for g in gene_set if g in at.cgi])
    X, _ = at.masked_lfc(T, cols)
    names = np.asarray(at.contrasts.var_names)[cols]
    w = T.weight.values.astype(float)
    M, top = {}, {}
    for rl, mask in rows.items():
        x = X[mask]
        mu = wmean(x, w[mask])
        fu, fd = (np.nan_to_num(x) > min_lfc).mean(0), (np.nan_to_num(x) < -min_lfc).mean(0)
        up = np.where((mu > min_lfc) & (fu >= min_frac))[0]
        dn = np.where((mu < -min_lfc) & (fd >= min_frac))[0]
        up, dn = up[np.argsort(-mu[up])], dn[np.argsort(mu[dn])]
        M[rl] = mu
        top[rl] = dict(up=[(names[j], float(mu[j])) for j in up], down=[(names[j], float(mu[j])) for j in dn])
    return pd.DataFrame(M, index=names).T, top


def change_indices(dr, drugs=None, classes=None, n_top=12, n_list=8):
    """Per row: surface and protein-coding net index ((genes up - genes down) / expressed genes, -1..1) and
    intensity (mean |log2FC| over expressed genes), plus the top induced / repressed genes of each set."""
    T = dr.contrasts
    rows = _rows(dr, drugs, classes, n_top)
    w = T.weight.values.astype(float)
    idx = pd.DataFrame({c: [float(np.average(T[c].values[m], weights=w[m])) for m in rows.values()]
                        for c in ("surface_net_index", "pc_net_index", "surface_intensity", "pc_intensity")},
                       index=list(rows))
    for gs, tag in (("surface", "surface"), ("protein_coding", "pc")):
        _, top = top_gene_changes(dr, gs, drugs, classes, n_top)
        idx[f"{tag}_top_up"] = [", ".join(f"{g} {v:+.1f}" for g, v in top[k]["up"][:n_list]) for k in idx.index]
        idx[f"{tag}_top_down"] = [", ".join(f"{g} {v:+.1f}" for g, v in top[k]["down"][:n_list]) for k in idx.index]
    return idx


def signature_changes(dr, drugs=None, classes=None, n_top=12, max_genes=40):
    """How the query's own genes change per row: weighted mean log2FC (rows x genes; unexpressed = 0).
    Columns are the up genes then the down genes (most-changed first, capped at max_genes in total);
    `attrs["direction"]` maps each gene to up/down. A reversal shows as blue up genes and red down genes."""
    at = dr.atlas
    mt = dr.match
    up = [g for g in mt.up if g in at.cgi]
    down = [g for g in mt.down if g in at.cgi]
    M, _ = top_gene_changes(dr, up + down, drugs, classes, n_top)
    keep_up = M[up].abs().mean().sort_values(ascending=False).index[: max_genes - min(len(down), max_genes // 2)] if up else []
    keep_dn = M[down].abs().mean().sort_values(ascending=False).index[: max_genes // 2] if down else []
    M = M[list(keep_up) + list(keep_dn)]
    M.attrs["direction"] = {g: ("up" if g in up else "down") for g in M.columns}
    return M


def surface_genes(dr, min_lfc=1.0, min_frac=0.5):
    """Long table drug x surface gene: weighted mean log2FC and fraction of contrasts up, for consistently
    induced surface genes."""
    T = dr.contrasts
    at = dr.atlas
    Xs = np.vstack(T._surf_lfc.values).astype(float)
    sg = np.asarray(at.contrasts.var_names)[at.surf_idx]
    out = []
    for d, ix in T.groupby("drug_key").indices.items():
        mean = wmean(Xs[ix], T.weight.values[ix])
        frac = (np.nan_to_num(Xs[ix]) > min_lfc).mean(0)
        for j in np.where((mean > min_lfc) & (frac >= min_frac))[0]:
            out.append(dict(drug=d, gene=sg[j], mean_log2fc=round(mean[j], 3), frac_contrasts_up=round(frac[j], 2),
                            n_contrasts=len(ix)))
    return pd.DataFrame(out).sort_values(["drug", "mean_log2fc"], ascending=[True, False]) if out else pd.DataFrame()


def run(atlas, query, cell_lines=5, models="cell_lines", name=None, **kw):
    """Convenience: match + rank_drugs in one call -> (MatchResult, DrugResult)."""
    mt = match(atlas, query, models=models, name=name, **kw)
    return mt, rank_drugs(atlas, mt, cell_lines=cell_lines)
