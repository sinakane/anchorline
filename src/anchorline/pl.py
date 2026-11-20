"""Plots. Every function takes a result from `al.tl` and decides only what to show.

Common parameters (all functions):
    n_top      : how many items (cell lines / drugs / classes) to show, best first
    names / cell_lines / drugs / classes : show exactly these instead of the top n
    figsize    : (width, height) in inches; None = sized to the content
    palette    : list of colours for categorical identity (default al.settings.palette)
    cmap       : colormap for continuous values (scores: viridis; log2FC: blue-grey-red)
    fontsize   : base font size
    title      : figure / panel title (None = automatic)
    ax         : draw into an existing matplotlib Axes (single-panel plots)
    save       : True (-> al.settings.figdir/<name>.<format>), a path, or None
    show       : True (default) displays and returns None; False returns the Figure (or Axes when `ax` given)

    al.pl.match_ranking(mt)         bar plot of the closest cell lines (or cell line | study)
    al.pl.pseudotime(mt)            where the matched lines sit on the stem -> myeloid axis (+ viridis match score)
    al.pl.drug_ranking(dr)          bar plot of drug scores
    al.pl.drug_shifts(dr)           per drug: untreated -> treated pseudotime arrows
    al.pl.state_map(dr)             2-D GMP->myeloid / GMP->primitive map with drug arrows
    al.pl.class_trends(dr)          forest plot of the pseudotime shift per drug class
    al.pl.class_shift_map(dr)       2-D shift map per drug class
    al.pl.gene_changes(dr, genes)   chosen genes: drug (or class) x cell line dot grid
    al.pl.top_gene_changes(dr)      surface / protein-coding change indices + top changed genes
    al.pl.signature_changes(dr)     how the query's own up / down genes move per drug
    al.pl.summary(mt, dr)           the three-panel overview figure
    al.pl.group_summary({name: (mt, dr)})   compare several queries
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from . import tl
from ._core import STATE_LABEL
from ._settings import GREY, GRID, INK, INK2, SURFACE, settings


# ============================================================================== helpers
def _style(fontsize=None):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": fontsize or settings.fontsize, "axes.spines.top": False,
                         "axes.spines.right": False})


def _div(cmap=None):
    if cmap is None:
        return LinearSegmentedColormap.from_list("al_div", settings.diverging)
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def _pal(palette):
    return list(palette) if palette is not None else list(settings.palette)


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _finish(fig, save, show, default_name, ax=None):
    if save:
        if save is True:                                   # automatic name: never treat a dot as an extension
            base, exts = settings.figdir / _slug(default_name), list(settings.file_format)
        else:
            p = Path(save)
            known = p.suffix.lower().lstrip(".") in ("png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff", "eps")
            base, exts = (p.with_suffix(""), [p.suffix.lstrip(".")]) if known else (p, list(settings.file_format))
        base.parent.mkdir(parents=True, exist_ok=True)
        for e in exts:
            fig.savefig(f"{base}.{e}", dpi=settings.dpi, bbox_inches="tight")
    if show is False:
        return ax if ax is not None else fig
    plt.show()
    return None


def _subplots(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _states(ax, labels=True, y=1.02):
    for p, lab in STATE_LABEL.items():
        ax.axvline(p, color=GRID, lw=1, zorder=0)
        if labels:
            ax.text(p, y, lab, ha="center", va="bottom", fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform())


def _lines(mt, n_top, cell_lines):
    return mt.select(cell_lines if cell_lines is not None else n_top)


def _drug_rows(dr, n_top, drugs):
    D = dr.drugs
    if drugs is None:
        return D.head(n_top)
    drugs = [drugs] if isinstance(drugs, str) else drugs
    low = {str(k).lower(): k for k in D.index}
    low.update({str(v).lower(): k for k, v in D.drug_name.items()})
    keys = [low[d.lower()] for d in drugs if d.lower() in low]
    miss = [d for d in drugs if d.lower() not in low]
    if miss:
        print(f"drugs not tested in the selected cell lines: {miss}")
    return D.loc[keys]


def _label(d, maxlen=60):
    s = d.drug_name if "drug_name" in d else d.name
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


# ============================================================================== matching
def match_ranking(mt, n_top=15, cell_lines=None, by="cell_line", highlight=5, palette=None, color=None,
                  figsize=None, fontsize=None, title=None, ax=None, save=None, show=True):
    """Bar plot of match scores.

    by        : "cell_line" (mean over studies, the score used downstream) or "study" (cell line | study pairs)
    highlight : the top n (or list of names) drawn in identity colours, the rest grey
    color     : single colour for all highlighted bars instead of the palette
    """
    _style(fontsize)
    pal = _pal(palette)
    top = mt.select(highlight) if highlight is not None else []
    cmap = {c: (color or pal[i % len(pal)]) for i, c in enumerate(top)}
    if by == "study":
        G = mt.groups if cell_lines is None else mt.groups[mt.groups.cell_line.isin(mt.select(cell_lines))]
        G = G.head(n_top).iloc[::-1]
        labels = [f"{c[:30]} | {a}" for c, a in zip(G.cell_line, G.accession)]
        names, extra = list(G.cell_line), [f"[{g}] n={int(n)}" for g, n in zip(G.genotype, G.n_samples)]
        med = mt.groups.score.median()
    else:
        G = mt.table.loc[_lines(mt, n_top, cell_lines)].iloc[::-1]
        labels = [f"{c[:34]}" for c in G.index]
        names = list(G.index)
        extra = [f"[{g}] {int(s)} studies, n={int(n)}" for g, s, n in zip(G.genotype, G.studies, G.n_control_samples)]
        med = mt.table.score.median()
    fig, ax_ = _subplots(ax, figsize or (8.5, 0.3 * len(G) + 1.5))
    ax_.barh(range(len(G)), G.score, color=[cmap.get(c, GREY) for c in names], height=0.72)
    ax_.set_yticks(range(len(G)))
    ax_.set_yticklabels(labels)
    for i, (s, e) in enumerate(zip(G.score, extra)):
        ax_.text(max(s, 0), i, f"  {s:.2f}  {e}", va="center", fontsize=7, color=INK2)
    ax_.axvline(med, color=INK2, lw=0.8, ls=":")
    ax_.text(med, -0.75, " median of all models", fontsize=7, color=INK2, va="top")
    ax_.set_ylim(-1.1, len(G) - 0.4)
    ax_.set_xlim(min(0, G.score.min() * 1.1), G.score.max() * 1.6)
    ax_.set_xlabel("match score (signature: centred mean rank; sample: Spearman rho)")
    ax_.grid(axis="x", color=GRID, lw=0.5)
    ax_.set_axisbelow(True)
    ax_.tick_params(axis="y", length=0)
    ax_.set_title(title or f"{mt.name}: closest untreated {'cell line | study' if by == 'study' else 'models'}",
                  loc="left", fontsize=10)
    return _finish(fig, save, show, f"{mt.name}_match_ranking", ax)


def _pseudotime_strip(ax, mt, lines, pal, show_query=True, point_size=18, legend=True, seed=0):
    rng = np.random.default_rng(seed)
    S = mt.samples
    other = S[~S.cell_line.isin(lines)]
    ax.scatter(other.pseudotime, rng.uniform(-0.35, 0.35, len(other)), s=5, c=GREY, lw=0,
               label="other untreated samples")
    for i, c in enumerate(lines):
        s = S[S.cell_line == c]
        r = mt.table.loc[c]
        ax.scatter(s.pseudotime, rng.uniform(-0.35, 0.35, len(s)), s=point_size, c=pal[i % len(pal)], lw=0.4,
                   edgecolors=SURFACE, label=f"{int(r['rank'])}. {c} [{r.genotype}]  {r.score:.2f}", zorder=3)
    if show_query and mt.query_state is not None:
        ax.scatter([mt.query_state.pseudotime], [0], marker="*", s=380, c=INK, zorder=5, label="query")
    ax.set_yticks([])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.6, 0.6)
    _states(ax)
    if legend:
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))


def pseudotime(mt, n_top=5, cell_lines=None, palette=None, cmap=None, show_query=True, point_size=18,
               figsize=None, fontsize=None, title=None, save=None, show=True):
    """Where the closest untreated cell lines sit on the Zeng 2022 differentiation axis
    (quiescent LSPC -> primed -> cycling -> GMP -> ProMono -> Mono).

    Top panel   : every untreated reference sample (grey); the n_top matched lines in identity colours, most
                  similar first (legend: rank, genotype, match score); a sample query is the black star.
    Bottom panel: only the matched lines, one row each, every untreated sample coloured by its own match score
                  (cmap, default viridis); black tick = the line's mean position.
    """
    _style(fontsize)
    pal = _pal(palette)
    lines = _lines(mt, n_top, cell_lines)
    fig = plt.figure(figsize=figsize or (13, 3.2 + 0.42 * len(lines)))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.3 + 0.32 * len(lines)], hspace=0.45)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _pseudotime_strip(a1, mt, lines, pal, show_query, point_size)
    a1.set_title("Untreated reference samples; coloured = most similar cell lines (legend: rank, genotype, score)",
                 loc="left", fontsize=10, pad=28)
    S = mt.samples[mt.samples.cell_line.isin(lines)]
    vals = S.score
    norm = Normalize(vals.min(), vals.max())
    cm = plt.get_cmap(cmap or settings.score_cmap)
    rng = np.random.default_rng(1)
    for i, c in enumerate(lines):
        s = S[S.cell_line == c]
        a2.scatter(s.pseudotime, i + rng.uniform(-0.18, 0.18, len(s)), s=point_size + 6, c=cm(norm(s.score)), lw=0.3,
                   edgecolors=SURFACE, zorder=3)
        a2.plot([s.pseudotime.mean()] * 2, [i - 0.32, i + 0.32], color=INK, lw=1.6, zorder=4)
    a2.set_yticks(range(len(lines)))
    a2.set_yticklabels([f"{c} ({mt.table.score[c]:.2f})" for c in lines])
    for t_, col in zip(a2.get_yticklabels(), pal):
        t_.set_color(col)
    a2.set_ylim(len(lines) - 0.5, -0.5)
    a2.set_xlim(-0.2, 4.2)
    _states(a2, labels=False)
    if show_query and mt.query_state is not None:
        a2.axvline(mt.query_state.pseudotime, color=INK, ls="--", lw=0.9)
    a2.set_xlabel("differentiation pseudotime (Zeng 2022 hierarchy): stem -> GMP -> myeloid")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    cb = fig.colorbar(sm, ax=a2, fraction=0.03, pad=0.01)
    cb.set_label("match score\n(per sample)", fontsize=7.5)
    a2.set_title("Matched cell lines only: each untreated sample coloured by its match score; tick = mean position",
                 loc="left", fontsize=10)
    fig.suptitle(title or f"{mt.name}: position before treatment", x=0.01, ha="left", fontsize=12, y=1.0)
    return _finish(fig, save, show, f"{mt.name}_pseudotime")


# ============================================================================== drugs
def drug_ranking(dr, n_top=15, drugs=None, color_by="tier", palette=None, components=True, figsize=None,
                 fontsize=None, title=None, ax=None, save=None, show=True):
    """Horizontal bars of the drug score.

    color_by   : "tier" (T1/T2/T3), "epigenetic", "drug_class" or a single colour
    components : also draw the three score terms (magnitude, reversal, differentiation; robust z) as markers
    """
    _style(fontsize)
    pal = _pal(palette)
    D = _drug_rows(dr, n_top, drugs).iloc[::-1]
    if color_by == "tier":
        cmap = {"T1": pal[0], "T2": pal[3], "T3": GREY}
        cols = [cmap[t] for t in D.tier]
        leg = [(k, v) for k, v in cmap.items()]
    elif color_by == "epigenetic":
        cols = [pal[6] if e else GREY for e in D.epigenetic]
        leg = [("epigenetic", pal[6]), ("other", GREY)]
    elif color_by == "drug_class":
        cls = list(dict.fromkeys(D.drug_class[::-1]))
        cmap = {c: (pal[i] if i < len(pal) else GREY) for i, c in enumerate(cls)}
        cols = [cmap[c] for c in D.drug_class]
        leg = [(c[:28], v) for c, v in cmap.items()][: len(pal)]
    else:
        cols, leg = [color_by] * len(D), []
    fig, ax_ = _subplots(ax, figsize or (10, 0.36 * len(D) + 1.6))
    y = np.arange(len(D))
    ax_.barh(y, D.score, color=cols, height=0.62, zorder=2)
    lows = [D.score.min()]
    if components:
        for c, mk, lab in (("magnitude_z", "o", "magnitude"), ("reversal_z", "s", "reversal"),
                           ("diff_shift_z", "D", "differentiation")):
            if c == "diff_shift_z":
                v = dr.contrasts.groupby("drug_key").apply(
                    lambda g: np.average(g.diff_shift_z, weights=g.weight)).reindex(D.index)
            elif c in D:
                v = D[c]
            else:
                continue
            lows.append(np.nanmin(v.values) / 3)
            ax_.scatter(v / 3, y, marker=mk, s=16, facecolors="none", edgecolors=INK2, lw=0.8, zorder=3,
                        label=f"{lab} z / 3")
    ax_.axvline(0, color=INK2, lw=0.7)
    ax_.set_yticks(y)
    ax_.set_yticklabels([f"{int(d['rank'])}. {_label(d)}" + (" *" if d.epigenetic else "") for _, d in D.iterrows()])
    xmax = max(D.score.max(), 0.3)
    for i, (_, d) in enumerate(D.iterrows()):
        ax_.text(xmax * 1.08, i, f"{d.tier}  {int(d.n_contrasts)}/{int(d.n_studies)}  {d.consistency:.0%}",
                 va="center", fontsize=7, color=INK2)
    ax_.text(xmax * 1.08, len(D) - 0.4, "tier  contrasts/studies  consistent", fontsize=7, color=INK2, va="bottom")
    ax_.set_xlim(min(-0.08, float(np.nanmin(lows)) * 1.15 - 0.05), xmax * 1.75)
    ax_.set_xlabel("score = composite x n/(n+1)   (markers: the three terms, robust z / 3)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in leg]
    h2, l2 = ax_.get_legend_handles_labels()
    ax_.legend(handles + h2, [k for k, _ in leg] + l2, fontsize=7, frameon=False, loc="lower right")
    ax_.grid(axis="x", color=GRID, lw=0.5)
    ax_.set_axisbelow(True)
    ax_.tick_params(axis="y", length=0)
    ax_.set_title(title or f"{dr.name}: drug ranking in {', '.join(dr.cell_lines.index)}  (* = epigenetic)",
                  loc="left", fontsize=10)
    return _finish(fig, save, show, f"{dr.name}_drug_ranking", ax)


def drug_shifts(dr, n_top=12, drugs=None, palette=None, show_contrasts=True, figsize=None, fontsize=None,
                title=None, ax=None, save=None, show=True):
    """Per drug: arrow from the untreated to the treated pseudotime (weighted means over contrasts); orange =
    toward myeloid, blue = toward stem; small dots = single contrasts."""
    _style(fontsize)
    pal = _pal(palette)
    D = _drug_rows(dr, n_top, drugs).iloc[::-1]
    fig, ax_ = _subplots(ax, figsize or (10, 0.4 * len(D) + 1.6))
    rng = np.random.default_rng(0)
    for i, (k, d) in enumerate(D.iterrows()):
        col = pal[1] if d.pt_treated >= d.pt_control else pal[0]
        ax_.annotate("", xy=(d.pt_treated, i), xytext=(d.pt_control, i),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=12))
        ax_.scatter([d.pt_control], [i], s=30, c=INK2, zorder=3)
        if show_contrasts:
            sub = dr.contrasts[dr.contrasts.drug_key == k]
            ax_.scatter(sub.pt_treated, i + rng.uniform(-0.18, 0.18, len(sub)), s=9, c=col, alpha=0.45, lw=0)
        ax_.text(4.35, i, f"{d.tier}  {d.score:+.2f}  {d.diff_shift:+.2f}", va="center", fontsize=7, color=INK2)
    ax_.text(4.35, len(D) - 0.3, "tier  score  shift", fontsize=7, color=INK2, va="bottom")
    ax_.set_yticks(range(len(D)))
    ax_.set_yticklabels([f"{int(d['rank'])}. {_label(d)}" for _, d in D.iterrows()])
    _states(ax_)
    ax_.set_xlim(-0.2, 5.3)
    ax_.set_xticks([0, 1, 2, 3, 4])
    ax_.spines["bottom"].set_bounds(0, 4)
    ax_.set_xlabel("pseudotime: grey dot = untreated, arrow head = treated (dots = single contrasts)")
    ax_.set_title(title or f"{dr.name}: how the top drugs move the matched cell lines", loc="left", fontsize=10,
                  pad=26)
    return _finish(fig, save, show, f"{dr.name}_drug_shifts", ax)


def state_map(dr, n_top=8, drugs=None, palette=None, show_background=True, show_query=True, figsize=None,
              fontsize=None, title=None, ax=None, save=None, show=True):
    """2-D state map: x = GMP->myeloid, y = GMP->primitive (rank-score differences). Grey = all untreated
    samples, coloured = the selected cell lines, arrows = drugs (untreated -> treated)."""
    _style(fontsize)
    pal = _pal(palette)
    D = _drug_rows(dr, n_top, drugs)
    at = dr.atlas
    fig, ax_ = _subplots(ax, figsize or (8, 7))
    if show_background:
        c = at.controls
        ax_.scatter(c.GMP_to_myeloid, c.GMP_to_primitive, s=4, c=GREY, lw=0)
        S = dr.match.samples
        for i, cl in enumerate(dr.cell_lines.index):
            s = S[S.cell_line == cl]
            ax_.scatter(s.GMP_to_myeloid, s.GMP_to_primitive, s=12, c=pal[i % len(pal)], lw=0, alpha=0.8,
                        label=cl)
    texts = []
    for i, (k, d) in enumerate(D.iterrows()):
        col = pal[i % len(pal)]
        ax_.annotate("", xy=(d.myeloid_treated, d.primitive_treated), xytext=(d.myeloid_control, d.primitive_control),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=13), zorder=4)
        texts.append(ax_.text(d.myeloid_treated, d.primitive_treated,
                              f"{int(d['rank'])}. {d.drug_name[:22]}{' (epi)' if d.epigenetic else ''}",
                              fontsize=8.5, color=col, fontweight="bold", zorder=5))
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax_, expand=(1.2, 1.5), arrowprops=dict(arrowstyle="-", color="#9a9994", lw=0.6))
    except ImportError:
        pass
    if show_query and dr.match.query_state is not None:
        q = dr.match.query_state
        ax_.scatter([q.GMP_to_myeloid], [q.GMP_to_primitive], marker="*", s=300, c=INK, zorder=5, label="query")
    ax_.axhline(0, color=GRID)
    ax_.axvline(0, color=GRID)
    ax_.set_xlabel("GMP -> myeloid (ProMono/Mono minus GMP rank score)")
    ax_.set_ylabel("GMP -> primitive (LSPC minus GMP rank score)")
    if show_background:
        ax_.legend(fontsize=7, frameon=False, loc="lower left", markerscale=1.5)
    ax_.set_title(title or f"{dr.name}: state map (arrows = drugs, number = rank)", loc="left", fontsize=10)
    return _finish(fig, save, show, f"{dr.name}_state_map", ax)


# ============================================================================== classes
def _class_rows(dr, n_top, classes, min_contrasts, exclude_other=True):
    D, T = tl.drug_classes(dr, min_contrasts=min_contrasts, _return_contrasts=True)
    if classes is not None:
        classes = [classes] if isinstance(classes, str) else classes
        miss = [c for c in classes if c not in set(D.drug_class)]
        if miss:
            print(f"classes with < {min_contrasts} contrasts in the selected cell lines: {miss}")
        D = D[D.drug_class.isin(classes)]
    else:
        if exclude_other:
            D = D[~D.drug_class.str.startswith(("other", "inactive", "unannotated", "unclassified"))]
        D = D.sort_values("n_contrasts", ascending=False).head(n_top)
    return D, T


def class_trends(dr, n_top=16, classes=None, min_contrasts=2, palette=None, exclude_other=True, figsize=None,
                 fontsize=None, title=None, ax=None, save=None, show=True):
    """Forest plot of the pseudotime shift (treated - untreated) per drug class: dots = contrasts, bar = IQR,
    diamond = median; > 0 toward myeloid. Default rows: the n_top classes with most contrasts."""
    _style(fontsize)
    pal = _pal(palette)
    D, T = _class_rows(dr, n_top, classes, min_contrasts, exclude_other)
    D = D.sort_values("diff_shift")
    fig, ax_ = _subplots(ax, figsize or (11, max(4, 0.42 * len(D) + 2)))
    rng = np.random.default_rng(1)
    xs = T[T.cls.isin(D.drug_class)].diff_shift.dropna()
    lim = max(0.15, float(np.nanpercentile(np.abs(xs), 97))) * 1.1 if len(xs) else 0.5
    for i, (_, d) in enumerate(D.iterrows()):
        g = T[T.cls == d.drug_class].dropna(subset=["diff_shift"])
        ax_.scatter(g.diff_shift.clip(-lim, lim), i + rng.uniform(-0.22, 0.22, len(g)), s=10, lw=0, alpha=0.75,
                    c=np.where(g.diff_shift > 0, pal[1], pal[0]), zorder=2)
        ax_.plot([d.diff_shift_q25, d.diff_shift_q75], [i, i], color=INK2, lw=2.2, zorder=3)
        ax_.scatter([d.diff_shift], [i], marker="D", s=46, c=pal[1] if d.diff_shift > 0 else pal[0], edgecolors=INK2,
                    lw=0.8, zorder=4)
        ax_.text(lim * 1.08, i, f"{int(d.n_contrasts)}/{int(d.n_drugs)}/{int(d.n_studies)}   "
                                f"{d.frac_toward_myeloid:.0%}   {d.diff_shift:+.2f}", va="center", fontsize=7.2,
                 color=INK2)
    ax_.axvline(0, color=INK2, lw=0.9)
    ax_.set_yticks(range(len(D)))
    ax_.set_yticklabels(D.drug_class)
    ax_.set_xlim(-lim, lim * 1.95)
    ax_.spines["bottom"].set_bounds(-lim, lim)
    ax_.set_xticks(np.round(np.linspace(-lim, lim, 5), 2))
    ax_.text(lim * 1.08, len(D) - 0.4, "contrasts/drugs/studies   % to myeloid   median", fontsize=7, color=INK2,
             va="bottom")
    ax_.text(-lim, -0.9, "<- toward stem / primitive", fontsize=7.5, color=pal[0], va="top")
    ax_.text(lim, -0.9, "toward myeloid ->", fontsize=7.5, color=pal[1], va="top", ha="right")
    ax_.set_ylim(-1.2, len(D) - 0.2)
    ax_.set_xlabel("pseudotime shift (treated - untreated): dots = contrasts, bar = IQR, diamond = median")
    ax_.grid(axis="x", color=GRID, lw=0.5)
    ax_.set_axisbelow(True)
    ax_.set_title(title or f"{dr.name}: drug-class trends in {', '.join(dr.cell_lines.index)}", loc="left",
                  fontsize=10)
    return _finish(fig, save, show, f"{dr.name}_class_trends", ax)


def class_shift_map(dr, n_top=12, classes=None, min_contrasts=2, palette=None, exclude_other=True,
                    show_contrasts=True, figsize=None, fontsize=None, title=None, ax=None, save=None, show=True):
    """Median change per drug class on GMP->myeloid (x) and GMP->primitive (y), drawn as arrows from the
    untreated state (origin); faint dots = single contrasts (clipped at the frame)."""
    _style(fontsize)
    pal = _pal(palette)
    D, T = _class_rows(dr, n_top, classes, min_contrasts, exclude_other)
    D = D.dropna(subset=["d_myeloid", "d_primitive"])
    ccol = {c: (pal[k] if k < len(pal) else INK2) for k, c in enumerate(D.drug_class)}
    fig, ax_ = _subplots(ax, figsize or (8, 7.5))
    arr = D[["d_myeloid", "d_primitive"]].abs().values
    Lm = max(0.004, float(np.nanmax(arr)) * 1.6) if arr.size else 0.05
    if show_contrasts:
        for c in D.drug_class:
            g = T[T.cls == c]
            ax_.scatter(g.d_myeloid.clip(-Lm, Lm), g.d_primitive.clip(-Lm, Lm), s=8, c=ccol[c], alpha=0.18, lw=0)
    texts = []
    for _, d in D.iterrows():
        ax_.annotate("", xy=(d.d_myeloid, d.d_primitive), xytext=(0, 0), zorder=4,
                     arrowprops=dict(arrowstyle="-|>", color=ccol[d.drug_class], lw=2.6, mutation_scale=15))
        texts.append(ax_.text(d.d_myeloid, d.d_primitive, f"{d.drug_class} ({int(d.n_contrasts)})", fontsize=9,
                              color=ccol[d.drug_class], fontweight="bold", zorder=5))
    ax_.set_xlim(-Lm, Lm)
    ax_.set_ylim(-Lm, Lm)
    ax_.set_aspect("equal")
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax_, expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color="#9a9994", lw=0.6))
    except ImportError:
        pass
    ax_.axhline(0, color=GRID, zorder=0)
    ax_.axvline(0, color=GRID, zorder=0)
    kw = dict(fontsize=7.5, color=INK2, transform=ax_.transAxes)
    ax_.text(0.98, 0.52, "toward myeloid", ha="right", va="bottom", **kw)
    ax_.text(0.02, 0.52, "away from myeloid", ha="left", va="bottom", **kw)
    ax_.text(0.52, 0.98, "toward primitive", ha="left", va="top", **kw)
    ax_.text(0.52, 0.02, "away from primitive", ha="left", va="bottom", **kw)
    ax_.set_xlabel("change in GMP -> myeloid score (treated - untreated)")
    ax_.set_ylabel("change in GMP -> primitive score")
    ax_.set_title(title or f"{dr.name}: shift map per drug class (n contrasts)", loc="left", fontsize=10)
    return _finish(fig, save, show, f"{dr.name}_class_shift_map", ax)


# ============================================================================== genes
def gene_changes(dr, genes=("CLEC12A", "CD84"), n_top=12, drugs=None, classes=None, cell_lines=None, cmap=None,
                 vmax=None, size_scale=45, figsize=None, fontsize=None, title=None, save=None, show=True):
    """How chosen genes change per drug (or class) x selected cell line, plus a 'pooled' column.
    Colour = weighted mean log2FC (cmap, default blue-grey-red; vmax sets the colour limit), size = number of
    contrasts, hollow = gene not expressed, blank = not tested. Rows: drugs=[...], classes=[...] or the top n."""
    _style(fontsize)
    L = tl.gene_changes(dr, genes, drugs, classes, n_top, cell_lines)
    rl, lines, genes = L.attrs["rows"], L.attrs["cell_lines"], L.attrs["genes"]
    cols_ = lines + ["pooled"]
    cm = _div(cmap)
    lim = vmax or (max(1.0, float(np.nanpercentile(np.abs(L.mean_log2fc), 95))) if len(L) else 1.0)
    norm = Normalize(-lim, lim)
    fig, axs = plt.subplots(1, len(genes), squeeze=False, sharey=True,
                            figsize=figsize or (1.6 + 0.75 * len(cols_) * len(genes) + 2.2, 1.6 + 0.36 * len(rl)))
    at = dr.atlas
    for ax, g in zip(axs[0], genes):
        ax.set_xlim(-0.6, len(cols_) - 0.4)
        ax.set_ylim(len(rl) - 0.5, -0.6)
        ax.axvline(len(cols_) - 1.5, color=GREY, lw=0.8)
        for i in range(len(rl)):
            ax.axhline(i, color="#ecebe7", lw=0.6, zorder=0)
        for _, d in L[L.gene == g].iterrows():
            x, y = cols_.index(d.cell_line), rl.index(d.row)
            sz = 40 + size_scale * min(d.n_contrasts, 6)
            if d.expressed:
                ax.scatter(x, y, s=sz, c=[cm(norm(d.mean_log2fc))], edgecolors=SURFACE, lw=0.5, zorder=3)
                if abs(d.mean_log2fc) >= 0.5:
                    ax.text(x + 0.2, y - 0.22, f"{d.mean_log2fc:+.1f}", ha="left", va="center", fontsize=6.5,
                            color=INK, zorder=4)
            else:
                ax.scatter(x, y, s=sz, c="none", edgecolors=INK2, lw=0.9, zorder=3)
        ax.set_xticks(range(len(cols_)))
        ax.set_xticklabels(cols_, rotation=60, ha="right")
        ax.set_title(g + (" (surface)" if g in at.surface_genes else ""), loc="left", fontsize=10)
        ax.tick_params(length=0)
    axs[0][0].set_yticks(range(len(rl)))
    axs[0][0].set_yticklabels(rl)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=axs[0].tolist(), fraction=0.025, pad=0.02)
    cb.set_label("log2FC treated vs control (weighted mean)")
    fig.suptitle((title or f"{dr.name}: {', '.join(genes)} per drug and cell line") + "\n", x=0.01, y=1.02, ha="left",
                 va="bottom", fontsize=11.5)
    fig.text(0.01, 1.01, "dot size = n contrasts; hollow = not expressed in treated or control samples; blank = not "
             "tested in that line; 'pooled' = all selected lines", fontsize=7.5, color=INK2, va="bottom")
    return _finish(fig, save, show, f"{dr.name}_genes_{'_'.join(genes)}")


def _heat(ax, H, cm, lim, label_min=1.0, direction=None, pal=None):
    im = ax.imshow(H.values, cmap=cm, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(H.shape[1]))
    ax.set_xticklabels(H.columns, rotation=90, fontsize=7.5)
    if direction is not None:
        for t_ in ax.get_xticklabels():
            t_.set_color(pal[1] if direction.get(t_.get_text()) == "up" else pal[0])
    for i in range(H.shape[0]):
        for k in range(H.shape[1]):
            v = H.values[i, k]
            if np.isfinite(v) and abs(v) >= label_min:
                ax.text(k, i, f"{v:.0f}", ha="center", va="center", fontsize=5.8,
                        color="white" if abs(v) > 0.7 * lim else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def top_gene_changes(dr, n_top=12, drugs=None, classes=None, n_up=3, n_down=2, max_genes=30, cmap=None,
                     palette=None, figsize=None, fontsize=None, title=None, save=None, show=True):
    """Per drug (or class): A) surface (SURFY) and protein-coding net index ((up - down) / expressed genes) and
    intensity (mean |log2FC|); B/C) heatmaps of each row's top n_up induced + n_down repressed surface /
    protein-coding genes (changed in >= 50% of its contrasts; union capped at max_genes)."""
    _style(fontsize)
    pal = _pal(palette)
    cm = _div(cmap)
    idx = tl.change_indices(dr, drugs, classes, n_top)
    rl = list(idx.index)
    heat = {}
    for gs_ in ("surface", "protein_coding"):
        M, top = tl.top_gene_changes(dr, gs_, drugs, classes, n_top)
        pick = list(dict.fromkeys(sum(([g for g, _ in top[k]["up"][:n_up]] + [g for g, _ in top[k]["down"][:n_down]]
                                       for k in rl), [])))
        if len(pick) > max_genes:
            pick = list(M[pick].abs().max().sort_values(ascending=False).index[:max_genes])
        if pick:
            sub = M[pick]
            mx = sub.abs().idxmax()
            pick = sorted(pick, key=lambda g: (sub.loc[mx[g], g] < 0, rl.index(mx[g]), -abs(sub.loc[mx[g], g])))
        heat[gs_] = M[pick] if pick else pd.DataFrame(index=rl)
    ns, npc = max(heat["surface"].shape[1], 1), max(heat["protein_coding"].shape[1], 1)
    fig = plt.figure(figsize=figsize or (4.4 + 0.24 * (ns + npc) + 3.2, 1.8 + 0.34 * len(rl)))
    gsp = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.3, 0.24 * ns + 0.4, 0.24 * npc + 0.4], wspace=0.12)
    a1, a2 = fig.add_subplot(gsp[0]), fig.add_subplot(gsp[1])
    y = np.arange(len(rl))
    for ax, cs, ttl in ((a1, ("surface_net_index", "pc_net_index"), "A  net index\n(up−down)/expr."),
                        (a2, ("surface_intensity", "pc_intensity"), "intensity\nmean |log2FC|")):
        ax.barh(y - 0.19, idx[cs[0]], height=0.36, color=pal[1], label="surface (SURFY)")
        ax.barh(y + 0.19, idx[cs[1]], height=0.36, color=pal[0], label="protein-coding")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_ylim(len(rl) - 0.5, -0.5)
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#ecebe7", lw=0.6)
        ax.set_axisbelow(True)
    a1.set_yticks(y)
    a1.set_yticklabels(rl)
    a2.set_yticks([])
    a1.legend(loc="upper left", bbox_to_anchor=(0, -0.06 * 12 / max(len(rl), 6)), fontsize=7.5, frameon=False)
    im = None
    for j, (gs_, ttl) in enumerate((("surface", "B  top surface genes"), ("protein_coding", "C  top protein-coding genes"))):
        ax = fig.add_subplot(gsp[2 + j])
        H = heat[gs_]
        if H.shape[1]:
            lim = max(2.0, float(np.nanpercentile(np.abs(H.values), 95)))
            im = _heat(ax, H, cm, lim)
        else:
            ax.text(0.5, 0.5, "no consistent changes", ha="center", va="center", transform=ax.transAxes, color=INK2)
            ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(len(rl) - 0.5, -0.5)
        ax.set_title(f"{ttl}\n(red = up, blue = down; log2FC)", loc="left", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=fig.axes[-1], fraction=0.04, pad=0.02).set_label("weighted mean log2FC")
    fig.suptitle((title or f"{dr.name}: change indices and top gene changes") + "\n", x=0.01, y=1.04, ha="left",
                 va="bottom", fontsize=11.5)
    fig.text(0.01, 1.03, f"genes: each row's top {n_up} induced + {n_down} repressed genes with |mean log2FC| > 1 in "
             f">= 50% of its contrasts (union, max {max_genes}); unexpressed genes count as 0; net index = (up - down) / "
             "expressed genes; intensity = mean |log2FC| over expressed genes", fontsize=7.5, color=INK2, va="bottom")
    return _finish(fig, save, show, f"{dr.name}_top_gene_changes")


def signature_changes(dr, n_top=12, drugs=None, classes=None, max_genes=40, cmap=None, palette=None, figsize=None,
                      fontsize=None, title=None, save=None, show=True):
    """How the query's own genes move per drug: heatmap rows = drugs (or classes), columns = signature up genes
    (orange labels) then down genes (blue labels), colour = weighted mean log2FC. Left bars: mean change of the
    up genes minus the down genes (negative = the drug reverses the signature)."""
    _style(fontsize)
    pal = _pal(palette)
    cm = _div(cmap)
    M = tl.signature_changes(dr, drugs, classes, n_top, max_genes)
    if M.shape[1] == 0:
        raise ValueError("no signature genes in the contrast matrix")
    dirn = M.attrs["direction"]
    up = [g for g in M.columns if dirn[g] == "up"]
    dn = [g for g in M.columns if dirn[g] == "down"]
    eff = M[up].mean(1) - (M[dn].mean(1) if dn else 0)
    rl = list(M.index)
    fig = plt.figure(figsize=figsize or (3.2 + 0.24 * M.shape[1] + 2.5, 1.8 + 0.34 * len(rl)))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.4, 0.24 * M.shape[1] + 0.4], wspace=0.08)
    a1, a2 = fig.add_subplot(gsp[0]), fig.add_subplot(gsp[1])
    y = np.arange(len(rl))
    a1.barh(y, eff, color=[pal[0] if v < 0 else pal[1] for v in eff], height=0.6)
    a1.axvline(0, color=INK2, lw=0.7)
    a1.set_yticks(y)
    a1.set_yticklabels(rl)
    a1.set_ylim(len(rl) - 0.5, -0.5)
    a1.tick_params(axis="y", length=0)
    a1.set_title("up − down\n(< 0 = reversed)", loc="left", fontsize=9)
    lim = max(1.0, float(np.nanpercentile(np.abs(M.values), 95)))
    im = _heat(a2, M, cm, lim, direction=dirn, pal=pal)
    if dn:
        a2.axvline(len(up) - 0.5, color=INK, lw=1.2)
    a2.set_yticks([])
    a2.set_ylim(len(rl) - 0.5, -0.5)
    a2.set_title(f"signature genes: {len(up)} up (orange labels) | {len(dn)} down (blue labels)\ncolour = log2FC "
                 "treated vs control", loc="left", fontsize=9)
    fig.colorbar(im, ax=a2, fraction=0.03, pad=0.01).set_label("weighted mean log2FC")
    fig.suptitle(title or f"{dr.name}: how the query signature changes per drug", x=0.01, ha="left", fontsize=11.5,
                 y=1.02)
    return _finish(fig, save, show, f"{dr.name}_signature_changes")


# ============================================================================== overviews
def summary(mt, dr, n_top=12, palette=None, figsize=None, fontsize=None, title=None, save=None, show=True):
    """Three-panel overview: A) pseudotime position of the matched lines (and query), B) drug shifts,
    C) state map."""
    _style(fontsize)
    pal = _pal(palette)
    fig = plt.figure(figsize=figsize or (16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.4], width_ratios=[1.45, 1], hspace=0.35, wspace=0.42)
    a, b, c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    _pseudotime_strip(a, mt, list(dr.cell_lines.index), pal)
    a.set_xlabel("differentiation pseudotime (Zeng 2022 hierarchy): stem -> GMP -> myeloid")
    a.set_title("A  Query and closest untreated cell lines before treatment", loc="left", fontsize=10, pad=26)
    drug_shifts(dr, n_top=n_top, palette=pal, ax=b, show=False, title="B  Top drugs: untreated -> treated")
    state_map(dr, n_top=min(n_top, 8), palette=pal, ax=c, show=False, show_background=True,
              title="C  State map (arrows = top drugs)")
    fig.suptitle(title or f"{mt.name}", x=0.01, ha="left", fontsize=12.5, y=0.995)
    return _finish(fig, save, show, f"{mt.name}_summary")


def group_summary(results, n_top_drugs=5, min_contrasts=3, max_classes=25,
                  pinned_classes=("menin-KMT2A", "DOT1L", "LSD1/KDM1A", "DNMT/HMA", "splicing (SF3B1/RBM39)"),
                  cmap_score=None, cmap_shift=None, figsize=None, fontsize=None, title=None, save=None, show=True):
    """Compare several queries: results = {name: (MatchResult, DrugResult)}.
    Panels: match score of each query's selected cell lines; median pseudotime shift per drug class
    (pinned classes first, then the most common); top drugs per query. Returns (fig, scores, shifts) when
    show=False."""
    _style(fontsize)
    names = list(results)
    cl = sorted({c for n in names for c in results[n][1].cell_lines.index})
    M = pd.DataFrame(np.nan, index=names, columns=cl)
    for n in names:
        mt, dr = results[n]
        M.loc[n, dr.cell_lines.index] = mt.table.score.reindex(dr.cell_lines.index).values
    K = pd.DataFrame({n: tl.drug_classes(results[n][1], min_contrasts).set_index("drug_class").diff_shift
                      for n in names}).T
    K = K[[c for c in K.columns if not c.startswith(("other", "inactive", "unannotated", "unclassified"))]]
    keep = [c for c in pinned_classes if c in K]
    rest = [c for c in K.notna().sum().sort_values(ascending=False).index if c not in keep]
    K = K[keep + rest[: max(0, max_classes - len(keep))]]
    geno = {}
    for n in names:
        geno.update(results[n][0].table.genotype.to_dict())
    fig = plt.figure(figsize=figsize or (max(16, 0.55 * len(cl) + 0.55 * K.shape[1] + 8), 0.55 * len(names) + 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[max(4, len(cl)), max(4, K.shape[1]), 6.5], wspace=0.08)
    a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    a1.imshow(M.values, cmap=cmap_score or settings.score_cmap, aspect="auto")
    a1.set_xticks(range(len(cl)))
    a1.set_xticklabels([f"{c[:22]} [{geno.get(c, '')}]" for c in cl], rotation=90, fontsize=7.5)
    a1.set_yticks(range(len(names)))
    a1.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(cl)):
            if np.isfinite(M.values[i, j]):
                hi = np.nanmin(M.values) + 0.65 * (np.nanmax(M.values) - np.nanmin(M.values))
                a1.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color=INK if M.values[i, j] > hi else "white")
    a1.set_title("Selected cell lines per query (match score)", loc="left", fontsize=9.5)
    lim = np.nanpercentile(np.abs(K.values), 95) if np.isfinite(K.values).any() else 0.1
    a2.imshow(K.values, cmap=_div(cmap_shift), vmin=-lim, vmax=lim, aspect="auto")
    a2.set_xticks(range(K.shape[1]))
    a2.set_xticklabels(K.columns, rotation=90, fontsize=8)
    a2.set_yticks([])
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if np.isfinite(K.values[i, j]):
                a2.text(j, i, f"{K.values[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    a2.set_title(f"Median pseudotime shift per drug class\n(red = toward myeloid, blue = toward stem; >= "
                 f"{min_contrasts} contrasts)", loc="left", fontsize=9.5)
    a3.axis("off")
    a3.set_ylim(len(names) - 0.5, -0.5)
    a3.set_xlim(0, 1)
    for i, n in enumerate(names):
        d = results[n][1].drugs.head(n_top_drugs)
        a3.text(0.02, i, "; ".join(f"{x[:18]}{'*' if e else ''} ({t})" for x, e, t in zip(d.drug_name, d.epigenetic,
                                                                                         d.tier)),
                va="center", fontsize=7.5)
    a3.set_title(f"Top {n_top_drugs} drugs (tier; * = epigenetic)", loc="left", fontsize=9.5)
    fig.suptitle(title or "Query comparison", x=0.01, ha="left", fontsize=12)
    out = _finish(fig, save, show, "group_summary")
    return (out, M, K) if show is False else None
