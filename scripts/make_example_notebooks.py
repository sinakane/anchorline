"""Write the three example notebooks in examples/ (python scripts/make_example_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

EX = Path(__file__).resolve().parents[1] / "examples"
EX.mkdir(exist_ok=True)

SETUP = """
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import anchorline as al

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 60)
al.settings.figdir = "figures"          # save=True writes here
al.settings.set_figure_params(fontsize=8.5, dpi=150)
atlas = al.load()                       # first time: downloads the atlas (~560 MB) to al.settings.data_dir
atlas
"""


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(c[1].strip()) if c[0] == "md" else nbf.v4.new_code_cell(c[1].strip())
                   for c in cells]
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, EX / name)
    print("wrote", EX / name)


# ============================================================================== 0. tutorial: how it works
build("00_tutorial_how_it_works.ipynb", [
    ("md", """
# anchorline tutorial: how each step works

This notebook walks through what `anchorline` computes. Each step has its formula, the call that runs it, and a
check that **recomputes the step by hand** from the outputs. The formulas are therefore verified against the code,
not just described.

**Notation**

| symbol | meaning |
|---|---|
| $x_{g,s}$ | expression of gene $g$ in sample $s$ (any scale) |
| $r_{g,s}$ | within-sample rank of gene $g$ in sample $s$, scaled to 0–1 |
| $p(s)$ | platform of sample $s$ (RNA-seq, microarray, sci-Plex pseudobulk) |
| $U$, $D$ | up and down genes of a signature |
| $c$ | a cell line; $j$ a study; $i$ a drug-vs-control comparison ("contrast") |
"""),
    ("code", SETUP),
    ("md", r"""
## Step 0: one scale for every platform

The atlas mixes RNA-seq, microarrays and single-cell pseudobulk, so expression values are not comparable.
Every sample is therefore stored as **within-sample ranks**:

$$r_{g,s} = \frac{\operatorname{rank}(x_{g,s}) - 1}{n_s - 1} \in [0, 1]$$

where $n_s$ is the number of measured genes in the sample. A value of 0.9 means "in the top 10 % of genes in this
sample", on any platform.

Each rank is then **centred on the gene's platform mean**, which removes probe and platform effects:

$$\tilde r_{g,s} = r_{g,s} - \bar r_{g,\,p(s)}$$

The reference holds 2,805 untreated samples (DMSO, vehicle or no treatment). Below is HOXA9 in a few lines.
"""),
    ("code", """
ctl = atlas.controls
g = atlas.gi["HOXA9"]
R = pd.DataFrame({"cell_line": ctl.cell_line.values, "platform": ctl.platform.values,
                  "rank_HOXA9": atlas.R[atlas.ctrl, g].astype(float),
                  "centred_HOXA9": atlas.Rc[:, g]})
R[R.cell_line.isin(["MOLM-13", "MV4-11", "NOMO-1", "Kasumi-1", "K562", "HL-60"])].groupby("cell_line")[
    ["rank_HOXA9", "centred_HOXA9"]].median().sort_values("rank_HOXA9", ascending=False).round(3)
"""),
    ("md", r"""
## Step 1: the differentiation axis (Zeng 2022 hierarchy)

Zeng et al. 2022 (*Nat Med* 28:1212–1223, doi:10.1038/s41591-022-01819-x) defined AML cell-type signatures from
single-cell data. For each state $k$ the **state score** is the mean rank of its genes $G_k$:

$$S_{k,s} = \frac{1}{|G_k|}\sum_{g \in G_k} r_{g,s}$$

Scores are z-scored over the reference ($z_{k,s}$) and turned into weights with a softmax. **Pseudotime** is the
weighted position on the axis:

$$w_{k,s} = \frac{e^{z_{k,s}}}{\sum_{k'} e^{z_{k',s}}}, \qquad
\tau_s = \sum_k w_{k,s}\,\mathrm{pos}_k$$

with $\mathrm{pos}$ = quiescent LSPC 0, primed LSPC 1, cycling LSPC 1.5, GMP 2, ProMono 3, Mono 4.

The 2-D **state map** uses two differences:

$$\text{GMP}\rightarrow\text{myeloid} = \tfrac12(S_\text{ProMono} + S_\text{Mono}) - S_\text{GMP}, \qquad
\text{GMP}\rightarrow\text{primitive} = \tfrac12(S_\text{Quiescent} + S_\text{Primed}) - S_\text{GMP}$$
"""),
    ("code", """
import numpy as np
from anchorline._core import STATES, state_scores, add_pseudotime
s0 = atlas.ctrl[np.isfinite(atlas.controls.pseudotime.values)][:5]   # 5 untreated samples (52 have no score)
S = add_pseudotime(state_scores(atlas.R[s0], atlas.genes, atlas.sets), atlas.mu, atlas.sd)
check = pd.DataFrame({"recomputed": S.pseudotime.values, "stored": atlas.obs.pseudotime.values[s0]})
print("pseudotime recomputed == stored:", np.allclose(check.recomputed, check.stored, atol=1e-3))
S_all = add_pseudotime(state_scores(atlas.R, atlas.genes, atlas.sets), atlas.mu, atlas.sd)
print("... and for all", len(S_all), "reference samples:",
      np.allclose(S_all.pseudotime.values, atlas.obs.pseudotime.values, atol=1e-3, equal_nan=True))
check.round(3)
"""),
    ("md", r"""
## Step 2a: matching a signature

For every untreated sample $s$, the match score is the mean centred rank of the up genes minus that of the down genes:

$$m_s = \frac{1}{|U|}\sum_{g\in U}\tilde r_{g,s} \;-\; \frac{1}{|D|}\sum_{g\in D}\tilde r_{g,s}$$

Genes missing from the reference are skipped.
"""),
    ("code", """
sig = al.sig.get("Umeda2024_KMT2Ar_high")
mt = al.tl.match(atlas, sig)
ui = [atlas.gi[x] for x in mt.up]; di = [atlas.gi[x] for x in mt.down]
m_hand = np.nanmean(atlas.Rc[:, ui], 1) - np.nanmean(atlas.Rc[:, di], 1)   # all untreated samples (NaN = gene not measured)
m_hand = pd.Series(m_hand, index=atlas.controls.index).loc[mt.samples.index]
print("signature score by hand == package:", np.allclose(m_hand.values, mt.samples.score.values, atol=1e-5, equal_nan=True))
"""),
    ("md", r"""
## Step 2b: matching a patient sample

A whole profile is compared without letting platform or tumour-vs-cell-line differences dominate.

1. Rank the query within itself: $r_g^{q}$.
2. **Centre it on its own cohort** $B$ (the other samples in your table or AnnData), so that only what is specific to
   this patient remains:
   $$q_g = r^{q}_g - \frac{1}{|B|}\sum_{b\in B} r_{g,b}$$
   Without a background, the atlas mean is used instead.
3. Keep the $H$ = 2,000 genes that vary most across the reference.
4. The score is the Spearman correlation on those genes:
   $$m_s = \rho_{\text{Spearman}}\big(q_H,\; \tilde r_{H,s}\big)$$

The query also gets its own pseudotime $\tau_q$ (the star in the plots) and a **patient-specific signature**: the
100 genes most above and the 100 most below its cohort. This signature is used for the reversal term below.
"""),
    ("code", """
adata = al.datasets.beataml_demo()          # 45 BeatAML samples (downloads the demo data the first time)
pid = adata.obs_names[adata.obs.group == "KMT2A-r"][0]
mt_p = al.tl.match(atlas, adata, sample=pid)
print(mt_p.query, "| pseudotime of the patient:", round(float(mt_p.query_state.pseudotime), 2))
print("patient-specific up genes:", mt_p.up[:12])
mt_p.table.head(5)[["score", "studies", "genotype"]]
"""),
    ("md", r"""
## Step 3: from samples to a cell-line score

Scores are averaged within each study, then across studies:

$$M_c = \frac{1}{J_c}\sum_{j=1}^{J_c}\;\frac{1}{|s\in(c,j)|}\sum_{s\in(c,j)} m_s$$

Taking the **mean** over studies, not the best study, avoids favouring lines profiled in many studies. MOLM-13 has
44 studies, so its best study would be high by chance alone.

The selected lines get a **match weight**, used to rank drugs:

$$w_c = \operatorname{clip}\!\left(\frac{M_c - \operatorname{median}_{c'} M_{c'}}{\max_{c\in\text{sel}} M_c - \operatorname{median}_{c'} M_{c'}},\;0.05,\;1\right)$$
"""),
    ("code", """
t = mt.samples.groupby(["cell_line", "accession"]).score.mean().groupby("cell_line").mean().sort_values(ascending=False)
print("cell-line score by hand == package:", np.allclose(t.values, mt.table.score.values))
sel = mt.select(5); med = mt.table.score.median()
w_hand = ((mt.table.score[sel] - med) / (mt.table.score[sel].max() - med)).clip(0.05, 1)
pd.DataFrame({"score": mt.table.score[sel], "weight_by_hand": w_hand.round(3), "weight_package": mt.weights(5)})
"""),
    ("code", """
al.pl.match_ranking(mt, n_top=12)
al.pl.pseudotime(mt, n_top=5)
"""),
    ("md", r"""
## Step 4: three terms per drug comparison

For each comparison $i$ (one drug, dose and time in one cell line, against matched controls), the atlas stores
$\mathrm{LFC}_{g,i}$ = mean log2 expression of the treated samples − that of the controls. Counts are log2(CPM+1).

| term | formula | meaning |
|---|---|---|
| magnitude | $A_i = \sqrt{\tfrac{1}{\lvert PC\rvert}\sum_{g\in PC}\mathrm{LFC}_{g,i}^2}$ | overall change of the ~19,300 protein-coding genes |
| reversal | $e_i = \overline{\mathrm{LFC}}_{U,i} - \overline{\mathrm{LFC}}_{D,i}$, scored as $-e_i$ | does the drug push the query signature down? |
| differentiation | $\Delta\tau_i = \tau(\text{treated}) - \tau(\text{control})$ | > 0: toward myeloid |

Each term becomes a **robust z-score**. Magnitude is scaled within each data type, because arrays give smaller
fold changes than RNA-seq:

$$z(x) = \frac{x - \operatorname{median}(x)}{1.4826\,\operatorname{MAD}(x)}$$

Each z-score is clipped to ±3 so that no single term dominates. Their mean is the comparison's **composite**:

$$C_i = \tfrac13\big[\operatorname{clip}(z(A_i)) + \operatorname{clip}(z(-e_i)) + \operatorname{clip}(z(\Delta\tau_i))\big]$$
"""),
    ("code", """
dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)
T = dr.contrasts
C_hand = T[["magnitude_z", "reversal_z", "diff_shift_z"]].clip(-3, 3).mean(1)
print("composite by hand == package:", np.allclose(C_hand, T.composite))
T[["drug_key", "cell_line", "dose", "time", "magnitude", "magnitude_z", "signature_change", "reversal_z",
   "diff_shift", "diff_shift_z", "composite"]].head(8).round(3)
"""),
    ("md", r"""
## Step 5: from comparisons to a drug score

Each comparison is weighted by its cell line's match weight and by its replicate support, with
$n_r = \min(n_\text{treated}, n_\text{control})$:

$$\omega_i = w_{c(i)}\cdot\frac{n_r}{n_r+1}$$

Per drug $d$, with $n_d$ comparisons:

$$C_d = \frac{\sum_{i\in d}\omega_i C_i}{\sum_{i\in d}\omega_i}, \qquad
\text{score}_d = C_d\cdot\frac{n_d}{n_d+1}$$

The factor $n_d/(n_d+1)$ shrinks thinly supported drugs: ×0.5 for one comparison, ×0.75 for three, ×0.9 for nine.
**Consistency** is the fraction of comparisons with $C_i > 0$.

**Tiers:**
- **T1:** at least 2 studies and consistency ≥ 75 %.
- **T2:** at least 2 comparisons.
- **T3:** a single comparison. Treat these as leads.
"""),
    ("code", """
d = "pinometostat"
g_ = T[T.drug_key == d]
C_d = np.average(g_.composite, weights=g_.weight)
score_hand = C_d * len(g_) / (len(g_) + 1)
print(f"{d}: {len(g_)} comparisons, C_d = {C_d:.3f}, score by hand = {score_hand:.3f}, package = {dr.drugs.score[d]:.3f}")
dr.drugs.head(10)[["drug_name", "tier", "score", "composite", "magnitude_z", "reversal_z", "diff_shift",
                   "consistency", "n_contrasts", "n_studies"]].round(3)
"""),
    ("code", """
al.pl.drug_ranking(dr, n_top=15)
al.pl.drug_shifts(dr, n_top=12)
al.pl.state_map(dr, n_top=6)
"""),
    ("md", r"""
## Step 6: drug classes

Comparisons are grouped by drug class; a combination counts in each of its classes. Per class, the plots show:
- the median $\Delta\tau$ with its interquartile range;
- the share of comparisons with $\Delta\tau > 0$ (toward myeloid);
- the median change on the GMP→myeloid and GMP→primitive axes (the shift map).
"""),
    ("code", """
dr.classes.head(8)[["drug_class", "n_contrasts", "n_drugs", "n_studies", "diff_shift", "diff_shift_q25",
                    "diff_shift_q75", "frac_toward_myeloid", "top_drugs"]].round(3)
"""),
    ("code", """
al.pl.class_trends(dr, n_top=12)
al.pl.class_shift_map(dr, classes=["menin-KMT2A", "DOT1L", "BET", "DNMT/HMA", "HDAC"])
"""),
    ("md", r"""
## Step 7: gene-level read-outs (reported, not scored)

**Expressed genes.** A gene counts as expressed in comparison $i$ if its mean within-sample rank is at least 0.5 in
the control samples or in the treated samples. Genes that are not expressed in either arm count as no change (0),
so noisy near-zero genes don't count.

$$E_{g,i} = \big[\max(\bar r_{g,\text{ctrl}},\ \bar r_{g,\text{treated}}) \ge 0.5\big]$$

**Change indices**, computed for the SURFY surfaceome (2,886 proteins; Bausch-Fluck et al. 2018 *PNAS*
115:E10988, doi:10.1073/pnas.1808790115) and for all protein-coding genes:

$$\text{net index}_i = \frac{\#\{g: E_{g,i},\ \mathrm{LFC}_{g,i} > 1\} - \#\{g: E_{g,i},\ \mathrm{LFC}_{g,i} < -1\}}{\#\{g: E_{g,i}\}} \in [-1, 1],
\qquad
\text{intensity}_i = \frac{\sum_g E_{g,i}\,|\mathrm{LFC}_{g,i}|}{\#\{g: E_{g,i}\}}$$

Per drug, gene values are $\omega$-weighted means. A gene is listed as a drug's top change only if
$|\overline{\mathrm{LFC}}| > 1$ **and** it changes in at least half of that drug's comparisons.
"""),
    ("code", """
i = T.index[0]
print("net index by hand:", round((T.n_surface_up[i] - T.n_surface_down[i]) / T.n_expressed_surface[i], 4),
      "| package:", round(T.surface_net_index[i], 4))
al.pl.gene_changes(dr, genes=["CLEC12A", "CD84"], n_top=10)
al.pl.top_gene_changes(dr, n_top=10)
al.pl.signature_changes(dr, n_top=10)
"""),
    ("md", """
## Summary of the workflow

```python
atlas = al.load()                                   # downloads the atlas the first time
mt = al.tl.match(atlas, query)                      # Steps 0-3
dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)      # Steps 4-5
dr.classes                                          # Step 6
al.tl.gene_changes(dr, genes); al.tl.change_indices(dr)   # Step 7
```

**Caveats**
- A match means "most similar available model", not the same disease. If a group has no matching line, it lands on generic lines, and a low top score with a flat ranking is the warning sign.
- Match scores are relative: compare them across lines within one query, not across queries.
- T3 drugs rest on a single comparison.

Continue with `01_builtin_signatures`, `02_custom_signature` and `03_sample_csv_and_anndata`.
"""),
])

# ============================================================================== 1. built-in signatures
build("01_builtin_signatures.ipynb", [
    ("md", """
# 1. Built-in signatures: from a signature to cell lines to drugs

`anchorline` anchors a query (a signature, a patient sample or single-cell data) to the closest untreated AML cell
lines in the drug-perturbation atlas. It then ranks the drugs tested in those lines by how much they change the cells,
whether they reverse the query signature, and how far they move the cells along the stem → myeloid axis.

The workflow has two steps, each computed once (`al.tl`), and plot functions that only choose what to show (`al.pl`):

```
mt = al.tl.match(atlas, query)              # rank all cell lines   -> al.pl.match_ranking, al.pl.pseudotime
dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)   # rank drugs in them -> al.pl.drug_ranking, drug_shifts, state_map,
                                                 #                       class_trends, class_shift_map, gene_changes, ...
```
Every plot takes `n_top=` (how many) or explicit names (`cell_lines=`, `drugs=`, `classes=`), plus `figsize`,
`palette`, `cmap`, `fontsize`, `title`, `ax`, `save` and `show`.
"""),
    ("code", SETUP),
    ("md", """
## The signature library
`al.sig.list()` shows every bundled signature with its source. Collections:
- **Umeda2024_pediatric_AML**: 20 pediatric AML groups (Umeda et al. 2024 *Nat Genet* 56:281–293, doi:10.1038/s41588-023-01640-3), top 50 protein-coding genes per group, up and down.
- **Zeng2022_AML_hierarchy**: the AML cell-type signatures behind the pseudotime axis (Zeng et al. 2022 *Nat Med* 28:1212–1223, doi:10.1038/s41591-022-01819-x).
- **stemness**:
  - LSC104 (Ng et al. 2016 *Nature* 540:433–437, doi:10.1038/nature20598);
  - MLL-LSC (Somervaille et al. 2009 *Cell Stem Cell* 4:129–140, doi:10.1016/j.stem.2008.11.015);
  - HSC_Cycle_Primed (source not verified).
- **menin_targets**: the HOXA/MEIS1 core (Issa et al. 2023 *Nature* 615:920–924, doi:10.1038/s41586-023-05812-3).
"""),
    ("code", """
al.sig.collections()
"""),
    ("code", """
al.sig.list("Umeda2024")
"""),
    ("code", """
display(al.sig.search("MEIS1"))         # which signatures contain a gene
s = al.sig.get("Umeda2024_KMT2Ar_high")  # exact name, or a unique part of it: al.sig.get("KMT2Ar_high")
print(s); print("up:", s.up[:10], "...")
"""),
    ("md", """
## Step 1: match the signature to cell lines
`al.tl.match` scores every untreated sample in the atlas and ranks all models. A cell line's score is the mean over its studies.
- `models="cell_lines"` is the default; `"patient_derived"` (PDX/primary only) and `"all"` are also available.
- Lymphoid lines are excluded unless you pass `exclude=()`.
"""),
    ("code", """
mt = al.tl.match(atlas, s)
mt
"""),
    ("code", """
mt.table.head(10)       # the full ranking is a DataFrame: filter it, export it, pick names from it
"""),
    ("code", """
al.pl.match_ranking(mt, n_top=15, highlight=5)
"""),
    ("code", """
al.pl.match_ranking(mt, by="study", n_top=15)     # cell line | study pairs
"""),
    ("md", """
**Pseudotime.**
- **Top panel:** every untreated sample in the atlas (grey), with the most similar lines in colour.
- **Bottom panel:** only the matched lines, each untreated sample coloured by its own match score (viridis). The black tick marks the line's mean position.
"""),
    ("code", """
al.pl.pseudotime(mt, n_top=5)
"""),
    ("md", """
## Step 2: rank the drugs tested in the chosen cell lines
Pass `cell_lines` as a number (the top n of the match) or as names, e.g. only the KMT2A-r lines.
"""),
    ("code", """
dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)
dr
"""),
    ("code", """
dr.drugs.head(10)
"""),
    ("code", """
al.pl.drug_ranking(dr, n_top=15)                       # bars = score; markers = the three terms
"""),
    ("code", """
al.pl.drug_ranking(dr, n_top=20, color_by="drug_class", components=False)
"""),
    ("code", """
al.pl.drug_shifts(dr, n_top=12)                        # untreated -> treated pseudotime per drug
"""),
    ("code", """
al.pl.state_map(dr, n_top=6)                           # top 6 drugs ...
al.pl.state_map(dr, drugs=["ziftomenib", "pinometostat", "MI-3454", "SGC0946"], title="selected drugs")   # ... or by name
"""),
    ("md", "## Drug classes"),
    ("code", """
dr.classes.head(10)
"""),
    ("code", """
al.pl.class_trends(dr, n_top=12)
"""),
    ("code", """
al.pl.class_shift_map(dr, classes=["menin-KMT2A", "DOT1L", "BET", "DNMT/HMA", "HDAC", "BCL2"])
"""),
    ("md", """
## Genes of interest
`gene_changes` shows any genes per drug (or class) × cell line. Colour is log2FC, size is the number of comparisons, and a hollow dot means the gene isn't expressed.
"""),
    ("code", """
al.pl.gene_changes(dr, genes=["CLEC12A", "CD84"], n_top=12)
"""),
    ("code", """
al.pl.gene_changes(dr, genes=["CLEC12A", "CD84", "CD33", "CD14"], classes=["menin-KMT2A", "DOT1L", "BET", "HDAC"],
                   title="selected genes by drug class")
"""),
    ("code", """
al.tl.gene_changes(dr, ["CLEC12A", "CD84"]).query("cell_line == 'pooled'").round(2)   # the numbers behind the plot
"""),
    ("md", """
**Top gene changes per drug.**
- **A:** change indices. The net index is (genes up − genes down) / expressed genes; intensity is the mean |log2FC|.
- **B:** the most changed surface (SURFY) genes.
- **C:** the most changed protein-coding genes.
"""),
    ("code", """
al.pl.top_gene_changes(dr, n_top=12)
"""),
    ("md", """
**The signature itself**: how each drug moves the query's up genes (orange labels) and down genes (blue labels). A reversal shows as blue up-genes and red down-genes, with a negative bar on the left.
"""),
    ("code", """
al.pl.signature_changes(dr, n_top=12)
"""),
    ("code", """
al.pl.summary(mt, dr, n_top=12, save=True)        # three-panel overview, saved to figures/
"""),
    ("md", """
## Your choice of cell lines
After seeing the match, you can rank drugs in exactly the lines you trust. Here that is the four KMT2A-r lines, leaving out IMS-M2.
"""),
    ("code", """
dr_kmt2a = al.tl.rank_drugs(atlas, mt, cell_lines=["OCI-AML2", "NOMO-1", "MOLM-13", "MOLM-14"])
al.pl.drug_ranking(dr_kmt2a, n_top=12)
"""),
    ("md", "## Several signatures at once"),
    ("code", """
res = {}
for g in ["KMT2Ar_high", "HOXA", "RUNX1T1", "AMKL", "Immature", "NUP98r"]:
    m = al.tl.match(atlas, f"Umeda2024_{g}")
    res[g] = (m, al.tl.rank_drugs(atlas, m, cell_lines=5))
al.pl.group_summary(res, title="Umeda 2024 groups: matched lines, class shifts, top drugs")
"""),
    ("code", """
# export any table
mt.table.to_csv("figures/KMT2Ar_high_cell_lines.tsv", sep="\\t")
dr.drugs.to_csv("figures/KMT2Ar_high_drugs.tsv", sep="\\t")
al.tl.surface_genes(dr).to_csv("figures/KMT2Ar_high_surface_genes.tsv", sep="\\t", index=False)
"""),
])

# ============================================================================== 2. user-defined signatures
build("02_custom_signature.ipynb", [
    ("md", """
# 2. Your own signature

A signature can be:
- a gene list (up genes only);
- a dict `{"up": [...], "down": [...]}`;
- a CSV/TSV with `gene` and `direction` columns (up/down, +/−, 1/−1), or with side-by-side `up` and `down` columns;
- a `.gmt` file (pairs named `X_UP` / `X_DOWN` are merged);
- a signature you register with `al.sig.register`, which then works like a built-in one.
"""),
    ("code", SETUP),
    ("md", "## A gene list"),
    ("code", """
mt = al.tl.match(atlas, ["HOXA9", "HOXA10", "MEIS1", "PBX3", "MEF2C"], name="my HOX genes")
al.pl.match_ranking(mt, n_top=12)
"""),
    ("md", "## Up and down genes"),
    ("code", """
stem = {"up": ["CD34", "HLF", "MECOM", "PROM1", "SPINK2", "CRHBP", "GATA2", "MEIS1"],
        "down": ["MPO", "ELANE", "LYZ", "CD14", "S100A8", "S100A9", "CEBPE", "ITGAM"]}
mt = al.tl.match(atlas, stem, name="stem vs myeloid")
mt.info                                   # how many genes were found in the atlas
"""),
    ("code", """
al.pl.pseudotime(mt, n_top=6)
"""),
    ("code", """
dr = al.tl.rank_drugs(atlas, mt, cell_lines=4)
al.pl.drug_ranking(dr, n_top=12)
al.pl.signature_changes(dr, n_top=10)
"""),
    ("md", "## From a CSV file"),
    ("code", """
from pathlib import Path
Path("figures").mkdir(exist_ok=True)
pd.DataFrame({"gene": stem["up"] + stem["down"],
              "direction": ["up"] * len(stem["up"]) + ["down"] * len(stem["down"])}).to_csv("figures/my_signature.csv", index=False)
q = al.io.read_query("figures/my_signature.csv")        # read_query shows what was understood
q
"""),
    ("code", """
mt_csv = al.tl.match(atlas, "figures/my_signature.csv")
(mt_csv.table.index[:5] == mt.table.index[:5]).all()     # same result as the dict
"""),
    ("md", "## Register it, then use it like a built-in signature"),
    ("code", """
al.sig.register("my_stem_vs_myeloid", up=stem["up"], down=stem["down"], description="toy stem-vs-myeloid set")
al.sig.list("my_")
"""),
    ("code", """
mt = al.tl.match(atlas, "my_stem_vs_myeloid")
dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)
al.pl.class_trends(dr, n_top=10)
al.pl.gene_changes(dr, genes=["CLEC12A", "CD84"], n_top=10)
"""),
    ("md", "## A .gmt file"),
    ("code", """
gmt = Path("figures/my_sets.gmt")
gmt.write_text("MYSET_UP\\tdemo\\t" + "\\t".join(stem["up"]) + "\\nMYSET_DOWN\\tdemo\\t" + "\\t".join(stem["down"]) + "\\n")
sets = al.sig.read_gmt(gmt)               # MYSET_UP + MYSET_DOWN -> one signature "MYSET"
sets
"""),
    ("md", """
## Including PDX and primary models
`models="all"` lets PDX and primary samples match too. Their drug lists are short, though: most have only 1–5 drug comparisons.
"""),
    ("code", """
mt_all = al.tl.match(atlas, "Umeda2024_KMT2Ar_high", models="all")
mt_all.table.head(8)[["score", "studies", "genotype", "is_cell_line"]]
"""),
])

# ============================================================================== 3. samples: CSV and AnnData
build("03_sample_csv_and_anndata.ipynb", [
    ("md", """
# 3. Patient samples: a CSV table and an AnnData

A sample query is a whole expression profile on any scale; it is converted to within-sample ranks. Give the rest of
the cohort as the **background** so that the match reflects what is specific to the patient. Tumour-vs-cell-line
differences shared by all samples are then centred away.
- **Table (CSV/TSV):** genes × samples; pick one with `sample=`. The other columns become the background.
- **AnnData:** pick an observation with `sample=`, or pseudobulk a group with `groupby=` / `group=`. Counts are summed and log-CPM normalised; log data are averaged.

The demo data are 45 BeatAML diagnosis bone-marrow samples: 15 KMT2A-r, 15 NPM1-mutant and 15 CBFB-MYH11 (BeatAML 2.0, Bottomly et al. 2022 *Cancer Cell* 40:850–864, doi:10.1016/j.ccell.2022.07.002). They are downloaded the first time with `al.datasets.download("demo")`, which runs automatically below.
"""),
    ("code", SETUP),
    ("code", """
D = al.datasets.demo_path().parent          # downloads the demo set the first time
samples = pd.read_csv(D / "beataml_demo_samples.csv", index_col=0)
samples.group.value_counts()
"""),
    ("md", "## A sample from a CSV table"),
    ("code", """
sid = samples.index[samples.group == "KMT2A-r"][0]
print(sid, samples.loc[sid, "fusion"])
mt = al.tl.match(atlas, D / "beataml_demo_expression.csv", sample=sid)
mt
"""),
    ("code", """
al.pl.match_ranking(mt, n_top=12)
al.pl.pseudotime(mt, n_top=5)            # the star = this patient's position
"""),
    ("code", """
dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)
al.pl.drug_ranking(dr, n_top=12)
al.pl.drug_shifts(dr, n_top=10)
al.pl.state_map(dr, n_top=6)
"""),
    ("code", """
al.pl.gene_changes(dr, genes=["CLEC12A", "CD84"], n_top=10)
al.pl.top_gene_changes(dr, n_top=10)
"""),
    ("md", """
For a sample query, `signature_changes` uses the patient's own most specific genes: the 100 most above and 100 most below its cohort, capped for display.
"""),
    ("code", """
al.pl.signature_changes(dr, n_top=10, max_genes=30)
"""),
    ("md", "## The same data as an AnnData"),
    ("code", """
adata = al.datasets.beataml_demo()
adata
"""),
    ("code", """
# one observation; the other observations are the background
sid2 = samples.index[samples.group == "CBFB-MYH11"][0]
mt_cbf = al.tl.match(atlas, adata, sample=sid2, name=f"{sid2} CBFB-MYH11")
al.pl.pseudotime(mt_cbf, n_top=5)
"""),
    ("md", """
**Pseudobulk per group** works the same way for single-cell data, e.g. `groupby="leiden"` / `group="3"`.

Here each genotype group is pooled, and the other two groups act as the background. With only three groups that background is small: for real single-cell data, use many clusters or samples.
"""),
    ("code", """
res = {}
for g in ["KMT2A-r", "NPM1-mut", "CBFB-MYH11"]:
    m = al.tl.match(atlas, adata, groupby="group", group=g, name=f"BeatAML {g} (pseudobulk)")
    res[g] = (m, al.tl.rank_drugs(atlas, m, cell_lines=5))
    print(g, "->", ", ".join(f"{c} [{t}]" for c, t in m.table.genotype.head(5).items()))
"""),
    ("code", """
al.pl.group_summary(res, title="BeatAML genotype groups (pseudobulk)")
"""),
    ("code", """
m, d = res["KMT2A-r"]
al.pl.summary(m, d, n_top=10)
al.pl.class_trends(d, classes=["menin-KMT2A", "DOT1L", "BET", "FLT3", "BCL2", "DNMT/HMA"])
"""),
])
