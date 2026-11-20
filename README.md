<!-- logo: replace with <p align="center"><img src="docs/logo.png" width="260"></p> once available -->

# anchorline

[![tests](https://github.com/sinakane/anchorline/actions/workflows/tests.yml/badge.svg)](https://github.com/sinakane/anchorline/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](pyproject.toml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Anchor an AML signature or patient sample to the closest drug-perturbed cell lines, then rank the drugs tested in
those lines by how they move the cells.**

`anchorline` compares your query with 2,805 untreated reference samples (cell lines, PDX and primary AML) in a
harmonised drug-perturbation atlas. The atlas holds 1,608 drug-vs-control comparisons from 226 public studies,
covering 345 drugs, and spans RNA-seq, microarrays and single-cell data. For the closest cell lines it ranks every drug tested in them on three things:

- **magnitude:** how much the drug changes the cells (protein-coding genes);
- **reversal:** whether it pushes *your* signature down;
- **differentiation:** how far it moves cells along the stem → GMP → monocyte axis (Zeng et al. 2022 hierarchy).

It also reports how chosen genes, surface proteins and protein-coding genes change, per drug and per drug class.

```
signature / patient sample / AnnData
        │  al.tl.match
        ▼
closest untreated cell lines ──► al.pl.match_ranking · al.pl.pseudotime
        │  al.tl.rank_drugs
        ▼
drugs tested in those lines ──► al.pl.drug_ranking · drug_shifts · state_map
                                 class_trends · class_shift_map
                                 gene_changes · top_gene_changes · signature_changes
```

---

## Installation

```bash
# 1. create an environment (any Python >= 3.9)
conda create -n anchorline python=3.11 -y && conda activate anchorline

# 2. install the package
pip install "git+https://github.com/sinakane/anchorline.git"
#    or, to edit the code:  git clone https://github.com/sinakane/anchorline && pip install -e "./anchorline[labels]"

# 3. download the atlas once (~560 MB, to ~/.anchorline/data)
anchorline-download            # or: python -c "import anchorline as al; al.datasets.download()"
```

- **Step 3 is optional:** `al.load()` downloads the atlas automatically the first time.
- **Demo data:** the example notebooks also need the BeatAML demo data (13 MB): `anchorline-download demo`, or `anchorline-download all` for everything.
- **Data folder:** to keep the data somewhere else, set `export ANCHORLINE_DATA=/path/to/folder` or `al.settings.data_dir = "/path"` in Python.

> **Private repository (current status):** installing and downloading need GitHub access. `pip` uses your git credentials. The downloader uses `$GITHUB_TOKEN` / `$GH_TOKEN`, or the token from `gh auth login` (set `GH_CONFIG_DIR` if your gh config is not in the default place). Once the repository is public, no token is needed.

Optional extras: `pip install "anchorline[labels]"` (non-overlapping labels via adjustText) and `"anchorline[notebooks]"`.

## Quick start

```python
import anchorline as al

atlas = al.load()

# 1. query -> closest cell lines
mt = al.tl.match(atlas, "Umeda2024_KMT2Ar_high")          # a built-in signature
al.pl.match_ranking(mt, n_top=15)
al.pl.pseudotime(mt, n_top=5)

# 2. drugs tested in the chosen lines
dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)            # or cell_lines=["MOLM-13", "NOMO-1"]
dr.drugs.head(10)
al.pl.drug_ranking(dr, n_top=15)
al.pl.drug_shifts(dr)
al.pl.state_map(dr, drugs=["ziftomenib", "pinometostat"])
al.pl.class_trends(dr)
al.pl.class_shift_map(dr, classes=["menin-KMT2A", "DOT1L", "BET"])
al.pl.gene_changes(dr, genes=["CLEC12A", "CD84"])
al.pl.top_gene_changes(dr)
al.pl.signature_changes(dr)
```

Every result is also a table you can filter or export: `mt.table`, `mt.groups`, `mt.samples`, `dr.drugs`,
`dr.contrasts` and `dr.classes`.

### Queries

`al.tl.match` accepts anything `al.io.read_query` understands:

| input | example |
|---|---|
| gene list | `al.tl.match(atlas, ["HOXA9", "MEIS1", "PBX3"])` |
| up / down genes | `al.tl.match(atlas, {"up": [...], "down": [...]})` |
| built-in signature | `al.tl.match(atlas, "Umeda2024_HOXA")` |
| signature file | CSV/TSV with `gene` + `direction` (up/down) columns, or a `.gmt` file |
| expression table | `al.tl.match(atlas, "cohort.csv", sample="P01")`: genes × samples; the other samples become the background |
| expression profile | a `pd.Series`, gene → value, any scale |
| AnnData | `al.tl.match(atlas, adata, sample="P01")`, or pseudobulk with `groupby="leiden", group="3"` |

Matching uses cell lines by default. `models="patient_derived"` (PDX and primary only) and `models="all"` are also available.

### Plot options (every `al.pl` function)

| option | meaning |
|---|---|
| `n_top` | how many items to show, best first |
| `cell_lines` / `drugs` / `classes` | show exactly these instead |
| `figsize`, `fontsize`, `title` | layout |
| `palette`, `cmap` | colours (match scores default to viridis; log2FC to blue-grey-red) |
| `ax` | draw into your own matplotlib axes (single-panel plots) |
| `save` | `True` writes to `al.settings.figdir`; or give a path |
| `show` | `False` returns the figure instead of displaying it |

Global defaults: `al.settings.set_figure_params(fontsize=..., dpi=..., file_format=..., figdir=..., palette=..., score_cmap=...)`.

### Built-in signatures

```python
al.sig.list()                      # name, collection, n_up, n_down, source, DOI
al.sig.search("MEIS1")             # which signatures contain a gene
al.sig.get("Umeda2024_RUNX1T1")    # -> Signature(up=[...], down=[...])
al.sig.register("mine", up=[...], down=[...])
al.sig.export("all_signatures.gmt")
```

| collection | content | source |
|---|---|---|
| Umeda2024_pediatric_AML | 20 pediatric AML groups, top 50 protein-coding genes (up + down) | Umeda et al. 2024 *Nat Genet* 56:281–293, [doi:10.1038/s41588-023-01640-3](https://doi.org/10.1038/s41588-023-01640-3) |
| Zeng2022_AML_hierarchy | LSPC, GMP, ProMono, Mono and cDC-like signatures | Zeng et al. 2022 *Nat Med* 28:1212–1223, [doi:10.1038/s41591-022-01819-x](https://doi.org/10.1038/s41591-022-01819-x) |
| stemness | LSC104 | Ng et al. 2016 *Nature* 540:433–437, [doi:10.1038/nature20598](https://doi.org/10.1038/nature20598) |
| | MLL-LSC (up + down) | Somervaille et al. 2009 *Cell Stem Cell* 4:129–140, [doi:10.1016/j.stem.2008.11.015](https://doi.org/10.1016/j.stem.2008.11.015) |
| | HSC_Cycle_Primed | source not verified |
| menin_targets | HOXA/MEIS1 core | Issa et al. 2023 *Nature* 615:920–924, [doi:10.1038/s41586-023-05812-3](https://doi.org/10.1038/s41586-023-05812-3) |

## Tutorials

| notebook | what it shows |
|---|---|
| [`00_tutorial_how_it_works`](examples/00_tutorial_how_it_works.ipynb) | every step with its formula, recomputed by hand from the outputs |
| [`01_builtin_signatures`](examples/01_builtin_signatures.ipynb) | the signature library, the full workflow, choosing your own lines, comparing groups |
| [`02_custom_signature`](examples/02_custom_signature.ipynb) | gene lists, up/down dicts, CSV and `.gmt` files, registering signatures, `models="all"` |
| [`03_sample_csv_and_anndata`](examples/03_sample_csv_and_anndata.ipynb) | a patient from a CSV table; AnnData (single sample and pseudobulk per group) |

## How it works (short version)

| step | what is computed |
|---|---|
| reference | every sample as within-sample gene ranks $r_{g,s}\in[0,1]$, centred on the gene's platform mean |
| hierarchy | Zeng 2022 state scores (mean rank); pseudotime = softmax-weighted position from quiescent LSPC (0) to Mono (4) |
| signature match | $m_s$ = mean centred rank of the up genes − mean of the down genes |
| sample match | Spearman ρ on the 2,000 most variable genes; the query is centred on its own cohort |
| cell-line score | mean over studies of the per-study mean; match weight $w_c$ = (score − median) / (best − median), clipped to [0.05, 1] |
| drug terms | magnitude (RMS log2FC, protein-coding), reversal (−Δ signature), differentiation (Δ pseudotime); robust z, clipped ±3, averaged |
| drug score | weighted mean over comparisons × n/(n+1); tiers T1 (≥2 studies, ≥75 % consistent), T2 (≥2 comparisons), T3 (single) |
| gene read-outs | expressed genes only; surface (SURFY) and protein-coding net index (up − down)/expressed and intensity (mean \|log2FC\|) |

The full formulas are in [`00_tutorial_how_it_works`](examples/00_tutorial_how_it_works.ipynb).

## Data

| file | content | size |
|---|---|---|
| `query_reference.h5ad` | 8,894 samples (2,805 untreated) × 42,003 genes, within-sample ranks, hierarchy scores | 292 MB |
| `AML_drug_atlas_bulk.h5ad` | 1,608 drug-vs-control comparisons × 44,444 genes (log2FC), with metadata | 289 MB |
| `drugs.tsv`, `cell_lines.tsv` | drug and model annotation (harmonised names, classes, Cellosaurus IDs) | < 1 MB |
| `beataml_demo*` | 45 BeatAML diagnosis samples for the examples | 13 MB |

- **Where they live:** the files are assets of the GitHub release [`data-v1`](https://github.com/sinakane/anchorline/releases/tag/data-v1). Each download is checked against its SHA-256 (`src/anchorline/resources/datasets.json`).
- **Atlas sources:** public GEO, ArrayExpress and LINCS studies, plus sci-Plex (Srivatsan et al. 2020 *Science*).
- **Demo data:** BeatAML 2.0 (Bottomly et al. 2022 *Cancer Cell* 40:850–864, [doi:10.1016/j.ccell.2022.07.002](https://doi.org/10.1016/j.ccell.2022.07.002)).
- **Surface proteins:** the SURFY in-silico surfaceome (Bausch-Fluck et al. 2018 *PNAS* 115:E10988, [doi:10.1073/pnas.1808790115](https://doi.org/10.1073/pnas.1808790115)).

## Caveats

- The reference is mostly cell lines, so a match means "most similar available model", not the same disease. Groups with no matching line (e.g. inv(16), CEBPA, CBFA2T3-GLIS2) land on generic lines. A low top score and a flat ranking are the warning signs.
- Match scores are relative: compare them across lines within one query.
- T3 drugs rest on a single comparison: treat them as leads to test.
- Pseudotime is relative (rank-based), not absolute developmental time.

## Development

```bash
pip install -e ".[dev,labels]"
pytest                                      # fast tests, no atlas needed (also run by GitHub Actions)
python tests/integration/run_smoke.py       # every query type and every plot (needs the data)
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use anchorline, please cite it ([CITATION.cff](CITATION.cff)) and the data sources above.

## License

[MIT](LICENSE)
