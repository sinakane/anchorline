# Changelog

## 0.1.0 (2026-11-20)
First release.
- `al.tl.match`: matches signatures, expression tables, Series and AnnData (single sample or pseudobulk) to the
  closest untreated cell lines (or PDX/primary models).
- `al.tl.rank_drugs`: scores drugs on magnitude, signature reversal and differentiation shift, with tiers.
- `al.tl` gene tools: gene changes, top gene changes, surface / protein-coding change indices, signature changes.
- `al.pl`: match ranking, pseudotime, drug ranking, drug shifts, state map, class trends, class shift map, gene
  changes, top gene changes, signature changes, summary, group summary.
- `al.sig`: 38 bundled signatures (Umeda 2024, Zeng 2022, LSC104, MLL-LSC, HOXA/MEIS1 core), search, register, gmt.
- `al.datasets` / `anchorline-download`: one-line download of the atlas (release `data-v1`) and BeatAML demo data.
- Results are identical to the original benchmark implementation (`tests/integration/compare_with_benchmark.py`).
