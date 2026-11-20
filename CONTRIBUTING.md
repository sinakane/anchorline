# Contributing

Bug reports and suggestions are welcome as [issues](https://github.com/sinakane/anchorline/issues).

## Development setup
```bash
git clone https://github.com/sinakane/anchorline && cd anchorline
python -m venv .venv && source .venv/bin/activate      # or a conda env with Python >= 3.9
pip install -e ".[dev,labels]"
pytest                                                  # fast tests, no atlas needed
anchorline-download all                                 # atlas + demo data (needed for notebooks / integration)
python tests/integration/run_smoke.py                  # every query type and every plot
```

## Guidelines
- Keep computation in `tl` and drawing in `pl`: plot functions take a result object and only choose what to show.
- Every `pl` function accepts `n_top` or explicit names, `figsize`, `palette`/`cmap`, `fontsize`, `title`, `save`,
  `show` (and `ax` for single-panel plots).
- If you change the scoring maths, update the formulas in `examples/00_tutorial_how_it_works.ipynb` and the README.
- New bundled signatures need a verified source (journal, year, DOI) in `resources/signatures/catalog.tsv`.
- Use short, descriptive commit messages.
