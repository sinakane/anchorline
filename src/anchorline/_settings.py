"""Global settings: where the atlas data live, where figures go, and figure defaults (scanpy-style)."""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA = Path.home() / ".anchorline" / "data"

# colour defaults (validated categorical order; diverging blue-grey-red with a neutral midpoint)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
GREY, INK, INK2, GRID, SURFACE = "#c9c8c3", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


class _Settings:
    """`al.settings` - edit attributes directly or use `set_figure_params`.

    data_dir : folder with the atlas (query_reference.h5ad, AML_drug_atlas_bulk.h5ad, drugs.tsv, cell_lines.tsv)
               and demo files. Default: $ANCHORLINE_DATA, else ~/.anchorline/data. Fill it with
               `al.datasets.download()` (al.load() does this automatically the first time).
    figdir   : where `save=True` writes figures (created on demand).
    file_format, dpi : format(s) and resolution for saved figures.
    """

    def __init__(self):
        self.data_dir = Path(os.environ.get("ANCHORLINE_DATA", _DEFAULT_DATA))
        self.figdir = Path("figures")
        self.file_format = ("png", "pdf")
        self.dpi = 170
        self.palette = list(CATEGORICAL)
        self.diverging = list(DIVERGING)
        self.score_cmap = "viridis"
        self.fontsize = 8.5
        self.verbosity = 1

    @property
    def figdir(self):
        return self._figdir

    @figdir.setter
    def figdir(self, v):
        self._figdir = Path(v)

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, v):
        self._data_dir = Path(v).expanduser()

    def set_figure_params(self, fontsize=None, dpi=None, file_format=None, figdir=None, palette=None,
                          score_cmap=None, facecolor=SURFACE):
        """Change figure defaults for every `al.pl` function.

        fontsize : base font size; dpi : saved-figure resolution; file_format : "png", "pdf" or a tuple;
        figdir : folder for `save=True`; palette : list of colours for categorical identity (cell lines, drugs);
        score_cmap : colormap for match scores (default "viridis"); facecolor : figure/axes background.
        """
        import matplotlib.pyplot as plt
        if fontsize is not None:
            self.fontsize = fontsize
        if dpi is not None:
            self.dpi = dpi
        if file_format is not None:
            self.file_format = (file_format,) if isinstance(file_format, str) else tuple(file_format)
        if figdir is not None:
            self.figdir = Path(figdir)
        if palette is not None:
            self.palette = list(palette)
        if score_cmap is not None:
            self.score_cmap = score_cmap
        plt.rcParams.update({"figure.facecolor": facecolor, "axes.facecolor": facecolor, "font.size": self.fontsize,
                             "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": facecolor})

    def __repr__(self):
        keys = ("data_dir", "figdir", "file_format", "dpi", "fontsize", "score_cmap")
        return "anchorline settings\n" + "\n".join(f"  {k}: {getattr(self, k)}" for k in keys)


settings = _Settings()
