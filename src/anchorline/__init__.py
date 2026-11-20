"""anchorline - anchor a patient signature or sample to the closest AML cell lines, then rank the drugs
tested in them by how they move those cells (magnitude, signature reversal, differentiation).

    import anchorline as al
    atlas = al.load()                  # downloads the atlas the first time (al.datasets.download())
    mt = al.tl.match(atlas, al.sig.get("Umeda2024_KMT2Ar_high"))
    al.pl.match_ranking(mt); al.pl.pseudotime(mt)
    dr = al.tl.rank_drugs(atlas, mt, cell_lines=5)
    al.pl.drug_ranking(dr); al.pl.drug_shifts(dr); al.pl.state_map(dr)
    al.pl.class_trends(dr); al.pl.class_shift_map(dr)
    al.pl.gene_changes(dr, ["CLEC12A", "CD84"]); al.pl.top_gene_changes(dr); al.pl.signature_changes(dr)
"""
from . import datasets, io, pl, sig, tl
from ._core import Atlas, load
from ._settings import settings

__version__ = "0.1.0"
__all__ = ["Atlas", "load", "datasets", "io", "tl", "pl", "sig", "settings"]
