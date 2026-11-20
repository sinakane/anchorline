"""Smoke test: every query type and every plot (needs the atlas + demo data; downloads them if missing).
Run: python tests/integration/run_smoke.py [figdir]"""
import sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import anndata as ad
import pandas as pd
import anchorline as al

FIG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures_smoke")
al.settings.figdir = FIG
atlas = al.load()
D = al.datasets.demo_path().parent
t0 = time.time()
# inputs
mt_sig = al.tl.match(atlas, "Umeda2024_KMT2Ar_high")
mt_list = al.tl.match(atlas, ["HOXA9", "MEIS1", "PBX3"], name="HOX3")
sig_csv = FIG / "my_sig.csv"; FIG.mkdir(exist_ok=True)
pd.DataFrame({"gene": ["HOXA9", "MEIS1", "CD34", "MPO"], "direction": ["up", "up", "up", "down"]}).to_csv(sig_csv, index=False)
mt_csv_sig = al.tl.match(atlas, sig_csv)
mt_csv = al.tl.match(atlas, D / "beataml_demo_expression.csv", sample=pd.read_csv(D / "beataml_demo_samples.csv").iloc[0, 0])
A = al.datasets.beataml_demo()
mt_ad1 = al.tl.match(atlas, A, sample=A.obs_names[0])
mt_ad2 = al.tl.match(atlas, A, groupby="group", group="KMT2A-r")
for m in (mt_sig, mt_list, mt_csv_sig, mt_csv, mt_ad1, mt_ad2):
    print(repr(m.query), "->", ", ".join(m.table.index[:4]))
print("matching s", round(time.time() - t0, 1))
# drugs + every plot (save, no show)
dr = al.tl.rank_drugs(atlas, mt_sig, cell_lines=5)
dr2 = al.tl.rank_drugs(atlas, mt_ad2, cell_lines=["MOLM-13", "MV4-11", "NOMO-1"])
print(dr); print(dr2)
kw = dict(save=True, show=False)
al.pl.match_ranking(mt_sig, **kw)
al.pl.match_ranking(mt_ad2, by="study", n_top=20, **kw)
al.pl.pseudotime(mt_ad2, n_top=6, **kw)
al.pl.pseudotime(mt_sig, **kw)
al.pl.drug_ranking(dr, **kw)
al.pl.drug_ranking(dr, color_by="drug_class", n_top=20, save=FIG / "dr_byclass", show=False)
al.pl.drug_shifts(dr, **kw)
al.pl.state_map(dr, drugs=["ziftomenib", "pinometostat", "MI-3454", "nonexistent"], **kw)
al.pl.class_trends(dr, **kw)
al.pl.class_shift_map(dr, classes=["menin-KMT2A", "DOT1L", "BET", "LSD1/KDM1A"], **kw)
al.pl.gene_changes(dr, ["CLEC12A", "CD84"], **kw)
al.pl.gene_changes(dr, ["CLEC12A", "CD84", "CD33"], classes=["menin-KMT2A", "DOT1L", "BET"],
                   save=FIG / "genes_classes", show=False)
al.pl.top_gene_changes(dr, **kw)
al.pl.signature_changes(dr, **kw)
al.pl.summary(mt_ad2, dr2, **kw)
al.pl.group_summary({"KMT2Ar_high": (mt_sig, dr), "BeatAML KMT2A-r pseudobulk": (mt_ad2, dr2)}, **kw)
print("all plots OK", round(time.time() - t0, 1), "s")
