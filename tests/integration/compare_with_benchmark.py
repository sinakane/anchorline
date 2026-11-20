"""Regression check: anchorline must reproduce the benchmark's query/atlas_query.py exactly.
Only runs on the machine that holds the benchmark (query/atlas_query.py). ~3 GB RAM.
Run: python tests/integration/compare_with_benchmark.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path.home() / "Desktop/AML_drug_perturbation_benchmark/01_data/cellline_drug_atlas/query"
sys.path.insert(0, str(BENCH))
import atlas_query as old  # noqa: E402

import anchorline as al  # noqa: E402

A_old = old.Atlas()
A_new = al.load()
checks = []


def compare(label, m_old, mt):
    r_old = A_old.rank_drugs(m_old)
    dr = al.tl.rank_drugs(A_new, mt, cell_lines=5)
    c1 = list(m_old["cell_lines"].index) == list(mt.table.index[:5])
    c2 = np.allclose(m_old["cell_lines"].score.values, mt.table.score.values[:5])
    c3 = np.allclose(m_old["cell_lines"].weight.values, dr.cell_lines.values)
    Do, Dn = r_old["drugs"], dr.drugs
    c4 = list(Do.index) == list(Dn.index)
    c5 = np.allclose(Do.score.values, Dn.loc[Do.index].score.values, equal_nan=True)
    cols = ["composite", "magnitude_z", "diff_shift", "n_surface_up", "surface_net_index", "pc_net_index"]
    c6 = all(np.allclose(Do[c].astype(float).values, Dn.loc[Do.index, c].astype(float).values, equal_nan=True) for c in cols)
    c7 = (Do.tier == Dn.loc[Do.index].tier).all() and (Do.surface_up_genes == Dn.loc[Do.index].surface_up_genes).all()
    ok = all([c1, c2, c3, c4, c5, c6, c7])
    checks.append(ok)
    print(f"{label:28s} cell lines {c1} scores {c2} weights {c3} | drug order {c4} scores {c5} columns {c6} "
          f"tier/surface {c7}  -> {'PASS' if ok else 'FAIL'}   top: {', '.join(Dn.drug_name[:3])}")
    return r_old, dr


HOX = ["HOXA5", "HOXA7", "HOXA9", "HOXA10", "MEIS1", "PBX3", "MEF2C"]
compare("signature HOX core", A_old.match_signature(HOX, models="cell_lines"), al.tl.match(A_new, HOX))
s = al.sig.get("Umeda2024_KMT2Ar_high")
compare("signature Umeda KMT2Ar_high", A_old.match_signature(s.up, s.down, models="cell_lines"), al.tl.match(A_new, s))

BA = Path.home() / "Desktop/Rene/programs/metaprograms/external/beataml"
cl = pd.read_excel(BA / "beataml_wv1to4_clinical.xlsx")
cl = cl[cl.dbgap_rnaseq_sample.notna() & cl.dxAtSpecimenAcquisition.str.startswith("ACUTE MYELOID", na=False)]
expr = pd.read_csv(BA / "beataml_waves1to4_norm_exp_dbgap.txt", sep="\t")
expr = expr.drop(columns=["stable_id", "description", "biotype"]).groupby("display_label").mean()
aml = [x for x in cl.dbgap_rnaseq_sample if x in expr.columns]
sid = "BA2031R"
m_old = A_old.match_sample(expr[sid], background=expr[aml].T, models="cell_lines")
mt = al.tl.match(A_new, expr[aml], sample=sid)          # table input: other columns become the background
r_old, dr = compare(f"sample BeatAML {sid} (table)", m_old, mt)
print("\nALL PASS" if all(checks) else "\nSOME FAILED")
