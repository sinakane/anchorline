"""Build small BeatAML demo inputs for the example notebooks (written to data/, which git ignores).

Source: BeatAML 2.0 waves 1-4 (Bottomly et al. 2022 Cancer Cell 40:850-864, doi:10.1016/j.ccell.2022.07.002),
normalised expression + clinical table, downloaded from https://biodev.github.io/BeatAML2/.
Picks diagnosis-stage bone-marrow AML samples: 15 KMT2A-rearranged, 15 NPM1-mutant (no fusion), 15 CBFB-MYH11.

Outputs
  data/beataml_demo.h5ad              AnnData, 45 samples x genes, X = log2 normalised expression, obs = genotype
  data/beataml_demo_expression.csv    the same as a genes x samples table
  data/beataml_demo_samples.csv       sample annotation

Usage: python scripts/make_beataml_demo.py [path/to/beataml folder]
"""
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

BA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop/Rene/programs/metaprograms/external/beataml"
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

cl = pd.read_excel(BA / "beataml_wv1to4_clinical.xlsx")
cl = cl[cl.dbgap_rnaseq_sample.notna() & cl.dxAtSpecimenAcquisition.str.startswith("ACUTE MYELOID", na=False)]
cl = cl[(cl.diseaseStageAtSpecimenCollection == "Initial Diagnosis") & (cl.specimenType == "Bone Marrow Aspirate")]
cl = cl.drop_duplicates("dbgap_rnaseq_sample").set_index("dbgap_rnaseq_sample")
fus = cl.consensusAMLFusions.astype(str)
groups = {"KMT2A-r": cl[fus.str.contains("KMT2A")],
          "NPM1-mut": cl[(cl.NPM1 == "positive") & cl.consensusAMLFusions.isna()],
          "CBFB-MYH11": cl[fus == "CBFB-MYH11"]}
pick = pd.concat([g.sort_index().head(15).assign(group=k) for k, g in groups.items()])
expr = pd.read_csv(BA / "beataml_waves1to4_norm_exp_dbgap.txt", sep="\t")
expr = expr.drop(columns=["stable_id", "description", "biotype"]).groupby("display_label").mean()
pick = pick[pick.index.isin(expr.columns)]
E = expr[pick.index]
obs = pick[["group", "consensusAMLFusions", "NPM1", "FLT3-ITD", "ELN2017", "ageAtDiagnosis"]].astype(str)
obs.columns = ["group", "fusion", "NPM1", "FLT3_ITD", "ELN2017", "age"]
A = ad.AnnData(E.T.values.astype("float32"), obs=obs, var=pd.DataFrame(index=E.index))
A.uns["source"] = "BeatAML 2.0 (Bottomly et al. 2022 Cancer Cell, doi:10.1016/j.ccell.2022.07.002); log2 normalised expression"
A.write_h5ad(OUT / "beataml_demo.h5ad")
E.round(4).to_csv(OUT / "beataml_demo_expression.csv")
obs.to_csv(OUT / "beataml_demo_samples.csv")
print(A)
print(obs.group.value_counts().to_dict())
