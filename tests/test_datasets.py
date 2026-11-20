import anchorline as al


def test_manifest_lists_atlas_and_demo():
    m = al.datasets.manifest()
    groups = {v["group"] for v in m["files"].values()}
    assert groups == {"atlas", "demo"}
    assert all(len(v["sha256"]) == 64 for v in m["files"].values())
    assert {"query_reference.h5ad", "AML_drug_atlas_bulk.h5ad", "drugs.tsv", "cell_lines.tsv"} <= set(m["files"])


def test_status_on_empty_dir(tmp_path):
    t = al.datasets.status(tmp_path)
    assert not t.present.any()
    assert not al.datasets.is_available("atlas", tmp_path)


def test_settings_paths():
    old = al.settings.figdir
    al.settings.figdir = "figs"
    assert al.settings.figdir.name == "figs"
    al.settings.figdir = old
