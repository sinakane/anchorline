import anchorline as al


def test_catalog_has_bundled_collections():
    cols = set(al.sig.collections().index)
    assert {"Umeda2024_pediatric_AML", "Zeng2022_AML_hierarchy", "stemness", "menin_targets"} <= cols
    assert len(al.sig.list("Umeda2024")) == 20


def test_get_exact_and_substring():
    s = al.sig.get("Umeda2024_KMT2Ar_high")
    assert s.up and s.down and s.doi == "10.1038/s41588-023-01640-3"
    assert al.sig.get("KMT2Ar_high").name == s.name


def test_every_bundled_signature_has_source_column():
    t = al.sig.list()
    assert (t.n_up > 0).all()
    assert t.source.str.len().gt(0).all()


def test_search_register_and_gmt_roundtrip(tmp_path):
    assert "HOXA_MEIS1_core" in set(al.sig.search("MEIS1").name)
    al.sig.register("t_reg", up=["A", "B"], down=["C"])
    assert al.sig.get("t_reg").down == ["C"]
    p = al.sig.export(tmp_path / "x.gmt", names=["t_reg", "HOXA_MEIS1_core"])
    back = al.sig.read_gmt(p)
    assert back["t_reg"].up == ["A", "B"] and back["t_reg"].down == ["C"]
