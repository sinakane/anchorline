"""Built-in signature library.

    al.sig.list()                       catalog table (name, collection, n_up, n_down, source, doi)
    al.sig.list("Umeda2024")            filter by collection or name substring
    s = al.sig.get("Umeda2024_KMT2Ar_high")   -> Signature(up=[...], down=[...])
    al.sig.search("MEIS1")              which signatures contain a gene
    al.sig.register("my_set", up=[...], down=[...], description="...")   add your own for this session
    al.sig.read_gmt(path)               load a .gmt file as {name: Signature}

Bundled collections: Umeda2024_pediatric_AML (20 groups, Umeda et al. 2024 Nat Genet), Zeng2022_AML_hierarchy
(Zeng et al. 2022 Nat Med), stemness (LSC104 Ng et al. 2016 Nature; MLL-LSC Somervaille et al. 2009 Cell Stem
Cell; HSC_Cycle_Primed, source not verified), menin_targets (HOXA/MEIS1 core, Issa et al. 2023 Nature).
"""
from __future__ import annotations

import builtins
from dataclasses import dataclass, field

import pandas as pd

from ._core import _resource


@dataclass
class Signature:
    """A gene signature: genes higher (`up`) and optionally lower (`down`) in the state of interest."""
    name: str
    up: list
    down: list = field(default_factory=list)
    collection: str = "user"
    description: str = ""
    source: str = ""
    doi: str = ""

    def __repr__(self):
        src = f", {self.source}" if self.source else ""
        return f"Signature('{self.name}': {len(self.up)} up, {len(self.down)} down{src})"

    def to_frame(self):
        return pd.DataFrame({"gene": self.up + self.down, "direction": ["up"] * len(self.up) + ["down"] * len(self.down)})


_CATALOG = None
_GENES = None
_USER = {}


def _load():
    global _CATALOG, _GENES
    if _CATALOG is None:
        with _resource("signatures/catalog.tsv").open() as f:
            _CATALOG = pd.read_csv(f, sep="\t", dtype=str).fillna("")
        with _resource("signatures/signatures.tsv").open() as f:
            _GENES = pd.read_csv(f, sep="\t", dtype={"name": str, "gene": str, "direction": str})
    return _CATALOG, _GENES


def list(pattern=None):  # noqa: A001 - mirrors the scanpy-style API
    """Catalog of available signatures (bundled + registered). `pattern` filters name/collection (case-insensitive)."""
    cat, _ = _load()
    t = cat.copy()
    if _USER:
        t = pd.concat([t, pd.DataFrame([dict(name=s.name, collection=s.collection, description=s.description,
                                             source=s.source, doi=s.doi) for s in _USER.values()])], ignore_index=True)
    t["n_up"] = [len(get(n).up) for n in t.name]
    t["n_down"] = [len(get(n).down) for n in t.name]
    if pattern:
        p = pattern.lower()
        t = t[t.name.str.lower().str.contains(p, regex=False) | t.collection.str.lower().str.contains(p, regex=False)]
    return t[["name", "collection", "n_up", "n_down", "description", "source", "doi"]].set_index("name")


def collections():
    """Collections and how many signatures each holds."""
    return list().groupby("collection").size().rename("n_signatures")


def get(name):
    """Return a Signature by name (exact, else a unique case-insensitive substring match)."""
    if name in _USER:
        return _USER[name]
    cat, genes = _load()
    names = builtins.list(cat.name) + builtins.list(_USER)
    if name not in names:
        hits = [n for n in names if name.lower() in n.lower()]
        if len(hits) != 1:
            raise KeyError(f"'{name}' not found" + (f"; did you mean one of {hits[:10]}?" if hits else
                                                    " - see al.sig.list()"))
        name = hits[0]
        if name in _USER:
            return _USER[name]
    r = cat.set_index("name").loc[name]
    g = genes[genes.name == name]
    return Signature(name, builtins.list(g.gene[g.direction == "up"]), builtins.list(g.gene[g.direction == "down"]),
                     r.collection, r.description, r.source, r.doi)


def search(gene):
    """Signatures that contain `gene`, with its direction."""
    _, genes = _load()
    t = genes[genes.gene == gene][["name", "direction"]]
    for s in _USER.values():
        if gene in s.up or gene in s.down:
            t = pd.concat([t, pd.DataFrame([dict(name=s.name, direction="up" if gene in s.up else "down")])])
    return t.reset_index(drop=True)


def register(name, up, down=(), description="", source="", doi="", collection="user"):
    """Add a signature for this session (it then works everywhere a built-in name does)."""
    _USER[name] = Signature(name, builtins.list(up), builtins.list(down), collection, description, source, doi)
    return _USER[name]


def export(path, names=None, fmt=None):
    """Write signatures to a file: .gmt (X_UP / X_DOWN lines) or .tsv/.csv (name, gene, direction).
    names: list of signature names (default: all bundled + registered)."""
    from pathlib import Path
    p = Path(path)
    names = names or builtins.list(list().index)
    fmt = fmt or p.suffix.lstrip(".") or "tsv"
    sigs = [get(n) for n in names]
    if fmt == "gmt":
        with open(p, "w") as f:
            for s in sigs:
                f.write(f"{s.name}_UP\t{s.source or s.collection}\t" + "\t".join(s.up) + "\n")
                if s.down:
                    f.write(f"{s.name}_DOWN\t{s.source or s.collection}\t" + "\t".join(s.down) + "\n")
    else:
        t = pd.concat([s.to_frame().assign(name=s.name) for s in sigs])[["name", "gene", "direction"]]
        t.to_csv(p, sep="," if fmt == "csv" else "\t", index=False)
    return p


def read_gmt(path, register_all=False, collection="gmt"):
    """Read a .gmt file -> {name: Signature} (up genes only). Pairs named X_UP / X_DOWN (any case) are merged."""
    raw = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) > 2:
            raw[f[0]] = [x for x in f[2:] if x]
    out, used = {}, set()
    for k in raw:
        for su, sd in (("_UP", "_DOWN"), ("_up", "_down"), ("_Up", "_Down")):
            if k.endswith(su) and k[: -len(su)] + sd in raw:
                base = k[: -len(su)]
                out[base] = Signature(base, raw[k], raw[base + sd], collection)
                used |= {k, base + sd}
    for k, v in raw.items():
        if k not in used:
            out[k] = Signature(k, v, [], collection)
    if register_all:
        for s in out.values():
            _USER[s.name] = s
    return out
