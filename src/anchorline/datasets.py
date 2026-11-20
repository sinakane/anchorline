"""Download the atlas and the demo data (one line).

    al.datasets.download()            # atlas (~560 MB) into al.settings.data_dir (default ~/.anchorline/data)
    al.datasets.download("demo")      # BeatAML demo data for the examples (~13 MB)
    al.datasets.download("all")
    al.datasets.status()              # what is present / missing / corrupted

Terminal:  anchorline-download [atlas|demo|all] [--dir PATH] [--force]

Files are public GitHub release assets of sinakane/anchorline (tag in resources/datasets.json), checked by
SHA-256. No login is needed; if $GITHUB_TOKEN / $GH_TOKEN or a `gh` login is present it is used (higher API rate
limit).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ._core import _resource
from ._settings import settings

GROUPS = ("atlas", "demo")


def manifest():
    return json.loads(_resource("datasets.json").read_text())


def _files(which):
    m = manifest()["files"]
    groups = GROUPS if which == "all" else (which,)
    if any(g not in GROUPS for g in groups):
        raise ValueError(f"which must be one of {GROUPS + ('all',)}")
    return {k: v for k, v in m.items() if v["group"] in groups}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def status(data_dir=None, check_hash=False):
    """Table of every downloadable file: present, size ok, (optionally) checksum ok."""
    import pandas as pd
    D = Path(data_dir or settings.data_dir)
    rows = []
    for name, meta in manifest()["files"].items():
        p = D / name
        ok_size = p.exists() and p.stat().st_size == meta["size"]
        rows.append(dict(file=name, group=meta["group"], size_mb=round(meta["size"] / 1e6, 1), present=p.exists(),
                         size_ok=ok_size, sha256_ok=(_sha256(p) == meta["sha256"]) if (check_hash and ok_size) else None))
    t = pd.DataFrame(rows).set_index("file")
    t.attrs["data_dir"] = str(D)
    return t


def is_available(which="atlas", data_dir=None):
    D = Path(data_dir or settings.data_dir)
    return all((D / n).exists() and (D / n).stat().st_size == m["size"] for n, m in _files(which).items())


def _token():
    for k in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    if shutil.which("gh"):
        envs = [dict(os.environ)]
        alt = Path.home() / ".gh"
        if "GH_CONFIG_DIR" not in os.environ and alt.is_dir():
            envs.append({**os.environ, "GH_CONFIG_DIR": str(alt)})
        for env in envs:
            try:
                t = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, env=env, timeout=15)
                if t.returncode == 0 and t.stdout.strip():
                    return t.stdout.strip()
            except Exception:
                pass
    return None


def _request(url, token, accept="application/vnd.github+json"):
    h = {"Accept": accept, "User-Agent": "anchorline"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=h)


def _release_assets(token):
    m = manifest()
    url = f"https://api.github.com/repos/{m['repo']}/releases/tags/{m['tag']}"
    try:
        with urllib.request.urlopen(_request(url, token), timeout=60) as r:
            rel = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"release '{m['tag']}' of {m['repo']} not found. If the repository is private, give a GitHub token: "
                "export GITHUB_TOKEN=... (or log in with `gh auth login`).") from None
        raise
    return {a["name"]: a for a in rel.get("assets", [])}


def _download(asset, dest, size, token):
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = _request(asset["url"], token, accept="application/octet-stream")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        done, step = 0, max(size // 40, 1 << 20)
        nxt = step
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if done >= nxt or done == size:
                pct = 100 * done / size if size else 100
                bar = "#" * int(pct / 5)
                sys.stdout.write(f"\r  {dest.name:32s} [{bar:<20s}] {pct:5.1f}%  {done / 1e6:7.1f} MB")
                sys.stdout.flush()
                nxt += step
    sys.stdout.write("\n")
    tmp.replace(dest)


def download(which="atlas", data_dir=None, force=False, verify=True):
    """Download `which` ("atlas", "demo" or "all") into data_dir (default al.settings.data_dir).
    Existing files with the right size are kept unless force=True. Returns the folder."""
    D = Path(data_dir or settings.data_dir).expanduser()
    D.mkdir(parents=True, exist_ok=True)
    files = _files(which)
    todo = {n: m for n, m in files.items() if force or not ((D / n).exists() and (D / n).stat().st_size == m["size"])}
    if not todo:
        print(f"{which}: all {len(files)} files already in {D}")
        return D
    token = _token()
    assets = _release_assets(token)
    print(f"downloading {len(todo)} file(s), {sum(m['size'] for m in todo.values()) / 1e6:.0f} MB -> {D}")
    for n, m in todo.items():
        if n not in assets:
            raise RuntimeError(f"{n} is not attached to release {manifest()['tag']}")
        _download(assets[n], D / n, m["size"], token)
        if verify and _sha256(D / n) != m["sha256"]:
            (D / n).unlink()
            raise RuntimeError(f"checksum mismatch for {n} - deleted, please retry")
    print("done" + (" (checksums verified)" if verify else ""))
    return D


def demo_path(name="beataml_demo.h5ad", data_dir=None, download_if_missing=True):
    """Path of a demo file (downloads the demo set the first time)."""
    D = Path(data_dir or settings.data_dir)
    if not (D / name).exists() and download_if_missing:
        download("demo", D)
    return D / name


def beataml_demo(data_dir=None):
    """AnnData: 45 BeatAML diagnosis bone-marrow samples (15 KMT2A-r, 15 NPM1-mut, 15 CBFB-MYH11) x genes,
    log2 normalised expression. BeatAML 2.0: Bottomly et al. 2022 Cancer Cell 40:850-864,
    doi:10.1016/j.ccell.2022.07.002."""
    import anndata as ad
    return ad.read_h5ad(demo_path("beataml_demo.h5ad", data_dir))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="anchorline-download", description="Download the anchorline atlas / demo data.")
    p.add_argument("which", nargs="?", default="atlas", choices=GROUPS + ("all",))
    p.add_argument("--dir", default=None, help=f"target folder (default: {settings.data_dir})")
    p.add_argument("--force", action="store_true", help="re-download files that are already present")
    a = p.parse_args(argv)
    download(a.which, a.dir, a.force)


if __name__ == "__main__":
    main()
