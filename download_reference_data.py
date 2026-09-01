#!/usr/bin/env python3
"""Download the small set of SwissTopo reference layers required by the GUI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "reference_data"
STAC_ROOT = "https://data.geo.admin.ch/api/stac/v1/collections"
USER_AGENT = "SwissTopo-STL-Converter"
SHAPEFILE_SUFFIXES = {".shp", ".shx", ".dbf", ".prj", ".cpg"}


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _latest_shapefile_zip(collection: str) -> str:
    url = f"{STAC_ROOT}/{collection}/items?limit=100"
    features: list[dict] = []
    while url:
        payload = _fetch_json(url)
        features.extend(feature for feature in payload.get("features", []) if isinstance(feature, dict))
        next_url = next(
            (link.get("href") for link in payload.get("links", []) if link.get("rel") == "next" and link.get("href")),
            None,
        )
        url = urljoin(url, next_url) if next_url else ""

    if not features:
        raise RuntimeError(f"SwissTopo STAC returned no items for {collection}.")
    latest = max(
        features,
        key=lambda feature: str((feature.get("properties") or {}).get("datetime") or feature.get("id") or ""),
    )
    for asset in (latest.get("assets") or {}).values():
        href = str(asset.get("href") or "")
        if href.lower().endswith(".shp.zip"):
            return href
    raise RuntimeError(f"No shapefile ZIP was found in the latest {collection} item.")


def _download(url: str, destination: Path) -> None:
    print(f"Downloading: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _is_required_member(dataset: str, member: Path) -> bool:
    if member.suffix.lower() not in SHAPEFILE_SUFFIXES:
        return False
    name = member.name.upper()
    parent = member.parent.name.upper()
    if dataset == "boundaries":
        return name.startswith(("LANDESGRENZE.", "KANTONSGRENZE.", "BEZIRKSGRENZE."))
    if parent == "TLM_GEWAESSER":
        return name.startswith(("SWISSTLM3D_TLM_STEHENDES_GEWAESSER.", "SWISSTLM3D_TLM_FLIESSGEWAESSER."))
    if parent == "TLM_STRASSEN":
        return name.startswith("SWISSTLM3D_TLM_STRASSE.")
    if parent == "TLM_OEV":
        return name.startswith("SWISSTLM3D_TLM_EISENBAHN.")
    return False


def _extract_required(zip_path: Path, dataset: str) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if info.is_dir() or member.is_absolute() or ".." in member.parts:
                continue
            if not _is_required_member(dataset, member):
                continue
            destination = REFERENCE_DIR / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    return extracted


def _has_reference_data() -> bool:
    required = ("LANDESGRENZE.shp", "KANTONSGRENZE.shp", "BEZIRKSGRENZE.shp", "swissTLM3D_TLM_STEHENDES_GEWAESSER.shp", "swissTLM3D_TLM_FLIESSGEWAESSER.shp")
    return all(any(REFERENCE_DIR.rglob(name)) for name in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prune required SwissTopo reference layers.")
    parser.add_argument("--force", action="store_true", help="Download the current packages even when reference_data already exists.")
    args = parser.parse_args()

    if _has_reference_data() and not args.force:
        print("Reference data already exists. Use --force to refresh it from SwissTopo.")
        return

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    downloads = (("boundaries", "ch.swisstopo.swissboundaries3d"), ("tlm", "ch.swisstopo.swisstlm3d"))
    temporary_dir = ROOT / "work" / "reference_downloads"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    try:
        for label, collection in downloads:
            url = _latest_shapefile_zip(collection)
            archive = temporary_dir / f"{label}.shp.zip"
            _download(url, archive)
            extracted = _extract_required(archive, label)
            print(f"Extracted {extracted} required {label} file(s).")
            archive.unlink(missing_ok=True)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    subprocess.run([sys.executable, str(ROOT / "tools" / "prune_geometry_data.py")], check=True)
    print(f"Reference layers are ready in: {REFERENCE_DIR}")


if __name__ == "__main__":
    main()
