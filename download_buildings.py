import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


STAC_ITEMS_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/"
    "ch.swisstopo.swissbuildings3d_3_0/items"
)
DEFAULT_OUTPUT_DIR = Path("work") / "buildings" / "auto"
SWISSALTI_KM_TILE_RE = re.compile(r"(?<!\d)(?P<east>\d{4})-(?P<north>\d{4})(?!\d)")
SWISSALTI_M_TILE_RE = re.compile(r"(?<!\d)(?P<east>\d{7})[-_](?P<north>\d{7})(?!\d)")
BUILDING_EDITION_RE = re.compile(r"(swissbuildings3d_3_0_)(?P<year>\d{4})(_)", re.IGNORECASE)


def iter_urls_from_csv(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                line = line.split(",", 1)[0].strip()
            if line.startswith("http://") or line.startswith("https://"):
                yield line


def _download_file(url: str, destination_path: Path) -> Path:
    if destination_path.exists() and destination_path.stat().st_size > 0:
        print(f"Skip (exists): {destination_path.name}")
        return destination_path

    print(f"Download: {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, destination_path.open("wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
    return destination_path


def _fetch_json(url: str) -> dict:
    import json

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _tile_bounds_from_url(url: str) -> tuple[float, float, float, float] | None:
    path = urlparse(url).path
    for match in SWISSALTI_KM_TILE_RE.finditer(path):
        east_raw = match.group("east")
        north_raw = match.group("north")
        east = int(east_raw)
        north = int(north_raw)
        if 2400 <= east <= 2900 and 1000 <= north <= 1400:
            return east * 1000.0, north * 1000.0, (east + 1) * 1000.0, (north + 1) * 1000.0
    for match in SWISSALTI_M_TILE_RE.finditer(path):
        east = int(match.group("east"))
        north = int(match.group("north"))
        if 2400000 <= east <= 2900000 and 1000000 <= north <= 1400000:
            return float(east), float(north), float(east + 1000), float(north + 1000)
    return None


def bounds_from_csv(csv_path: Path) -> tuple[float, float, float, float]:
    bounds = []
    for url in iter_urls_from_csv(csv_path):
        tile_bounds = _tile_bounds_from_url(url)
        if tile_bounds:
            bounds.append(tile_bounds)
    if not bounds:
        raise ValueError(
            "No SwissALTI LV95 tile coordinates found in the CSV URLs. "
            "Expected names like swissalti3d_2019_2683-1247_0.5_2056_5728.tif."
        )
    west = min(b[0] for b in bounds)
    south = min(b[1] for b in bounds)
    east = max(b[2] for b in bounds)
    north = max(b[3] for b in bounds)
    return west, south, east, north


def _lv95_to_wgs84(east: float, north: float) -> tuple[float, float]:
    y_aux = (east - 2600000.0) / 1000000.0
    x_aux = (north - 1200000.0) / 1000000.0
    lat = (
        16.9023892
        + 3.238272 * x_aux
        - 0.270978 * y_aux * y_aux
        - 0.002528 * x_aux * x_aux
        - 0.0447 * y_aux * y_aux * x_aux
        - 0.0140 * x_aux * x_aux * x_aux
    )
    lon = (
        2.6779094
        + 4.728982 * y_aux
        + 0.791484 * y_aux * x_aux
        + 0.1306 * y_aux * x_aux * x_aux
        - 0.0436 * y_aux * y_aux * y_aux
    )
    return lon * 100.0 / 36.0, lat * 100.0 / 36.0


def _lv95_bounds_to_wgs84(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    corners = [
        _lv95_to_wgs84(west, south),
        _lv95_to_wgs84(west, north),
        _lv95_to_wgs84(east, south),
        _lv95_to_wgs84(east, north),
    ]
    lons = [pt[0] for pt in corners]
    lats = [pt[1] for pt in corners]
    return min(lons), min(lats), max(lons), max(lats)


def _asset_score(asset_id: str, asset: dict) -> int:
    href = str(asset.get("href") or "")
    text = " ".join(str(asset.get(key) or "") for key in ("title", "description", "type", "roles"))
    lower = f"{asset_id} {href} {text}".lower()
    if not href:
        return -100
    score = 0
    if "citygml" in lower:
        score += 8
    if "gml" in lower:
        score += 5
    if href.lower().endswith(".zip"):
        score += 3
    if href.lower().endswith((".gml", ".xml")):
        score += 3
    if any(bad in lower for bad in ("filegdb", "fgdb", "dwg", "dxf", "3dtiles", "i3s")):
        score -= 10
    return score


def _candidate_asset_hrefs(feature: dict) -> list[str]:
    scored = []
    for asset_id, asset in (feature.get("assets") or {}).items():
        score = _asset_score(asset_id, asset)
        if score > 0:
            scored.append((score, str(asset.get("href"))))
    scored.sort(reverse=True)
    return [href for _score, href in scored]


def _building_package_key_and_year(href: str) -> tuple[str, int]:
    """Group yearly editions of the same CityGML coverage package."""
    match = BUILDING_EDITION_RE.search(href)
    if match is None:
        return href, 0
    key = BUILDING_EDITION_RE.sub(
        lambda edition: f"{edition.group(1).lower()}{{edition}}{edition.group(3)}",
        href,
    )
    return key, int(match.group("year"))


def query_building_assets(bounds_lv95: tuple[float, float, float, float], *, max_items: int = 0) -> list[str]:
    bbox = _lv95_bounds_to_wgs84(bounds_lv95)
    params = {"bbox": ",".join(f"{value:.8f}" for value in bbox), "limit": "100"}
    url = f"{STAC_ITEMS_URL}?{urlencode(params)}"
    latest_hrefs: dict[str, tuple[int, str]] = {}
    inspected_assets: list[str] = []

    while url:
        payload = _fetch_json(url)
        for feature in payload.get("features", []):
            for asset_id, asset in (feature.get("assets") or {}).items():
                inspected_assets.append(f"{asset_id}: {asset.get('href', '')}")
            for href in _candidate_asset_hrefs(feature):
                key, year = _building_package_key_and_year(href)
                existing = latest_hrefs.get(key)
                if existing is None or year > existing[0]:
                    latest_hrefs[key] = (year, href)
        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break
        url = next_url

    hrefs = [entry[1] for _key, entry in sorted(latest_hrefs.items())]
    if max_items:
        hrefs = hrefs[:max_items]
    if not hrefs and inspected_assets:
        print("No CityGML-like assets found. First available assets:")
        for line in inspected_assets[:20]:
            print(f"  {line}")
    return hrefs


def _safe_extract_citygml(zip_path: Path, target_dir: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for info in zip_ref.infolist():
            member_path = Path(info.filename)
            if info.is_dir() or member_path.is_absolute() or ".." in member_path.parts:
                continue
            if member_path.suffix.lower() not in {".gml", ".xml"}:
                continue
            target_path = target_dir / member_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(info, "r") as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


def _target_dir_for_href(output_dir: Path, href: str) -> Path:
    filename = os.path.basename(urlparse(href).path)
    stem = Path(filename).stem or "swissbuildings3d"
    return output_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _marker_path(target_dir: Path) -> Path:
    return target_dir / ".download_complete.json"


def _target_is_complete(target_dir: Path, href: str) -> bool:
    has_gml = any(target_dir.rglob("*.gml"))
    marker = _marker_path(target_dir)
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("href") == href and payload.get("citygml_files", 0) > 0 and has_gml:
            return True
    return has_gml


def _write_marker(target_dir: Path, href: str, extracted: int) -> None:
    marker = _marker_path(target_dir)
    marker.write_text(
        json.dumps({"href": href, "citygml_files": int(extracted)}, indent=2),
        encoding="utf-8",
    )


def _sync_output_dir(output_dir: Path, hrefs: list[str]) -> None:
    wanted_dirs = {_target_dir_for_href(output_dir, href).resolve() for href in hrefs}
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if not child.is_dir() or child.name == "_download_temp":
            continue
        try:
            resolved = child.resolve()
        except OSError:
            continue
        if resolved not in wanted_dirs:
            shutil.rmtree(child, ignore_errors=True)
            print(f"Removed stale building tile: {child.name}")


def download_and_extract_assets(hrefs: list[str], output_dir: Path, workers: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_download_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    total = len(hrefs)
    completed = 0
    extracted_total = 0

    def one(href: str) -> tuple[str, int]:
        filename = os.path.basename(urlparse(href).path)
        lower = filename.lower()
        target_dir = _target_dir_for_href(output_dir, href)
        if target_dir.exists() and _target_is_complete(target_dir, href):
            print(f"[BUILDINGS] Reusing cached CityGML: {filename}")
            return filename, 0
        if lower.endswith(".zip"):
            zip_path = temp_dir / filename
            print(f"[BUILDINGS] Downloading and extracting: {filename}")
            _download_file(href, zip_path)
            target_dir.mkdir(parents=True, exist_ok=True)
            extracted = _safe_extract_citygml(zip_path, target_dir)
            _write_marker(target_dir, href, extracted)
            try:
                zip_path.unlink()
            except OSError:
                pass
            return filename, extracted
        if lower.endswith((".gml", ".xml")):
            target_path = target_dir / filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[BUILDINGS] Downloading: {filename}")
            _download_file(href, target_path)
            _write_marker(target_dir, href, 1)
            return filename, 1
        return filename, 0

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_map = {executor.submit(one, href): href for href in hrefs}
        for future in as_completed(future_map):
            href = future_map[future]
            completed += 1
            try:
                filename, extracted = future.result()
                extracted_total += extracted
            except Exception as exc:
                filename = os.path.basename(urlparse(href).path) or href
                print(f"Failed: {href} ({exc})")
            print(f"[PROGRESS] {completed}/{total} {filename}")

    try:
        temp_dir.rmdir()
    except OSError:
        pass
    return extracted_total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download swissBUILDINGS3D 3.0 Beta CityGML for the region covered by a SwissALTI CSV."
    )
    parser.add_argument("--csv", dest="csv_path", required=True, help="SwissALTI CSV with download URLs.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Folder for extracted CityGML files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel building downloads (default: 4).")
    parser.add_argument("--max-items", type=int, default=0, help="Optional test limit for STAC items/assets.")
    parser.add_argument("--clean", action="store_true", help="Delete the output folder before downloading.")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Remove auto-downloaded building tiles from the output folder if they are not needed for this CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        bounds_lv95 = bounds_from_csv(csv_path)
    except ValueError as exc:
        print(exc)
        return 1

    print(
        "SwissALTI bounds LV95: "
        f"{bounds_lv95[0]:.0f}, {bounds_lv95[1]:.0f}, {bounds_lv95[2]:.0f}, {bounds_lv95[3]:.0f}"
    )
    hrefs = query_building_assets(bounds_lv95, max_items=max(0, int(args.max_items)))
    if not hrefs:
        print(
            "No swissBUILDINGS3D CityGML downloads were found for this area. "
            "CityGML is only available where SwissTopo publishes EGID-prepared beta data."
        )
        return 1

    print(f"Found {len(hrefs)} matching swissBUILDINGS3D asset(s).")
    if args.sync:
        _sync_output_dir(output_dir, hrefs)
    extracted = download_and_extract_assets(hrefs, output_dir, workers=max(1, int(args.workers)))
    if extracted <= 0 and not list(output_dir.rglob("*.gml")):
        print("Downloads completed, but no CityGML files were extracted.")
        return 1
    print(f"Buildings ready in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
