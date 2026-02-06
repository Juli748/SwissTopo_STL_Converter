import argparse
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request


def iter_urls_from_csv(csv_path):
    with csv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Allow simple CSVs where the URL is the first column.
            if "," in line:
                line = line.split(",", 1)[0].strip()
            if line.startswith("http://") or line.startswith("https://"):
                yield line


def download_file(url, destination_path):
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


def unzip_file(zip_path, extract_dir):
    print(f"Unzip: {zip_path.name} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def clear_directory(directory):
    for path in directory.iterdir():
        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
        except OSError:
            print(f"Failed to remove: {path.name}")


def _iter_csv_paths(repo_root: Path, csv_path: str | None) -> list[Path]:
    if csv_path:
        selected = Path(csv_path)
        if not selected.exists():
            print(f"CSV not found: {selected}")
            return []
        return [selected]

    data_dir = repo_root / "data"
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {data_dir}")
        return []
    return csv_files


def _maybe_clear_inputs(xyz_dir: Path, tif_dir: Path, clean_xyz: bool) -> None:
    existing_xyz_files = [p for p in xyz_dir.iterdir() if p.is_file()]
    existing_tif_files = [p for p in tif_dir.iterdir() if p.is_file()]
    if not existing_xyz_files and not existing_tif_files:
        return

    if clean_xyz:
        for file_path in existing_xyz_files + existing_tif_files:
            try:
                file_path.unlink()
            except OSError:
                print(f"Failed to delete: {file_path.name}")
        return

    if not sys.stdin.isatty():
        print("Existing XYZ/TIF files found in ./data/terrain. Keeping them (non-interactive).")
        return

    response = input("Existing files found in ./data/terrain/xyz or ./data/terrain/tif. Delete them? [y/N]: ").strip().lower()
    if response in {"y", "yes"}:
        for file_path in existing_xyz_files + existing_tif_files:
            try:
                file_path.unlink()
            except OSError:
                print(f"Failed to delete: {file_path.name}")


def _download_and_unzip(url, filename, zips_temp_dir, extract_dir):
    zip_path = zips_temp_dir / filename
    download_file(url, zip_path)
    target_dir = extract_dir / Path(filename).stem
    target_dir.mkdir(parents=True, exist_ok=True)
    unzip_file(zip_path, target_dir)
    return filename


def _download_and_unzip_flat(url, filename, zips_temp_dir, extract_dir):
    zip_path = zips_temp_dir / filename
    download_file(url, zip_path)
    unzip_file(zip_path, extract_dir)
    return filename


def _has_xyz_for_zip(xyz_dir: Path, filename: str) -> bool:
    stem = Path(filename).stem
    if (xyz_dir / f"{stem}.xyz").exists():
        return True
    matches = list(xyz_dir.glob(f"{stem}*.xyz"))
    return bool(matches)


def _classify_url(url: str) -> tuple[str, str]:
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        return "unknown", filename
    lower = filename.lower()
    if lower.endswith(".gdb.zip") or lower.endswith(".gdb"):
        return "building_gdb", filename
    if "swissbuildings3d" in lower:
        return "building_gdb", filename
    if lower.endswith(".zip"):
        return "zip", filename
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        return "tif", filename
    if lower.endswith(".xyz"):
        return "xyz", filename
    return "unknown", filename


def main():
    parser = argparse.ArgumentParser(description="Download SwissTopo terrain tiles or building GDBs from CSV URLs.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="CSV file with download URLs (if omitted, uses all CSV files in ./data).",
    )
    parser.add_argument(
        "--clean-xyz",
        action="store_true",
        help="Delete existing files in ./data/terrain/xyz and ./data/terrain/tif before downloading.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = _iter_csv_paths(repo_root, args.csv_path)
    if not csv_files:
        return 1

    terrain_dir = data_dir / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    terrain_extract_dir = terrain_dir / "_extract_temp"
    terrain_extract_dir.mkdir(parents=True, exist_ok=True)
    terrain_zips_temp_dir = terrain_dir / "_zips_temp"
    terrain_zips_temp_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(terrain_extract_dir)
    clear_directory(terrain_zips_temp_dir)
    xyz_dir = terrain_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    tif_dir = terrain_dir / "tif"
    tif_dir.mkdir(parents=True, exist_ok=True)
    _maybe_clear_inputs(xyz_dir, tif_dir, args.clean_xyz)

    buildings_dir = data_dir / "buildings"
    buildings_dir.mkdir(parents=True, exist_ok=True)
    buildings_zips_temp_dir = buildings_dir / "_zips_temp"
    buildings_zips_temp_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(buildings_zips_temp_dir)

    urls = []
    for csv_path in csv_files:
        urls.extend(iter_urls_from_csv(csv_path))

    items = []
    for url in urls:
        kind, filename = _classify_url(url)
        if not filename:
            print(f"Skip (bad URL): {url}")
            continue
        already = False
        if kind == "zip":
            already = _has_xyz_for_zip(xyz_dir, filename)
        elif kind == "tif":
            already = (tif_dir / filename).exists()
        elif kind == "xyz":
            already = (xyz_dir / filename).exists()
        elif kind == "building_gdb":
            if filename.lower().endswith(".gdb"):
                target_name = filename
            else:
                target_name = Path(filename).stem
            already = (buildings_dir / target_name).exists()
        else:
            print(f"Skip (unsupported URL): {url}")
            continue
        items.append((url, filename, kind, already))

    if not items:
        print("No URLs found in CSV files.")
        return 1

    total = len(items)
    workers = max(1, int(args.workers))
    completed = 0
    tasks = []
    for url, filename, kind, already in items:
        if already:
            completed += 1
            print(f"Skip (already downloaded): {filename}")
            print(f"[PROGRESS] {completed}/{total} {filename}")
            continue
        tasks.append((url, filename, kind))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_download_and_unzip, url, filename, terrain_zips_temp_dir, terrain_extract_dir): (url, filename)
            for url, filename, kind in tasks
            if kind == "zip"
        }
        for url, filename, kind in tasks:
            if kind == "tif":
                future_map[executor.submit(download_file, url, tif_dir / filename)] = (url, filename)
            elif kind == "xyz":
                future_map[executor.submit(download_file, url, xyz_dir / filename)] = (url, filename)
            elif kind == "building_gdb":
                future_map[executor.submit(_download_and_unzip_flat, url, filename, buildings_zips_temp_dir, buildings_dir)] = (url, filename)
        for future in as_completed(future_map):
            url, filename = future_map[future]
            completed += 1
            try:
                future.result()
            except Exception as exc:
                print(f"Failed: {url} ({exc})")
            print(f"[PROGRESS] {completed}/{total} {filename}")
    # Best-effort cleanup of temp zip folder.
    for zip_path in terrain_zips_temp_dir.glob("*"):
        try:
            zip_path.unlink()
        except OSError:
            pass
    try:
        terrain_zips_temp_dir.rmdir()
    except OSError:
        pass
    for zip_path in buildings_zips_temp_dir.glob("*"):
        try:
            zip_path.unlink()
        except OSError:
            pass
    try:
        buildings_zips_temp_dir.rmdir()
    except OSError:
        pass

    copied = 0
    for file_path in terrain_extract_dir.rglob("*.xyz"):
        if file_path.is_file():
            shutil.copy2(file_path, xyz_dir / file_path.name)
            copied += 1
    if copied:
        print(f"Copied {copied} XYZ files to {xyz_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
