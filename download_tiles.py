import argparse
import os
import shutil
import sys
import zipfile
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


def _maybe_clear_xyz(xyz_dir: Path, clean_xyz: bool) -> None:
    existing_xyz_files = [p for p in xyz_dir.iterdir() if p.is_file()]
    if not existing_xyz_files:
        return

    if clean_xyz:
        for file_path in existing_xyz_files:
            try:
                file_path.unlink()
            except OSError:
                print(f"Failed to delete: {file_path.name}")
        return

    if not sys.stdin.isatty():
        print("Existing XYZ files found in ./data/xyz. Keeping them (non-interactive).")
        return

    response = input("Existing files found in ./data/xyz. Delete them? [y/N]: ").strip().lower()
    if response in {"y", "yes"}:
        for file_path in existing_xyz_files:
            try:
                file_path.unlink()
            except OSError:
                print(f"Failed to delete: {file_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Download SwissTopo tiles from CSV URLs.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="CSV file with download URLs (if omitted, uses all CSV files in ./data).",
    )
    parser.add_argument(
        "--clean-xyz",
        action="store_true",
        help="Delete existing files in ./data/xyz before downloading.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = _iter_csv_paths(repo_root, args.csv_path)
    if not csv_files:
        return 1

    extract_dir = data_dir / "_extract_temp"
    extract_dir.mkdir(parents=True, exist_ok=True)
    zips_temp_dir = data_dir / "_zips_temp"
    zips_temp_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(extract_dir)
    clear_directory(zips_temp_dir)
    xyz_dir = data_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    _maybe_clear_xyz(xyz_dir, args.clean_xyz)

    urls = []
    for csv_path in csv_files:
        urls.extend(iter_urls_from_csv(csv_path))

    if not urls:
        print("No URLs found in CSV files.")
        return 1

    for url in urls:
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            print(f"Skip (bad URL): {url}")
            continue
        zip_path = zips_temp_dir / filename
        try:
            download_file(url, zip_path)
            unzip_file(zip_path, extract_dir)
        except Exception as exc:
            print(f"Failed: {url} ({exc})")
    # Best-effort cleanup of temp zip folder.
    for zip_path in zips_temp_dir.glob("*"):
        try:
            zip_path.unlink()
        except OSError:
            pass
    try:
        zips_temp_dir.rmdir()
    except OSError:
        pass

    copied = 0
    for file_path in extract_dir.rglob("*.xyz"):
        if file_path.is_file():
            shutil.copy2(file_path, xyz_dir / file_path.name)
            copied += 1
    print(f"Copied {copied} XYZ files to {xyz_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
