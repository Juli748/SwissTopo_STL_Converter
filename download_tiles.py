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


def main():
    repo_root = Path(__file__).resolve().parent
    downloads_dir = repo_root / "Download"
    if not downloads_dir.exists():
        print(f"Missing downloads folder: {downloads_dir}")
        return 1

    csv_files = sorted(downloads_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {downloads_dir}")
        return 1

    extract_dir = downloads_dir / "terrain_temp"
    extract_dir.mkdir(parents=True, exist_ok=True)
    zips_temp_dir = downloads_dir / "zips_temp"
    zips_temp_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(extract_dir)
    clear_directory(zips_temp_dir)
    terrain_dir = repo_root / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    existing_terrain_files = [p for p in terrain_dir.iterdir() if p.is_file()]
    if existing_terrain_files:
        response = input(
            "Existing files found in ./terrain. Delete them before copying? [y/N]: "
        ).strip().lower()
        if response in {"y", "yes"}:
            for file_path in existing_terrain_files:
                try:
                    file_path.unlink()
                except OSError:
                    print(f"Failed to delete: {file_path.name}")

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

    response = input(
        "Copy unzipped files into ./terrain for processing? [y/N]: "
    ).strip().lower()
    if response in {"y", "yes"}:
        copied = 0
        for file_path in extract_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, terrain_dir / file_path.name)
                copied += 1
        print(f"Copied {copied} files to {terrain_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
