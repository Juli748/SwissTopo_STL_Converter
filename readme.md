# SwissTopo STL Converter (GUI-first)

This project is built around the GUI. Use it to download SwissTopo XYZ tiles, convert them into STL tiles, and merge them into a single printable STL with optional base.

Run the GUI:

```bash
python gui.py
```

---

## GUI Workflow (recommended)

The GUI is organized into three steps that match the full workflow:

1) **Download XYZ tiles**
2) **Convert XYZ to STL tiles**
3) **Merge tiles into final STL**

You can run just one step or all three in sequence.

---

## Step 1: Download XYZ tiles

This step downloads ZIP tiles from a CSV of URLs and extracts XYZ files into `data/xyz`.

**How to use it**
- Click **Browse** and select your CSV with download URLs.
- (Optional) Click **Copy to data/** to keep a copy inside the project.
- Click **Run Download**.
- If existing XYZ tiles are found, the GUI asks whether to delete them before downloading.

**CSV expectations**
- One URL per line (simple CSVs are fine; only the first column is used).
- Comment lines starting with `#` are ignored.

**Typical SwissTopo sources**
- swissALTI3D (DTM)
- swissSURFACE3D Raster (DSM)

---

## Step 2: Convert XYZ to STL tiles

This step converts every `.xyz` file in `data/xyz` into one STL tile per file in `output/tiles`.

**Options**

**Detail mode**
- **Auto (size in mm)**: You choose a target print size; the tool calculates a downsample step.
  - **Target smallest edge (mm)**: The shorter edge of the final print.
  - **Target XY spacing (mm)**: Desired point spacing in the final STL.
- **Manual (step)**: You set the downsample step directly.
  - **Downsample step**: Keep every Nth grid point in X and Y.

**Grid tolerance**
- **Grid tolerance**: Snap XY to a grid to handle tiny floating noise (example: `0.001`).

**Z scale**
- **Z scale (tile conversion)**: Multiplies elevation for exaggeration (example: `2.0`).

---

## Step 3: Merge tiles into final STL

This step merges all tiles in `output/tiles` into a single STL and optionally adds a printable base.

**Options**
- **Output STL path**: Where the final STL should be saved.
- **Weld tolerance**: Removes tile seams by welding nearby vertices (example: `0.001` or `0.01`).
- **Merge Z scale**: Z scaling applied during merge only.
- **Make solid**: Adds side walls and a flat base for printing.
  - **Base thickness**: Base thickness below the minimum elevation.
  - **Base Z (optional)**: Explicit base elevation (overrides thickness).

---

## Files and Folders

```
project/
  gui.py
  download_tiles.py
  build_stl.py
  data/
    your_urls.csv
    xyz/
      tile_001.xyz
      tile_002.xyz
  output/
    tiles/
      tile_001.stl
      tile_002.stl
    terrain.stl
```

---

## CLI (optional)

Everything the GUI does is available via CLI if you prefer:

```bash
python download_tiles.py --csv path/to/urls.csv
python build_stl.py --all --target-size-mm 150
python build_stl.py --merge-stl output/terrain.stl --weld-tol 0.001 --make-solid
```

---

## Troubleshooting

**Nothing downloads**
- Check that your CSV has valid `http://` or `https://` URLs.

**Grid not detected**
- Increase **Grid tolerance** (example: `0.001`).
- Check for missing points in the XYZ grid.

**STL too large**
- Increase **Downsample step** (manual) or use a larger **Target XY spacing** (auto).

---

## License

Free to use, modify, and adapt for terrain processing, GIS, CAD, or 3D printing workflows.
