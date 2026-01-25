# XYZ → STL Terrain Converter

This tool converts terrain point clouds stored as `.xyz` files into STL surface meshes.
It is designed for **large gridded terrain datasets** (e.g. 0.5 m resolution, 1 km × 1 km)
and includes options to **reduce file size** and **batch-process many files**.

---

## Input format

Each `.xyz` file must be plain text with one point per line:

```
X Y Z
793483.25 1139000.25 3602.40
793483.75 1139000.25 3602.23
...
```

Notes:

* A header line like `X Y Z` is allowed
* Blank lines and `#` comments are ignored
* Coordinates must form a **complete X/Y grid** for best results

---

## Getting terrain data

Terrain data used with this tool can be obtained from **swisstopo**:

[https://www.swisstopo.admin.ch/de/hoehenmodell-swissalti3d](https://www.swisstopo.admin.ch/de/hoehenmodell-swissalti3d)

Alternatively, if you want the **surface model** (including buildings, vegetation, etc.), you can use **swissSURFACE3D (Raster)**:

[https://www.swisstopo.admin.ch/de/hoehenmodell-swisssurface3d-raster](https://www.swisstopo.admin.ch/de/hoehenmodell-swisssurface3d-raster)

The swissSURFACE3D dataset can likewise be exported or converted to **XYZ (X Y Z)** format and processed with this tool.

Download the **swissALTI3D** dataset for your area of interest and export or convert the height model to **XYZ (X Y Z)** point format using your GIS or processing software.

Typical workflow:

* Download swissALTI3D tiles as .xyz
* Export the elevation grid as **XYZ**
* Place the resulting `.xyz` files into the `./terrain` folder

---

## Requirements

* Python (tested with conda environments)
* NumPy (required)
* SciPy (only needed if grid detection fails)

Install dependencies (recommended with conda):

```
conda install numpy scipy
```

---

## Folder structure (recommended)

```
project/
├─ build_stl.py
├─ terrain/
│  ├─ tile_001.xyz
│  ├─ tile_002.xyz
│  └─ ...
```

All batch operations assume `.xyz` files live in the `./terrain` folder.

---

## Basic usage

### 0) Merge existing STL tiles into ONE STL (no XYZ processing)

If you already converted all tiles to STL and just want **one combined terrain STL**, use:

```
python build_stl.py --merge-stl merged_terrain.stl --binary
```

This:

* Searches **recursively** for all `.stl` files under `./terrain`
* Does **not** re-read or re-triangulate any `.xyz` files
* Merges all triangles into **one big STL**
* Preserves geometry exactly (no resampling)

Notes:

* Binary output is strongly recommended for large terrains
* Both ASCII and Binary input STLs are supported

---

## Basic usage

### 1) Convert a single file

```
python build_stl.py input.xyz output.stl
```

This:

* Reads `input.xyz`
* Builds a terrain surface
* Writes `output.stl` (ASCII STL by default)

#### Make it printable (solid)

If you want a **watertight, 3D-printable** model (filled below the terrain), add `--make-solid`.

```
python build_stl.py input.xyz output.stl --binary --make-solid --base-thickness 10
```

Options:

* `--make-solid` adds a flat bottom + side walls
* `--base-thickness T` sets the bottom plane to `minZ - T`
* `--base-z Z` sets an explicit bottom plane Z (overrides thickness)

```
python build_stl.py input.xyz output.stl
```

This:

* Reads `input.xyz`
* Builds a terrain surface
* Writes `output.stl` (ASCII STL by default)

---

### 2) Convert all XYZ files in `./terrain/` (batch mode)

```
python build_stl.py --all
```

This:

* Looks for all `.xyz` files inside `./terrain`
* Converts each file independently
* Writes `file.stl` next to `file.xyz`

Example:

```
terrain/
 ├─ tile_001.xyz → tile_001.stl
 ├─ tile_002.xyz → tile_002.stl
```

---

## Important options (you will usually want these)

### `--binary` (strongly recommended)

Writes **Binary STL** instead of ASCII.
Binary STL files are **10–20× smaller**.

```
--binary
```

Example:

```
python build_stl.py --all --binary
```

---

### `--step N` (reduce mesh size)

Downsamples the terrain grid by keeping every `N`th point in X and Y.

Your data: **0.5 m resolution**

| Step | Effective resolution | Use case                |
| ---- | -------------------- | ----------------------- |
| 1    | 0.5 m                | Very detailed, huge STL |
| 2    | 1.0 m                | High detail             |
| 4    | 2.0 m                | Good balance            |
| 10   | 5.0 m                | Very manageable         |

Example (recommended for large areas):

```
python build_stl.py --all --step 10 --binary
```

---

### `--tol` (grid tolerance)

If grid detection fails due to tiny floating-point noise in X/Y coordinates,
use a tolerance to snap points together.

Typical values:

```
--tol 0.001
--tol 0.01
```

Example:

```
python build_stl.py --all --tol 0.001 --step 10 --binary
```

---

### `--z-scale` (vertical scaling)

Multiplies all Z values.

Examples:

* exaggerate terrain: `--z-scale 2`
* reduce vertical scale: `--z-scale 0.5`

```
python build_stl.py input.xyz output.stl --z-scale 2 --binary
```

---

### `--name` (single-file mode only)

Sets the STL “solid name” inside the file.

```
python build_stl.py input.xyz output.stl --name MyTerrain
```

In batch mode (`--all`), the filename is used automatically.

---

## How triangulation works (automatic)

You do **not** select this manually.

1. **Structured grid mode (preferred)**

   * Used when `unique X × unique Y == number of points`
   * Fast, predictable, supports `--step`

2. **Delaunay triangulation (fallback)**

   * Used if the grid is incomplete or irregular
   * Requires SciPy
   * `--step` is ignored in this mode

The script prints which mode is used.

---

## Recommended commands (most users)

### Large terrain tiles (best balance)

```
python build_stl.py --all --step 10 --binary
```

### Higher detail terrain

```
python build_stl.py --all --step 4 --binary
```

### Debug a single tile

```
python build_stl.py terrain/tile_001.xyz terrain/tile_001.stl --step 4
```

---

## Output size expectations (1 km × 1 km)

Approximate triangle counts:

| Step | Triangles  |
| ---- | ---------- |
| 1    | ~8 million |
| 2    | ~2 million |
| 4    | ~500k      |
| 10   | ~80k       |

Binary STL at `step=10` is usually **a few MB**.

---

## Troubleshooting

### “Not a complete grid → Delaunay”

* Try adding `--tol 0.001`
* Check for missing points in the XYZ file

### “SciPy not available”

```
conda install scipy
```

### Very slow or huge STL

* Increase `--step`
* Always use `--binary`

---

## License / Usage

Free to use, modify, and adapt for terrain processing, GIS, CAD, or 3D printing workflows.
