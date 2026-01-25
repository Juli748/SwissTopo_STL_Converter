#!/usr/bin/env python3
"""
xyz2stl.py

Convert ASCII XYZ point cloud(s) (x y z per line) into STL surface(s).

Input format (whitespace separated):
  x y z
  x y z
  ...

Triangulation strategy:
  1) If points form a structured grid (unique X * unique Y == N), build a grid mesh.
     Optionally downsample the grid with --step to reduce STL size.
  2) Otherwise, try a 2D Delaunay triangulation (requires SciPy).

Output:
  - ASCII STL by default
  - Binary STL with --binary (much smaller)

Batch mode:
  If --all is passed, the script converts all .xyz files in ./terrain (relative to where you run it).
  Output STL names are derived from input names (e.g. terrain.xyz -> terrain.stl) and are written
  next to the input file (inside ./terrain).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray     # (M, 3) int64 indices into vertices


def load_xyz(path: Path) -> np.ndarray:
    """
    Loads whitespace-separated XYZ points, ignoring blank lines, comment lines,
    and optional header lines like: "X Y Z".
    Returns an (N, 3) float64 array.
    """
    print(f"[1/4] Reading XYZ: {path}")

    pts: List[Tuple[float, float, float]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                skipped += 1
                continue

            parts = s.split()
            if len(parts) < 3:
                skipped += 1
                continue  # skip short/non-data lines

            # Skip header lines like "X Y Z" (or any non-numeric first 3 tokens)
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                # If it's a header (e.g. X Y Z), just ignore it; otherwise raise
                # Heuristic: if tokens look alphabetic, treat as header
                if parts[0].isalpha() or parts[1].isalpha() or parts[2].isalpha():
                    skipped += 1
                    continue
                raise ValueError(f"{path}:{line_no}: could not parse floats: {s}")

            pts.append((x, y, z))

            # Progress print for large files
            if len(pts) % 500_000 == 0:
                print(f"  ... parsed {len(pts):,} points (line {line_no:,})")

    if not pts:
        raise ValueError(f"{path}: no points found")

    arr = np.asarray(pts, dtype=np.float64)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    print(f"  Done: {arr.shape[0]:,} points (skipped {skipped:,} lines)")
    print(f"  Bounds X: {mins[0]:.3f} .. {maxs[0]:.3f}")
    print(f"  Bounds Y: {mins[1]:.3f} .. {maxs[1]:.3f}")
    print(f"  Bounds Z: {mins[2]:.3f} .. {maxs[2]:.3f}")

    return arr


def try_structured_grid(points: np.ndarray, tol: float = 0.0, step: int = 1) -> Optional[Mesh]:
    """
    If points form a full X-by-Y grid (unique X * unique Y == N), build triangles
    by cell connectivity. Works even if the file ordering is arbitrary.

    step: keep every Nth point in X and Y (downsample) to reduce output size.
    tol: optional tolerance for treating values as identical (0.0 means exact).
    """
    print("[2/4] Checking whether points form a structured grid...")

    if step < 1:
        raise ValueError("--step must be >= 1")

    xy = points[:, :2]

    if tol > 0.0:
        # Quantize to grid for grouping (helps with tiny floating noise)
        q = np.round(xy / tol) * tol
        xs = np.unique(q[:, 0])
        ys = np.unique(q[:, 1])
        key_xy = q
    else:
        xs = np.unique(xy[:, 0])
        ys = np.unique(xy[:, 1])
        key_xy = xy

    nx, ny = xs.size, ys.size
    n = points.shape[0]
    print(f"  Unique X: {nx:,} | Unique Y: {ny:,} | Product: {nx*ny:,} | Points: {n:,}")

    if nx * ny != n:
        print("  Not a complete grid -> will fall back to Delaunay triangulation (SciPy).")
        return None

    # Map each (x, y) to vertex index in a deterministic grid (iy, ix)
    xs_sorted = np.sort(xs)
    ys_sorted = np.sort(ys)

    ix = np.searchsorted(xs_sorted, key_xy[:, 0])
    iy = np.searchsorted(ys_sorted, key_xy[:, 1])

    if np.any(ix < 0) or np.any(ix >= nx) or np.any(iy < 0) or np.any(iy >= ny):
        print("  Grid indexing failed -> fall back to Delaunay triangulation (SciPy).")
        return None

    # Ensure every grid slot is filled exactly once
    grid_index = iy * nx + ix
    if np.unique(grid_index).size != n:
        print("  Duplicate/missing grid cells detected -> fall back to Delaunay triangulation (SciPy).")
        return None

    print("  Grid detected. Reordering vertices...")

    # Reorder vertices into grid order so face building is straightforward
    order = np.argsort(grid_index)
    verts = points[order]

    # Optional downsample
    if step > 1:
        print(f"  Downsampling grid by step={step} (keeping every {step}th point in X and Y)...")
        verts_grid = verts.reshape(ny, nx, 3)
        verts_grid = verts_grid[::step, ::step, :]
        ny2, nx2 = verts_grid.shape[0], verts_grid.shape[1]
        verts = verts_grid.reshape(ny2 * nx2, 3)
        print(f"  Grid size: {nx:,}x{ny:,} -> {nx2:,}x{ny2:,}")
        nx, ny = nx2, ny2

    print("  Building faces...")

    faces: List[Tuple[int, int, int]] = []
    total_cells = (ny - 1) * (nx - 1)
    report_every = max(1, total_cells // 10)

    cell_count = 0
    for row in range(ny - 1):
        for col in range(nx - 1):
            v00 = row * nx + col
            v10 = row * nx + (col + 1)
            v01 = (row + 1) * nx + col
            v11 = (row + 1) * nx + (col + 1)

            # Two triangles per quad. Winding chosen for +Z normals if surface is locally flat.
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

            cell_count += 1
            if cell_count % report_every == 0 or cell_count == total_cells:
                pct = (cell_count / total_cells) * 100.0
                print(f"  Faces: {cell_count:,}/{total_cells:,} cells ({pct:.0f}%)")

    faces_arr = np.asarray(faces, dtype=np.int64)
    print(f"  Built {faces_arr.shape[0]:,} triangles from {total_cells:,} cells.")
    return Mesh(vertices=verts, faces=faces_arr)


def delaunay_triangulation(points: np.ndarray) -> Mesh:
    """
    General triangulation using 2D Delaunay on (x, y).
    Requires SciPy.
    """
    print("[2/4] Running 2D Delaunay triangulation (SciPy)...")
    try:
        from scipy.spatial import Delaunay  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Point set does not look like a complete structured grid and SciPy is not available.\n"
            "Install SciPy (conda install scipy) or provide a gridded XYZ."
        ) from e

    tri = Delaunay(points[:, :2])
    faces = tri.simplices.astype(np.int64, copy=False)
    print(f"  Delaunay produced {faces.shape[0]:,} triangles.")
    return Mesh(vertices=points, faces=faces)


def compute_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute unit normal for triangle (a, b, c). If degenerate, returns (0,0,0).
    """
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    norm = float(np.linalg.norm(n))
    if norm == 0.0 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0)
    n = n / norm
    return (float(n[0]), float(n[1]), float(n[2]))


def write_ascii_stl(mesh: Mesh, out_path: Path, solid_name: str = "terrain") -> None:
    """
    Writes an ASCII STL.
    """
    print(f"[3/4] Writing ASCII STL: {out_path}")

    v = mesh.vertices
    f = mesh.faces

    report_every = max(1, f.shape[0] // 10)

    with out_path.open("w", encoding="utf-8") as w:
        w.write(f"solid {solid_name}\n")
        for idx, (i0, i1, i2) in enumerate(f, 1):
            a, b, c = v[i0], v[i1], v[i2]
            nx, ny, nz = compute_normal(a, b, c)
            w.write(f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}\n")
            w.write("    outer loop\n")
            w.write(f"      vertex {a[0]:.8e} {a[1]:.8e} {a[2]:.8e}\n")
            w.write(f"      vertex {b[0]:.8e} {b[1]:.8e} {b[2]:.8e}\n")
            w.write(f"      vertex {c[0]:.8e} {c[1]:.8e} {c[2]:.8e}\n")
            w.write("    endloop\n")
            w.write("  endfacet\n")

            if idx % report_every == 0 or idx == f.shape[0]:
                pct = (idx / f.shape[0]) * 100.0
                print(f"  ... {idx:,}/{f.shape[0]:,} triangles written ({pct:.0f}%)")

        w.write(f"endsolid {solid_name}\n")

    print("[4/4] Done.")


def write_binary_stl(mesh: Mesh, out_path: Path, solid_name: str = "terrain") -> None:
    """
    Writes a Binary STL (much smaller than ASCII).
    """
    print(f"[3/4] Writing Binary STL: {out_path}")

    v = mesh.vertices
    f = mesh.faces

    report_every = max(1, f.shape[0] // 10)

    header = (solid_name[:80]).encode("ascii", errors="ignore").ljust(80, b"\0")

    import struct
    with out_path.open("wb") as w:
        w.write(header)
        w.write(struct.pack("<I", int(f.shape[0])))

        for idx, (i0, i1, i2) in enumerate(f, 1):
            a, b, c = v[i0], v[i1], v[i2]
            nx, ny, nz = compute_normal(a, b, c)

            w.write(struct.pack("<3f", float(nx), float(ny), float(nz)))
            w.write(struct.pack("<3f", float(a[0]), float(a[1]), float(a[2])))
            w.write(struct.pack("<3f", float(b[0]), float(b[1]), float(b[2])))
            w.write(struct.pack("<3f", float(c[0]), float(c[1]), float(c[2])))
            w.write(struct.pack("<H", 0))  # attribute byte count

            if idx % report_every == 0 or idx == f.shape[0]:
                pct = (idx / f.shape[0]) * 100.0
                print(f"  ... {idx:,}/{f.shape[0]:,} triangles written ({pct:.0f}%)")

    print("[4/4] Done.")


def convert_one(xyz_path: Path, stl_path: Path, *, name: str, tol: float, z_scale: float, step: int, binary: bool) -> None:
    print("\n" + "=" * 80)
    print(f"Converting: {xyz_path.name} -> {stl_path.name}")
    print("=" * 80)

    pts = load_xyz(xyz_path)
    if z_scale != 1.0:
        print(f"[1/4] Applying Z scale: {z_scale}")
        pts = pts.copy()
        pts[:, 2] *= float(z_scale)

    mesh = try_structured_grid(pts, tol=float(tol), step=int(step))
    if mesh is None:
        mesh = delaunay_triangulation(pts)

    if binary:
        write_binary_stl(mesh, stl_path, solid_name=name)
    else:
        write_ascii_stl(mesh, stl_path, solid_name=name)

    print(f"Wrote {stl_path} with {mesh.faces.shape[0]:,} triangles and {mesh.vertices.shape[0]:,} vertices.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert XYZ terrain point cloud(s) to STL surface(s).")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Convert all .xyz files in ./terrain (relative to the current working directory).",
    )
    ap.add_argument("xyz", nargs="?", type=Path, help="Input XYZ file (x y z per line)")
    ap.add_argument("stl", nargs="?", type=Path, help="Output STL file")

    ap.add_argument("--name", default="terrain", help="STL solid name (default: terrain)")
    ap.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Grid tolerance for matching X/Y values (default: 0.0 exact). "
             "Try e.g. 0.001 if your coordinates have tiny floating noise.",
    )
    ap.add_argument(
        "--z-scale",
        type=float,
        default=1.0,
        help="Multiply Z by this factor (default: 1.0)",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="Downsample structured grid by keeping every Nth point (default: 1 = no downsample). "
             "Example: step=10 turns 0.5m spacing into 5m spacing.",
    )
    ap.add_argument(
        "--binary",
        action="store_true",
        help="Write Binary STL instead of ASCII (much smaller).",
    )
    args = ap.parse_args()

    if args.all:
        folder = Path("./terrain")
        if not folder.exists():
            raise SystemExit(f"Folder not found: {folder.resolve()}")

        xyz_files = sorted(folder.glob("*.xyz"))
        if not xyz_files:
            raise SystemExit(f"No .xyz files found in: {folder.resolve()}")

        print(f"Found {len(xyz_files)} .xyz file(s) in {folder.resolve()}")
        for xyz_path in xyz_files:
            stl_path = xyz_path.with_suffix(".stl")
            per_file_name = xyz_path.stem
            convert_one(
                xyz_path,
                stl_path,
                name=per_file_name,
                tol=float(args.tol),
                z_scale=float(args.z_scale),
                step=int(args.step),
                binary=bool(args.binary),
            )
        return

    if args.xyz is None or args.stl is None:
        ap.error("Either provide xyz stl arguments, or run with --all to convert all .xyz files in ./terrain.")

    convert_one(
        args.xyz,
        args.stl,
        name=str(args.name),
        tol=float(args.tol),
        z_scale=float(args.z_scale),
        step=int(args.step),
        binary=bool(args.binary),
    )


if __name__ == "__main__":
    main()
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

### 1) Convert a single file

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
