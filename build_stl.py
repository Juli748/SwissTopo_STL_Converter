#!/usr/bin/env python3
"""
xyz2stl.py

Convert ASCII XYZ point cloud(s) (x y z per line) into STL surface(s).

NEW:
- If --merge-stl is passed, the script merges existing STL files found under ./output/tiles
  into ONE combined STL WITHOUT re-processing any XYZ.

Notes:
- Binary STL merge needs a two-pass write (to know triangle count). ASCII can stream.
- This merger supports both ASCII and Binary STL inputs.
- Output can be Binary (default) or ASCII (--ascii).

Printable solid:
- If --make-solid is passed in XYZ->STL modes, the terrain surface is turned into a printable solid
  by adding a flat bottom and side walls (watertight mesh).

Global solid (important for printing):
- If --merge-stl is used together with --make-solid, the script will:
    1) load ALL STL tiles into memory
    2) weld shared vertices (tile seams) using --weld-tol
    3) solidify ONCE globally (one bottom, one outer wall)
  This avoids internal “steps” between tiles.
"""

#TODO: Add support for tiff input instead of only xyz, see sample data in /data

from __future__ import annotations

import argparse
import math
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray     # (M, 3) int64 indices into vertices


# ----------------------------
# XYZ -> mesh
# ----------------------------

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
                continue

            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                if parts[0].isalpha() or parts[1].isalpha() or parts[2].isalpha():
                    skipped += 1
                    continue
                raise ValueError(f"{path}:{line_no}: could not parse floats: {s}")

            pts.append((x, y, z))

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


def _min_spacing(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    values_sorted = sorted(values)
    diffs = [b - a for a, b in zip(values_sorted, values_sorted[1:]) if b > a]
    if not diffs:
        return None
    return min(diffs)


def scan_xyz_bounds_and_resolution(
    xyz_files: List[Path],
    *,
    round_decimals: int = 6,
) -> Tuple[float, float, float, float, float]:
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    min_spacing = float("inf")

    for path in xyz_files:
        print(f"[AUTO] Scanning XYZ: {path}")
        xs: set[float] = set()
        ys: set[float] = set()
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
                    continue

                try:
                    x, y = float(parts[0]), float(parts[1])
                except ValueError:
                    if parts[0].isalpha() or parts[1].isalpha():
                        skipped += 1
                        continue
                    raise ValueError(f"{path}:{line_no}: could not parse floats: {s}")

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                xs.add(round(x, round_decimals))
                ys.add(round(y, round_decimals))

        dx = _min_spacing(list(xs))
        dy = _min_spacing(list(ys))
        spacing_candidates = [v for v in (dx, dy) if v is not None and v > 0]
        file_spacing = min(spacing_candidates) if spacing_candidates else None
        if file_spacing is not None:
            min_spacing = min(min_spacing, file_spacing)
        print(f"[AUTO]  Skipped {skipped:,} lines")

    if not math.isfinite(min_x) or not math.isfinite(min_y):
        raise ValueError("No valid points found while scanning XYZ files.")

    if not math.isfinite(min_spacing):
        raise ValueError("Could not determine XY resolution from XYZ files.")

    return min_x, max_x, min_y, max_y, min_spacing


def _prompt_optional_float(prompt: str) -> Optional[float]:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if value <= 0:
            print("Value must be greater than zero.")
            continue
        return value


def _auto_scale_and_step(
    xyz_files: List[Path],
    *,
    target_size_mm: float,
    target_resolution_mm: float,
    edge_mode: str,
) -> Tuple[float, int]:
    min_x, max_x, min_y, max_y, min_spacing = scan_xyz_bounds_and_resolution(xyz_files)
    span_x = max_x - min_x
    span_y = max_y - min_y
    min_edge = min(span_x, span_y)
    max_edge = max(span_x, span_y)
    if min_edge <= 0 or max_edge <= 0:
        raise ValueError("Combined XY bounds have zero size.")

    edge_mode = edge_mode.lower()
    if edge_mode not in {"shortest", "longest"}:
        raise ValueError("edge_mode must be 'shortest' or 'longest'.")

    chosen_edge = min_edge if edge_mode == "shortest" else max_edge
    scale = float(target_size_mm) / float(chosen_edge)
    spacing_mm = min_spacing * scale
    if spacing_mm <= 0:
        raise ValueError("Computed spacing is not positive; check XYZ resolution.")

    if target_resolution_mm <= 0:
        raise ValueError("Target resolution must be > 0.")

    step = max(1, int(math.ceil(target_resolution_mm / spacing_mm)))

    print(f"[AUTO] Combined X span: {span_x:.3f}")
    print(f"[AUTO] Combined Y span: {span_y:.3f}")
    print(f"[AUTO] Min XY spacing: {min_spacing:.6f}")
    print(f"[AUTO] Target {edge_mode} edge: {target_size_mm:.2f} mm")
    print(f"[AUTO] Scale factor: {scale:.6f} (input units -> mm)")
    print(f"[AUTO] Target XY spacing: {target_resolution_mm:.2f} mm")
    print(f"[AUTO] Output XY spacing: {spacing_mm:.3f} mm -> step={step}")

    return scale, step


def _prompt_edge_mode() -> str:
    while True:
        raw = input("Which edge should match the target size? [shortest/longest]: ").strip().lower()
        if raw in {"shortest", "s"}:
            return "shortest"
        if raw in {"longest", "l"}:
            return "longest"
        print("Please enter 'shortest' or 'longest'.")


def _grid_prepare(points: np.ndarray, tol: float = 0.0) -> Optional[Tuple[np.ndarray, int, int]]:
    xy = points[:, :2]
    if tol > 0.0:
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
    if nx * ny != n:
        return None

    xs_sorted = np.sort(xs)
    ys_sorted = np.sort(ys)
    ix = np.searchsorted(xs_sorted, key_xy[:, 0])
    iy = np.searchsorted(ys_sorted, key_xy[:, 1])

    if np.any(ix < 0) or np.any(ix >= nx) or np.any(iy < 0) or np.any(iy >= ny):
        return None

    grid_index = iy * nx + ix
    if np.unique(grid_index).size != n:
        return None

    order = np.argsort(grid_index)
    verts = points[order]
    return verts, nx, ny


def try_structured_grid(
    points: np.ndarray,
    tol: float = 0.0,
    step: int = 1,
    *,
    assume_grid: bool = False,
) -> Optional[Mesh]:
    print("[2/4] Checking whether points form a structured grid...")

    if step < 1:
        raise ValueError("--step must be >= 1")

    prepared = _grid_prepare(points, tol=tol)
    if prepared is None:
        if assume_grid:
            raise ValueError("Structured grid expected but points do not form a complete grid.")
        print("  Not a complete grid -> will fall back to Delaunay triangulation (SciPy).")
        return None

    verts, nx, ny = prepared
    print(f"  Grid detected: {nx:,} x {ny:,} = {nx*ny:,} points")

    # Optional downsample
    if step > 1:
        print(f"  Downsampling grid by step={step} (keeping every {step}th point in X and Y)...")
        verts_grid = verts.reshape(ny, nx, 3)

        # Keep boundaries so adjacent tiles still touch even when (nx-1) or (ny-1) is not divisible by step.
        x_idx = np.unique(np.r_[np.arange(0, nx, step), nx - 1])
        y_idx = np.unique(np.r_[np.arange(0, ny, step), ny - 1])

        verts_grid = verts_grid[np.ix_(y_idx, x_idx)]
        ny2, nx2 = verts_grid.shape[0], verts_grid.shape[1]
        verts = verts_grid.reshape(ny2 * nx2, 3)

        print(f"  Grid size: {nx:,}x{ny:,} -> {nx2:,}x{ny2:,}")
        nx, ny = nx2, ny2

    print("  Building faces (vectorized)...")

    grid = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
    v00 = grid[:-1, :-1].ravel()
    v10 = grid[:-1, 1:].ravel()
    v01 = grid[1:, :-1].ravel()
    v11 = grid[1:, 1:].ravel()

    faces_arr = np.vstack([
        np.stack([v00, v10, v11], axis=1),
        np.stack([v00, v11, v01], axis=1),
    ])

    total_cells = (ny - 1) * (nx - 1)
    print(f"  Built {faces_arr.shape[0]:,} triangles from {total_cells:,} cells.")
    return Mesh(vertices=verts, faces=faces_arr)


def delaunay_triangulation(points: np.ndarray) -> Mesh:
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


# ----------------------------
# Mesh utilities / STL IO
# ----------------------------

def compute_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    norm = float(np.linalg.norm(n))
    if norm == 0.0 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0)
    n = n / norm
    return (float(n[0]), float(n[1]), float(n[2]))


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    n = np.divide(n, norm[:, None], out=np.zeros_like(n), where=norm[:, None] != 0)
    return n


def write_ascii_stl(mesh: Mesh, out_path: Path, solid_name: str = "terrain") -> None:
    print(f"[3/4] Writing ASCII STL: {out_path}")

    v = mesh.vertices
    f = mesh.faces
    report_every = max(1, f.shape[0] // 10)
    normals = compute_normals(v, f)

    with out_path.open("w", encoding="utf-8") as w:
        w.write(f"solid {solid_name}\n")
        for idx, (i0, i1, i2) in enumerate(f, 1):
            a, b, c = v[i0], v[i1], v[i2]
            nx, ny, nz = normals[idx - 1]
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
    print(f"[3/4] Writing Binary STL: {out_path}")

    v = mesh.vertices
    f = mesh.faces

    header = (solid_name[:80]).encode("ascii", errors="ignore").ljust(80, b"\0")
    normals = compute_normals(v, f).astype(np.float32, copy=False)
    a = v[f[:, 0]].astype(np.float32, copy=False)
    b = v[f[:, 1]].astype(np.float32, copy=False)
    c = v[f[:, 2]].astype(np.float32, copy=False)

    tri_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    tri_data = np.empty((f.shape[0],), dtype=tri_dtype)
    tri_data["normal"] = normals
    tri_data["v1"] = a
    tri_data["v2"] = b
    tri_data["v3"] = c
    tri_data["attr"] = 0

    with out_path.open("wb") as w:
        w.write(header)
        w.write(struct.pack("<I", int(f.shape[0])))
        w.write(tri_data.tobytes())

    print("[4/4] Done.")


def _is_probably_binary_stl(path: Path) -> bool:
    """
    Heuristic: binary STL has 80-byte header + 4-byte tri count, and file size matches 84 + 50*n.
    ASCII STL usually starts with 'solid'.
    """
    try:
        size = path.stat().st_size
        if size < 84:
            return False
        with path.open("rb") as f:
            header = f.read(80)
            count_bytes = f.read(4)
        n = int.from_bytes(count_bytes, "little", signed=False)
        expected = 84 + 50 * n
        if expected == size:
            return True
        if header[:5].lower() == b"solid":
            return False
        return False
    except Exception:
        return False


def read_stl_to_mesh(path: Path) -> Mesh:
    """
    Read an STL (ASCII or Binary) into a Mesh.
    Triangles are kept as-is; vertices are not welded/deduplicated here.
    """
    print(f"[READ] STL: {path}")
    if _is_probably_binary_stl(path):
        with path.open("rb") as f:
            f.seek(80)
            n = int.from_bytes(f.read(4), "little", signed=False)
            verts: List[Tuple[float, float, float]] = []
            faces: List[Tuple[int, int, int]] = []
            for _ in range(n):
                rec = f.read(50)
                if len(rec) != 50:
                    break
                vals = struct.unpack("<12fH", rec)
                ax, ay, az = vals[3], vals[4], vals[5]
                bx, by, bz = vals[6], vals[7], vals[8]
                cx, cy, cz = vals[9], vals[10], vals[11]
                i0 = len(verts)
                verts.append((ax, ay, az))
                verts.append((bx, by, bz))
                verts.append((cx, cy, cz))
                faces.append((i0, i0 + 1, i0 + 2))

        v = np.asarray(verts, dtype=np.float64)
        fcs = np.asarray(faces, dtype=np.int64)
        print(f"[READ]  Triangles: {fcs.shape[0]:,} | Raw vertices: {v.shape[0]:,}")
        return Mesh(vertices=v, faces=fcs)

    # ASCII
    verts2: List[Tuple[float, float, float]] = []
    faces2: List[Tuple[int, int, int]] = []
    tri: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("vertex"):
                parts = s.split()
                if len(parts) >= 4:
                    tri.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    if len(tri) == 3:
                        i0 = len(verts2)
                        verts2.extend(tri)
                        faces2.append((i0, i0 + 1, i0 + 2))
                        tri = []
    v = np.asarray(verts2, dtype=np.float64)
    fcs = np.asarray(faces2, dtype=np.int64)
    print(f"[READ]  Triangles: {fcs.shape[0]:,} | Raw vertices: {v.shape[0]:,}")
    return Mesh(vertices=v, faces=fcs)


def weld_vertices(mesh: Mesh, weld_tol: float) -> Mesh:
    """
    Deduplicate vertices by snapping to a grid of size weld_tol.
    This is critical to remove tile seams so global boundary detection works.

    weld_tol=0 disables welding.
    """
    if weld_tol <= 0.0:
        return mesh

    print(f"[WELD] Welding vertices with tol={weld_tol} ...")

    v = mesh.vertices
    f = mesh.faces

    q = np.round(v / weld_tol).astype(np.int64)

    mapping: Dict[Tuple[int, int, int], int] = {}
    new_verts: List[Tuple[float, float, float]] = []
    remap = np.empty((v.shape[0],), dtype=np.int64)

    for i in range(v.shape[0]):
        key = (int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
        j = mapping.get(key)
        if j is None:
            j = len(new_verts)
            mapping[key] = j
            new_verts.append((float(v[i, 0]), float(v[i, 1]), float(v[i, 2])))
        remap[i] = j

    f2 = remap[f]
    v2 = np.asarray(new_verts, dtype=np.float64)

    print(f"[WELD]  Vertices: {v.shape[0]:,} -> {v2.shape[0]:,}")
    return Mesh(vertices=v2, faces=f2.astype(np.int64, copy=False))


def concat_meshes(meshes: List[Mesh]) -> Mesh:
    """
    Concatenate multiple meshes (no welding).
    """
    if not meshes:
        raise ValueError("No meshes to concatenate")

    v_all: List[np.ndarray] = []
    f_all: List[np.ndarray] = []
    offset = 0
    for m in meshes:
        v_all.append(m.vertices)
        f_all.append(m.faces + offset)
        offset += int(m.vertices.shape[0])

    v = np.vstack(v_all)
    f = np.vstack(f_all).astype(np.int64, copy=False)
    return Mesh(vertices=v, faces=f)


# ----------------------------
# Solidify (printable)
# ----------------------------

def make_solid(mesh: Mesh, base_z: float) -> Mesh:
    """
    Make the terrain printable by filling everything below it:
      - duplicate all vertices at a constant Z = base_z
      - add a bottom cap
      - add side walls along the mesh boundary

    This is applied after the surface mesh is built.
    """
    print(f"[SOLID] Making watertight solid with base_z={base_z:.3f}")

    v_top = mesh.vertices
    f_top = mesh.faces.astype(np.int64, copy=False)

    n = int(v_top.shape[0])
    v_bot = v_top.copy()
    v_bot[:, 2] = float(base_z)

    vertices = np.vstack([v_top, v_bot])

    # Bottom cap (reverse winding relative to top)
    f_bottom = np.column_stack([f_top[:, 0] + n, f_top[:, 2] + n, f_top[:, 1] + n]).astype(np.int64)

    # Find boundary edges: edges that belong to exactly one triangle
    e01 = f_top[:, [0, 1]]
    e12 = f_top[:, [1, 2]]
    e20 = f_top[:, [2, 0]]
    edges_oriented = np.vstack([e01, e12, e20]).astype(np.int64)

    edges_sorted = np.sort(edges_oriented, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges_sorted = unique_edges[counts == 1]

    if boundary_edges_sorted.size == 0:
        print("[SOLID] No boundary edges found (mesh may already be closed). Returning original mesh.")
        return mesh

    boundary_set = {(int(a), int(b)) for a, b in boundary_edges_sorted}

    boundary_oriented: List[Tuple[int, int]] = []
    seen = set()
    for a, b in edges_oriented:
        key = (int(min(a, b)), int(max(a, b)))
        if key in boundary_set and key not in seen:
            boundary_oriented.append((int(a), int(b)))
            seen.add(key)

    side_faces: List[Tuple[int, int, int]] = []
    for a, b in boundary_oriented:
        a2 = a + n
        b2 = b + n
        side_faces.append((a, b, b2))
        side_faces.append((a, b2, a2))

    f_side = np.asarray(side_faces, dtype=np.int64)
    faces = np.vstack([f_top, f_bottom, f_side]).astype(np.int64, copy=False)

    print(f"[SOLID] Added bottom cap ({f_bottom.shape[0]:,} tris) and walls ({f_side.shape[0]:,} tris).")
    print(f"[SOLID] Total triangles: {faces.shape[0]:,}")

    return Mesh(vertices=vertices, faces=faces)


# ----------------------------
# Scaling
# ----------------------------

def scale_mesh_z(mesh: Mesh, z_scale: float) -> Mesh:
    if z_scale == 1.0:
        return mesh
    print(f"[SCALE] Scaling Z by {z_scale}")
    v = mesh.vertices.copy()
    v[:, 2] *= float(z_scale)
    return Mesh(vertices=v, faces=mesh.faces)


# ----------------------------
# Conversion helpers
# ----------------------------

def convert_one(
    xyz_path: Path,
    stl_path: Path,
    *,
    name: str,
    tol: float,
    z_scale: float,
    scale: float,
    step: int,
    binary: bool,
    make_solid_flag: bool,
    base_thickness_value: float,
    base_z_value: Optional[float],
    base_mode: str,
    assume_grid: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"Converting: {xyz_path.name} -> {stl_path.name}")
    print("=" * 80)

    pts = load_xyz(xyz_path)
    if scale != 1.0:
        print(f"[1/4] Applying scale: {scale}")
        pts = pts.copy()
        pts *= float(scale)
    if z_scale != 1.0:
        print(f"[1/4] Applying Z scale: {z_scale}")
        pts = pts.copy()
        pts[:, 2] *= float(z_scale)

    mesh = try_structured_grid(pts, tol=float(tol), step=int(step), assume_grid=bool(assume_grid))
    if mesh is None:
        mesh = delaunay_triangulation(pts)

    if make_solid_flag:
        if base_z_value is not None:
            base_z = float(base_z_value)
        elif base_mode == "sealevel":
            base_z = 0.0
        else:
            base_z = float(mesh.vertices[:, 2].min() - float(base_thickness_value))
        mesh = make_solid(mesh, base_z)

    if binary:
        write_binary_stl(mesh, stl_path, solid_name=name)
    else:
        write_ascii_stl(mesh, stl_path, solid_name=name)

    print(f"Wrote {stl_path} with {mesh.faces.shape[0]:,} triangles and {mesh.vertices.shape[0]:,} vertices.")


def _convert_worker(
    payload: Tuple[Path, Path, str, float, float, float, int, float, Optional[float], str, bool]
) -> str:
    (
        xyz_path,
        stl_path,
        per_file_name,
        tol,
        z_scale,
        scale,
        step,
        base_thickness_value,
        base_z_value,
        base_mode,
        assume_grid,
    ) = payload
    convert_one(
        xyz_path,
        stl_path,
        name=per_file_name,
        tol=float(tol),
        z_scale=float(z_scale),
        scale=float(scale),
        step=int(step),
        binary=True,
        make_solid_flag=False,
        base_thickness_value=float(base_thickness_value),
        base_z_value=base_z_value,
        base_mode=str(base_mode),
        assume_grid=bool(assume_grid),
    )
    return xyz_path.name


def _list_xyz_files() -> List[Path]:
    folder = Path("./data/xyz")
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder.resolve()}")
    xyz_files = sorted(folder.rglob("*.xyz"))
    if not xyz_files:
        raise SystemExit(f"No .xyz files found in: {folder.resolve()}")
    return xyz_files


def _list_stl_files() -> List[Path]:
    folder = Path("./output/tiles")
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder.resolve()}")
    stl_files = sorted(folder.rglob("*.stl"))
    if not stl_files:
        raise SystemExit(f"No .stl files found in: {folder.resolve()}")
    return stl_files


# ----------------------------
# STL merge modes
# ----------------------------

def merge_stls_streaming(out_stl: Path, *, binary_out: bool, solid_name: str = "terrain_merged") -> None:
    """
    Fast merge that simply concatenates triangles.
    Does not load the whole mesh. Cannot do global solidification.
    """
    stl_files = _list_stl_files()
    print(f"Found {len(stl_files)} STL file(s) under ./output/tiles")
    print(f"Merging into: {out_stl}")

    if binary_out:
        print("[MERGE] Counting triangles (required for binary STL output)...")
        total_tris = 0
        counts: List[int] = []
        for i, p in enumerate(stl_files, 1):
            if _is_probably_binary_stl(p):
                with p.open("rb") as f:
                    f.seek(80)
                    n = int.from_bytes(f.read(4), "little", signed=False)
            else:
                n = 0
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.lstrip().startswith("facet normal"):
                            n += 1
            counts.append(n)
            total_tris += n
            print(f"  ({i}/{len(stl_files)}) {p.name}: {n:,} tris (total {total_tris:,})")

        header = (solid_name[:80]).encode("ascii", errors="ignore").ljust(80, b"\0")
        with out_stl.open("wb") as out:
            out.write(header)
            out.write(struct.pack("<I", int(total_tris)))

            written = 0
            report_every = max(1, total_tris // 20)

            for i, p in enumerate(stl_files, 1):
                print("\n" + "=" * 80)
                print(f"[MERGE] ({i}/{len(stl_files)}) Appending {p}")
                print("=" * 80)

                if _is_probably_binary_stl(p):
                    with p.open("rb") as f:
                        f.seek(80)
                        n = int.from_bytes(f.read(4), "little", signed=False)
                        out.write(f.read(50 * n))
                        written += n
                else:
                    with p.open("r", encoding="utf-8", errors="ignore") as f:
                        tri_vertices: List[Tuple[float, float, float]] = []
                        for line in f:
                            s = line.strip()
                            if s.startswith("vertex"):
                                parts = s.split()
                                if len(parts) >= 4:
                                    tri_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                                    if len(tri_vertices) == 3:
                                        a = np.array(tri_vertices[0], dtype=np.float64)
                                        b = np.array(tri_vertices[1], dtype=np.float64)
                                        c = np.array(tri_vertices[2], dtype=np.float64)
                                        nx, ny, nz = compute_normal(a, b, c)

                                        out.write(struct.pack("<3f", float(nx), float(ny), float(nz)))
                                        out.write(struct.pack("<3f", float(a[0]), float(a[1]), float(a[2])))
                                        out.write(struct.pack("<3f", float(b[0]), float(b[1]), float(b[2])))
                                        out.write(struct.pack("<3f", float(c[0]), float(c[1]), float(c[2])))
                                        out.write(struct.pack("<H", 0))

                                        tri_vertices.clear()
                                        written += 1

                if written % report_every == 0 or written == total_tris:
                    pct = (written / total_tris) * 100.0
                    print(f"[MERGE] Progress: {written:,}/{total_tris:,} triangles ({pct:.0f}%)")

        print(f"[MERGE] Done. Wrote {out_stl} with {total_tris:,} triangles.")
        return

    # ASCII streaming
    print("[MERGE] Writing combined ASCII STL (streaming)...")
    with out_stl.open("w", encoding="utf-8") as out:
        out.write(f"solid {solid_name}\n")
        total_tris_streamed = 0

        for i, p in enumerate(stl_files, 1):
            print("\n" + "=" * 80)
            print(f"[MERGE] ({i}/{len(stl_files)}) Appending {p}")
            print("=" * 80)

            if _is_probably_binary_stl(p):
                with p.open("rb") as f:
                    f.seek(80)
                    n = int.from_bytes(f.read(4), "little", signed=False)
                    for _ in range(n):
                        rec = f.read(50)
                        if len(rec) != 50:
                            break
                        vals = struct.unpack("<12fH", rec)
                        ax, ay, az = vals[3], vals[4], vals[5]
                        bx, by, bz = vals[6], vals[7], vals[8]
                        cx, cy, cz = vals[9], vals[10], vals[11]
                        a = np.array([ax, ay, az], dtype=np.float64)
                        b = np.array([bx, by, bz], dtype=np.float64)
                        c = np.array([cx, cy, cz], dtype=np.float64)
                        nx, ny, nz = compute_normal(a, b, c)

                        out.write(f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}\n")
                        out.write("    outer loop\n")
                        out.write(f"      vertex {a[0]:.8e} {a[1]:.8e} {a[2]:.8e}\n")
                        out.write(f"      vertex {b[0]:.8e} {b[1]:.8e} {b[2]:.8e}\n")
                        out.write(f"      vertex {c[0]:.8e} {c[1]:.8e} {c[2]:.8e}\n")
                        out.write("    endloop\n")
                        out.write("  endfacet\n")
                        total_tris_streamed += 1
            else:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("solid") or s.startswith("endsolid"):
                            continue
                        out.write(line)
                        if s.startswith("facet normal"):
                            total_tris_streamed += 1

        out.write(f"endsolid {solid_name}\n")

    print(f"[MERGE] Done. Wrote {out_stl} with ~{total_tris_streamed:,} triangles.")


def merge_stls_mesh(
    out_stl: Path,
    *,
    binary_out: bool,
    solid_name: str,
    weld_tol: float,
    make_solid_flag: bool,
    base_thickness_value: float,
    base_z_value: Optional[float],
    base_mode: str,
    z_scale: float,
) -> None:
    """
    Full merge that loads all tiles into memory, welds vertices, and can solidify globally.
    """
    stl_files = _list_stl_files()
    print(f"Found {len(stl_files)} STL file(s) under ./output/tiles")
    print(f"[MERGE-MESH] Output: {out_stl}")

    meshes: List[Mesh] = []
    for i, p in enumerate(stl_files, 1):
        print("\n" + "=" * 80)
        print(f"[MERGE-MESH] ({i}/{len(stl_files)}) Reading {p}")
        print("=" * 80)
        meshes.append(read_stl_to_mesh(p))

    merged = concat_meshes(meshes)
    print(f"[MERGE-MESH] Combined raw vertices: {merged.vertices.shape[0]:,} | triangles: {merged.faces.shape[0]:,}")

    # Welding is the key to removing internal tile boundaries
    merged = weld_vertices(merged, weld_tol=float(weld_tol))

    if z_scale != 1.0:
        merged = scale_mesh_z(merged, float(z_scale))

    if make_solid_flag:
        if base_z_value is not None:
            base_z = float(base_z_value)
        elif base_mode == "sealevel":
            base_z = 0.0
        else:
            base_z = float(merged.vertices[:, 2].min() - float(base_thickness_value))
        merged = make_solid(merged, base_z)

    if binary_out:
        write_binary_stl(merged, out_stl, solid_name=solid_name)
    else:
        write_ascii_stl(merged, out_stl, solid_name=solid_name)

    print(f"[MERGE-MESH] Done. Wrote {out_stl}")


# ----------------------------
# Main CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert XYZ terrain point cloud(s) to STL surface(s).")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Convert all .xyz files in ./data/xyz into ./output/tiles.",
    )
    ap.add_argument(
        "--merge-stl",
        type=Path,
        default=None,
        help="Merge existing .stl files found under ./output/tiles into one combined STL (no XYZ processing).",
    )

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
        "--merge-z-scale",
        type=float,
        default=1.0,
        help="Multiply Z by this factor when using --merge-stl (default: 1.0).",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="Downsample structured grid by keeping every Nth point (default: 1 = no downsample). "
             "Example: step=10 turns 0.5m spacing into 5m spacing.",
    )
    ap.add_argument(
        "--target-size-mm",
        type=float,
        default=None,
        help="Target size of the chosen XY edge in mm. "
             "When set in --all mode, auto-compute scale and step from XYZ bounds/resolution.",
    )
    ap.add_argument(
        "--target-edge",
        type=str,
        default="shortest",
        choices=["shortest", "longest"],
        help="Which XY edge should match --target-size-mm (default: shortest).",
    )
    ap.add_argument(
        "--target-resolution-mm",
        type=float,
        default=0.3,
        help="Target XY point spacing in the final STL (mm) when using --target-size-mm (default: 0.3).",
    )
    ap.add_argument(
        "--make-solid",
        action="store_true",
        help="Make the terrain printable by adding a flat bottom and side walls (watertight solid). "
             "Ignored in --all mode; use --merge-stl for a global base.",
    )
    ap.add_argument(
        "--base-thickness",
        type=float,
        default=5.0,
        help="If --make-solid is set and --base-z is not given, base_z = min_z - base_thickness "
             "when --base-mode=fixed (default: 5.0).",
    )
    ap.add_argument(
        "--base-z",
        type=float,
        default=None,
        help="Explicit Z value for the bottom plane when using --make-solid (overrides --base-thickness).",
    )
    ap.add_argument(
        "--base-mode",
        type=str,
        default="fixed",
        choices=["fixed", "sealevel"],
        help="Base depth mode for --make-solid when --base-z is not set. "
             "'fixed' uses --base-thickness; 'sealevel' uses Z=0.",
    )

    ap.add_argument(
        "--weld-tol",
        type=float,
        default=0.001,
        help="Vertex weld tolerance used for merge/solidify (default: 0.001). "
             "Use something like 0.001 or 0.01 to remove tile seams.",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Optional model name used for output naming and STL solid name.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for --all conversion (default: 1).",
    )

    args = ap.parse_args()

    if args.target_size_mm is not None and not args.all:
        ap.error("--target-size-mm can only be used with --all.")

    if args.merge_stl is not None:
        # Load + weld for seamless joins; solidify once globally if requested.
        merge_name = args.model_name.strip() or "terrain_merged"
        merge_stls_mesh(
            Path(args.merge_stl),
            binary_out=True,
            solid_name=merge_name,
            weld_tol=float(args.weld_tol),
            make_solid_flag=bool(args.make_solid),
            base_thickness_value=float(args.base_thickness),
            base_z_value=args.base_z,
            base_mode=str(args.base_mode),
            z_scale=float(args.merge_z_scale),
        )
        return

    if args.all:
        if bool(args.make_solid):
            print("[WARN] --make-solid is ignored in --all mode. "
                  "Use --merge-stl --make-solid to add a global base after merging.")

        xyz_files = _list_xyz_files()
        print(f"Found {len(xyz_files)} .xyz file(s) in {Path('./data/xyz').resolve()}")

        target_size_mm = args.target_size_mm
        target_edge = str(args.target_edge)
        if target_size_mm is None and args.step == 1 and sys.stdin.isatty():
            target_edge = _prompt_edge_mode()
            edge_label = "shortest" if target_edge == "shortest" else "longest"
            target_size_mm = _prompt_optional_float(
                f"Target {edge_label} edge length in mm (blank to keep --step 1): "
            )

        auto_scale = 1.0
        auto_step = int(args.step)
        if target_size_mm is not None:
            auto_scale, auto_step = _auto_scale_and_step(
                xyz_files,
                target_size_mm=float(target_size_mm),
                target_resolution_mm=float(args.target_resolution_mm),
                edge_mode=target_edge,
            )

        output_tiles_dir = Path("./output/tiles")
        output_tiles_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[Tuple[Path, Path, str, float, float, float, int, float, Optional[float], str, bool]] = []
        model_name = args.model_name.strip()
        for xyz_path in xyz_files:
            if model_name:
                tile_stem = f"{model_name}_{xyz_path.stem}"
            else:
                tile_stem = xyz_path.stem
            stl_path = output_tiles_dir / f"{tile_stem}.stl"
            per_file_name = tile_stem
            tasks.append(
                (
                    xyz_path,
                    stl_path,
                    per_file_name,
                    float(args.tol),
                    float(args.z_scale),
                    float(auto_scale),
                    int(auto_step),
                    float(args.base_thickness),
                    args.base_z,
                    str(args.base_mode),
                    True,
                )
            )

        workers = max(1, int(args.workers))
        if workers == 1 or len(tasks) == 1:
            for task in tasks:
                xyz_path = task[0]
                try:
                    _convert_worker(task)
                except Exception as e:
                    print(f"ERROR converting {xyz_path}: {e}")
            return

        total = len(tasks)
        completed = 0
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_convert_worker, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                xyz_path = task[0]
                completed += 1
                try:
                    future.result()
                except Exception as e:
                    failures += 1
                    print(f"ERROR converting {xyz_path}: {e}")
                print(f"[PROGRESS] {completed}/{total} {xyz_path.name}")

        if failures:
            print(f"[WARN] {failures} tile(s) failed during conversion.")
        return

    ap.error("Use --all to convert tiles or --merge-stl <out.stl> to merge.")


if __name__ == "__main__":
    main()

    
