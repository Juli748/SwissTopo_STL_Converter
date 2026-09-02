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
  This avoids internal "steps" between tiles.
"""


from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


GEOMETRY_DATA_DIRNAME = "reference_data"
AUTO_BUILDINGS_DIR = Path("work") / "buildings" / "auto"
DEFAULT_RIVERBANK_GEOJSON = Path("work") / "water" / "riverbanks.geojson"


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray     # (M, 3) int64 indices into vertices


class CropOutsideInput(ValueError):
    pass


@dataclass(frozen=True)
class CropRect:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def _parse_crop_rect(values: Optional[List[float]]) -> Optional[CropRect]:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("--crop-rect requires exactly 4 values: WEST SOUTH EAST NORTH")

    west, south, east, north = (float(v) for v in values)
    if west >= east:
        raise ValueError("--crop-rect WEST must be smaller than EAST")
    if south >= north:
        raise ValueError("--crop-rect SOUTH must be smaller than NORTH")
    return CropRect(min_x=west, min_y=south, max_x=east, max_y=north)


def _bounds_intersect_rect(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    crop_rect: Optional[CropRect],
) -> bool:
    if crop_rect is None:
        return True
    return not (
        max_x < crop_rect.min_x or
        min_x > crop_rect.max_x or
        max_y < crop_rect.min_y or
        min_y > crop_rect.max_y
    )


def _crop_points(points: np.ndarray, crop_rect: Optional[CropRect], *, label: str) -> np.ndarray:
    if crop_rect is None:
        return points

    mask = (
        (points[:, 0] >= crop_rect.min_x) &
        (points[:, 0] <= crop_rect.max_x) &
        (points[:, 1] >= crop_rect.min_y) &
        (points[:, 1] <= crop_rect.max_y)
    )
    cropped = points[mask]
    print(
        f"[CROP] {label}: kept {cropped.shape[0]:,}/{points.shape[0]:,} points "
        f"inside X {crop_rect.min_x:.3f}..{crop_rect.max_x:.3f}, "
        f"Y {crop_rect.min_y:.3f}..{crop_rect.max_y:.3f}"
    )
    if cropped.size == 0:
        raise CropOutsideInput(f"{label}: no points remain after --crop-rect")
    return cropped


def _crop_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    crop_rect: Optional[CropRect],
    *,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if crop_rect is None:
        return xs, ys, zs

    if xs.ndim == 1 and ys.ndim == 1:
        x_mask = (xs >= crop_rect.min_x) & (xs <= crop_rect.max_x)
        y_mask = (ys >= crop_rect.min_y) & (ys <= crop_rect.max_y)
        if int(x_mask.sum()) < 2 or int(y_mask.sum()) < 2:
            raise CropOutsideInput(f"{label}: crop rectangle leaves fewer than 2 raster samples in X or Y")
        cropped_xs = xs[x_mask]
        cropped_ys = ys[y_mask]
        cropped_zs = zs[np.ix_(y_mask, x_mask)]
        print(
            f"[CROP] {label}: raster grid {len(xs):,}x{len(ys):,} -> "
            f"{len(cropped_xs):,}x{len(cropped_ys):,}"
        )
        return cropped_xs, cropped_ys, cropped_zs

    mask = (
        (xs >= crop_rect.min_x) &
        (xs <= crop_rect.max_x) &
        (ys >= crop_rect.min_y) &
        (ys <= crop_rect.max_y)
    )
    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)
    if int(row_mask.sum()) < 2 or int(col_mask.sum()) < 2:
        raise CropOutsideInput(f"{label}: crop rectangle leaves fewer than 2 raster samples in X or Y")
    cropped_xs = xs[np.ix_(row_mask, col_mask)]
    cropped_ys = ys[np.ix_(row_mask, col_mask)]
    cropped_zs = zs[np.ix_(row_mask, col_mask)]
    print(
        f"[CROP] {label}: raster grid {zs.shape[1]:,}x{zs.shape[0]:,} -> "
        f"{cropped_zs.shape[1]:,}x{cropped_zs.shape[0]:,}"
    )
    return cropped_xs, cropped_ys, cropped_zs


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


def load_geotiff_grid(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a GeoTIFF/COG into grid coordinates.
    Returns (xs, ys, z_grid) where xs shape (nx,), ys shape (ny,), z_grid shape (ny, nx).
    """
    try:
        import rasterio
        from rasterio.transform import xy as transform_xy
    except Exception as e:
        raise RuntimeError(
            "GeoTIFF input requires rasterio. Install it via: pip install rasterio"
        ) from e

    print(f"[1/4] Reading GeoTIFF: {path}")
    with rasterio.open(path) as dataset:
        if dataset.count < 1:
            raise ValueError(f"{path}: no raster bands found")
        z = dataset.read(1, masked=True)
        transform = dataset.transform
        height, width = z.shape

        rotated = abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12
        if rotated:
            rows = np.arange(height)
            cols = np.arange(width)
            xs, ys = transform_xy(transform, rows[:, None], cols[None, :], offset="center")
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
        else:
            xs = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
            ys = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e

    if np.ma.is_masked(z):
        mask = np.ma.getmaskarray(z)
        if mask.any():
            valid = z.compressed()
            fill_value = float(valid.min()) if valid.size else 0.0
            z = z.filled(fill_value)
            print(f"[1/4]  Filled {int(mask.sum()):,} nodata pixels with {fill_value:.3f}")
        else:
            z = z.filled(0.0)
    else:
        z = np.asarray(z, dtype=np.float64)

    z = np.asarray(z, dtype=np.float64)
    if xs.ndim == 1 and ys.ndim == 1:
        if z.shape != (len(ys), len(xs)):
            raise ValueError(f"{path}: unexpected raster shape {z.shape} vs grid {len(ys)}x{len(xs)}")
    else:
        if z.shape != xs.shape or z.shape != ys.shape:
            raise ValueError(f"{path}: unexpected raster shape {z.shape} vs XY grids {xs.shape}/{ys.shape}")

    print(f"  Grid size: {len(xs):,} x {len(ys):,} = {z.size:,} samples")
    print(f"  Bounds X: {float(xs.min()):.3f} .. {float(xs.max()):.3f}")
    print(f"  Bounds Y: {float(ys.min()):.3f} .. {float(ys.max()):.3f}")
    print(f"  Bounds Z: {float(z.min()):.3f} .. {float(z.max()):.3f}")

    return xs, ys, z


def _min_spacing(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    values_sorted = sorted(values)
    diffs = [b - a for a, b in zip(values_sorted, values_sorted[1:]) if b > a]
    if not diffs:
        return None
    return min(diffs)


def _scan_xyz_bounds_and_resolution(
    xyz_files: List[Path],
    *,
    round_decimals: int = 6,
    crop_rect: Optional[CropRect] = None,
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

                if crop_rect is None or (
                    crop_rect.min_x <= x <= crop_rect.max_x and
                    crop_rect.min_y <= y <= crop_rect.max_y
                ):
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


def _scan_tif_bounds_and_resolution(
    tif_files: List[Path],
    *,
    crop_rect: Optional[CropRect] = None,
) -> Tuple[float, float, float, float, float]:
    try:
        import rasterio
    except Exception as e:
        raise RuntimeError(
            "GeoTIFF input requires rasterio. Install it via: pip install rasterio"
        ) from e

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    min_spacing = float("inf")

    for path in tif_files:
        print(f"[AUTO] Scanning GeoTIFF: {path}")
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            if not _bounds_intersect_rect(bounds.left, bounds.bottom, bounds.right, bounds.top, crop_rect):
                print("[AUTO]  Outside crop rectangle")
                continue
            file_min_x = max(float(bounds.left), crop_rect.min_x) if crop_rect is not None else float(bounds.left)
            file_max_x = min(float(bounds.right), crop_rect.max_x) if crop_rect is not None else float(bounds.right)
            file_min_y = max(float(bounds.bottom), crop_rect.min_y) if crop_rect is not None else float(bounds.bottom)
            file_max_y = min(float(bounds.top), crop_rect.max_y) if crop_rect is not None else float(bounds.top)
            min_x = min(min_x, file_min_x)
            max_x = max(max_x, file_max_x)
            min_y = min(min_y, file_min_y)
            max_y = max(max_y, file_max_y)
            res_x, res_y = dataset.res
            spacing = min(abs(float(res_x)), abs(float(res_y)))
            if spacing > 0:
                min_spacing = min(min_spacing, spacing)

    if not math.isfinite(min_x) or not math.isfinite(min_y):
        raise ValueError("No valid GeoTIFF bounds found while scanning.")

    if not math.isfinite(min_spacing):
        raise ValueError("Could not determine XY resolution from GeoTIFF files.")

    return min_x, max_x, min_y, max_y, min_spacing


def scan_input_bounds_and_resolution(
    input_files: List[Path],
    *,
    crop_rect: Optional[CropRect] = None,
) -> Tuple[float, float, float, float, float]:
    xyz_files = [p for p in input_files if p.suffix.lower() == ".xyz"]
    tif_files = [p for p in input_files if p.suffix.lower() in {".tif", ".tiff"}]

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    min_spacing = float("inf")

    if xyz_files:
        x0, x1, y0, y1, spacing = _scan_xyz_bounds_and_resolution(xyz_files, crop_rect=crop_rect)
        min_x = min(min_x, x0)
        max_x = max(max_x, x1)
        min_y = min(min_y, y0)
        max_y = max(max_y, y1)
        min_spacing = min(min_spacing, spacing)

    if tif_files:
        x0, x1, y0, y1, spacing = _scan_tif_bounds_and_resolution(tif_files, crop_rect=crop_rect)
        min_x = min(min_x, x0)
        max_x = max(max_x, x1)
        min_y = min(min_y, y0)
        max_y = max(max_y, y1)
        min_spacing = min(min_spacing, spacing)

    if not math.isfinite(min_x) or not math.isfinite(min_y):
        raise ValueError("No valid points found while scanning input files.")

    if not math.isfinite(min_spacing):
        raise ValueError("Could not determine XY resolution from input files.")

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


def _parse_scale_ratio(raw: str) -> float:
    """
    Parse scale ratios like "100", "1:100", or "1/100" into a numeric ratio.
    Returns the real-world : model ratio (e.g. "1:100" -> 100.0).
    """
    s = raw.strip()
    if not s:
        raise ValueError("Scale ratio is empty.")

    if ":" in s:
        parts = s.split(":")
    elif "/" in s:
        parts = s.split("/")
    else:
        parts = [s]

    if len(parts) == 1:
        ratio = float(parts[0])
    elif len(parts) == 2:
        a = float(parts[0])
        b = float(parts[1])
        if a == 0:
            raise ValueError("Scale ratio numerator cannot be zero.")
        ratio = b / a
    else:
        raise ValueError("Invalid scale ratio format.")

    if ratio <= 0:
        raise ValueError("Scale ratio must be > 0.")
    return ratio


def _auto_scale_and_step(
    input_files: List[Path],
    *,
    target_size_mm: float,
    target_resolution_mm: float,
    edge_mode: str,
    crop_rect: Optional[CropRect] = None,
) -> Tuple[float, int]:
    min_x, max_x, min_y, max_y, min_spacing = scan_input_bounds_and_resolution(
        input_files,
        crop_rect=crop_rect,
    )
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


def _auto_step_from_scale(
    input_files: List[Path],
    *,
    scale: float,
    target_resolution_mm: float,
    crop_rect: Optional[CropRect] = None,
) -> int:
    _, _, _, _, min_spacing = scan_input_bounds_and_resolution(input_files, crop_rect=crop_rect)
    spacing_mm = min_spacing * float(scale)
    if spacing_mm <= 0:
        raise ValueError("Computed spacing is not positive; check input resolution/scale.")
    if target_resolution_mm <= 0:
        raise ValueError("Target resolution must be > 0.")
    step = max(1, int(math.ceil(target_resolution_mm / spacing_mm)))
    print(f"[AUTO] Min XY spacing: {min_spacing:.6f}")
    print(f"[AUTO] Scale factor: {float(scale):.6f} (input units -> mm)")
    print(f"[AUTO] Target XY spacing: {target_resolution_mm:.2f} mm")
    print(f"[AUTO] Output XY spacing: {spacing_mm:.3f} mm -> step={step}")
    return step


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


def mesh_from_grid(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, *, step: int = 1) -> Mesh:
    if step < 1:
        raise ValueError("--step must be >= 1")
    if xs.ndim == 1 and ys.ndim == 1:
        if zs.shape != (len(ys), len(xs)):
            raise ValueError("Grid dimensions do not match Z array.")
    else:
        if zs.shape != xs.shape or zs.shape != ys.shape:
            raise ValueError("Grid dimensions do not match Z array.")

    if step > 1:
        if xs.ndim == 1 and ys.ndim == 1:
            x_idx = np.unique(np.r_[np.arange(0, len(xs), step), len(xs) - 1])
            y_idx = np.unique(np.r_[np.arange(0, len(ys), step), len(ys) - 1])
            xs = xs[x_idx]
            ys = ys[y_idx]
            zs = zs[np.ix_(y_idx, x_idx)]
        else:
            y_idx = np.unique(np.r_[np.arange(0, zs.shape[0], step), zs.shape[0] - 1])
            x_idx = np.unique(np.r_[np.arange(0, zs.shape[1], step), zs.shape[1] - 1])
            xs = xs[np.ix_(y_idx, x_idx)]
            ys = ys[np.ix_(y_idx, x_idx)]
            zs = zs[np.ix_(y_idx, x_idx)]

    ny, nx = zs.shape
    if xs.ndim == 1 and ys.ndim == 1:
        xv, yv = np.meshgrid(xs, ys)
    else:
        xv, yv = xs, ys
    verts = np.column_stack([xv.ravel(), yv.ravel(), zs.ravel()])

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
    crop_rect: Optional[CropRect],
) -> None:
    print("\n" + "=" * 80)
    print(f"Converting: {xyz_path.name} -> {stl_path.name}")
    print("=" * 80)

    if xyz_path.suffix.lower() in {".tif", ".tiff"}:
        xs, ys, zs = load_geotiff_grid(xyz_path)
        xs, ys, zs = _crop_grid(xs, ys, zs, crop_rect, label=xyz_path.name)
        if scale != 1.0:
            print(f"[1/4] Applying scale: {scale}")
            xs = xs * float(scale)
            ys = ys * float(scale)
            zs = zs * float(scale)
        if z_scale != 1.0:
            print(f"[1/4] Applying Z scale: {z_scale}")
            zs = zs * float(z_scale)
        print("[2/4] Building mesh from raster grid...")
        mesh = mesh_from_grid(xs, ys, zs, step=int(step))
    else:
        pts = load_xyz(xyz_path)
        pts = _crop_points(pts, crop_rect, label=xyz_path.name)
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
    payload: Tuple[Path, Path, str, float, float, float, int, float, Optional[float], str, bool, Optional[CropRect]]
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
        crop_rect,
    ) = payload
    try:
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
            crop_rect=crop_rect,
        )
    except CropOutsideInput as exc:
        print(f"[SKIP] {exc}")
    return xyz_path.name


def _list_input_files() -> List[Path]:
    xyz_folder = Path("./work/terrain/xyz")
    tif_folder = Path("./work/terrain/tif")
    xyz_files = sorted(xyz_folder.rglob("*.xyz")) if xyz_folder.exists() else []
    tif_files = []
    if tif_folder.exists():
        tif_files = sorted(tif_folder.rglob("*.tif")) + sorted(tif_folder.rglob("*.tiff"))
    input_files = []
    for path in tif_files:
        if path not in input_files:
            input_files.append(path)

    if input_files and xyz_files:
        print(
            "[INFO] GeoTIFF inputs found. Ignoring XYZ tiles to avoid duplicate coverage."
        )
        return input_files

    input_files = xyz_files + input_files
    if not input_files:
        raise SystemExit(
            "No .xyz or .tif files found in ./work/terrain/xyz or ./work/terrain/tif."
        )
    return input_files


def _list_stl_files() -> List[Path]:
    folder = Path("./output/tiles")
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder.resolve()}")
    stl_files = sorted(folder.rglob("*.stl"))
    if not stl_files:
        raise SystemExit(f"No .stl files found in: {folder.resolve()}")
    return stl_files


def _tile_id_from_path(path: Path) -> Optional[Tuple[int, int]]:
    matches = re.findall(r"(?<!\d)(\d{4})-(\d{4})(?!\d)", path.stem)
    if not matches:
        return None
    x_raw, y_raw = matches[-1]
    return int(x_raw), int(y_raw)


def _validate_no_enclosed_tile_holes(stl_files: List[Path]) -> None:
    tile_ids = {_tile_id_from_path(path) for path in stl_files}
    tile_ids.discard(None)
    present = {tile_id for tile_id in tile_ids if tile_id is not None}
    if len(present) < 2:
        return

    xs = [x for x, _ in present]
    ys = [y for _, y in present]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    search_min_x, search_max_x = min_x - 1, max_x + 1
    search_min_y, search_max_y = min_y - 1, max_y + 1

    outside: set[Tuple[int, int]] = set()
    queue: List[Tuple[int, int]] = []
    for x in range(search_min_x, search_max_x + 1):
        queue.append((x, search_min_y))
        queue.append((x, search_max_y))
    for y in range(search_min_y + 1, search_max_y):
        queue.append((search_min_x, y))
        queue.append((search_max_x, y))

    while queue:
        x, y = queue.pop()
        cell = (x, y)
        if cell in outside or cell in present:
            continue
        if x < search_min_x or x > search_max_x or y < search_min_y or y > search_max_y:
            continue
        outside.add(cell)
        queue.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))

    holes = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            cell = (x, y)
            if cell not in present and cell not in outside:
                holes.append(cell)

    if not holes:
        return

    preview = ", ".join(f"{x}-{y}" for x, y in holes[:20])
    if len(holes) > 20:
        preview += f", ... ({len(holes)} total)"
    raise SystemExit(
        "Refusing to merge because the STL tile set has enclosed square hole(s): "
        f"{preview}. Re-download or re-convert the missing tile(s), then run the merge again."
    )


def _clear_tiles_dir() -> None:
    tiles_dir = Path("./output/tiles")
    if not tiles_dir.exists():
        return
    tiles_root = tiles_dir.resolve()
    for path in tiles_dir.iterdir():
        resolved = path.resolve()
        if not str(resolved).lower().startswith(str(tiles_root).lower()):
            raise RuntimeError(f"Refusing to delete outside tiles directory: {resolved}")
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    print(f"[CLEAN] Removed intermediate tile files from {tiles_dir}")


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
                print(f"[PROGRESS] {i}/{len(stl_files)} {p.name}")

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
            print(f"[PROGRESS] {i}/{len(stl_files)} {p.name}")

        out.write(f"endsolid {solid_name}\n")

    print(f"[MERGE] Done. Wrote {out_stl} with ~{total_tris_streamed:,} triangles.")


def _iter_stl_triangles(path: Path):
    """
    Yield triangles from an STL file as ((ax, ay, az), (bx, by, bz), (cx, cy, cz)).
    Uses a streaming reader to avoid loading the full mesh into memory.
    """
    if _is_probably_binary_stl(path):
        with path.open("rb") as f:
            f.seek(80)
            n = int.from_bytes(f.read(4), "little", signed=False)
            for _ in range(n):
                rec = f.read(50)
                if len(rec) != 50:
                    break
                vals = struct.unpack("<12fH", rec)
                yield (
                    (float(vals[3]), float(vals[4]), float(vals[5])),
                    (float(vals[6]), float(vals[7]), float(vals[8])),
                    (float(vals[9]), float(vals[10]), float(vals[11])),
                )
        return

    tri: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("vertex"):
                parts = s.split()
                if len(parts) >= 4:
                    tri.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    if len(tri) == 3:
                        yield (tri[0], tri[1], tri[2])
                        tri = []


def _read_last_scale_info(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("scale_xy")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _geometry_search_roots() -> List[Path]:
    base = Path(__file__).resolve().parent
    roots = [base / GEOMETRY_DATA_DIRNAME, base / "borders"]
    seen = set()
    unique: List[Path] = []
    for path in roots:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _is_border_shapefile(path: Path) -> bool:
    name = path.name.upper()
    return any(token in name for token in ("LANDES", "KANTON", "BEZIRK"))


def _is_lake_shapefile(path: Path) -> bool:
    name = path.name.upper()
    return "LAKE_POLYGONS" in name or ("GEWAESSER" in name and "STEHENDES" in name)


def _is_river_shapefile(path: Path) -> bool:
    return "FLIESSGEWAESSER" in path.name.upper()


def _is_bridge_candidate_shapefile(path: Path) -> bool:
    name = path.name.upper()
    return any(token in name for token in ("BRIDGE_PROTECTION", "BRUECKEN", "BRUCKEN", "STRASSE", "EISENBAHN"))


def _default_border_shp() -> Optional[Path]:
    shp_files: List[Path] = []
    for root in _geometry_search_roots():
        if root.exists():
            shp_files.extend(p for p in root.rglob("*.shp") if p.is_file() and _is_border_shapefile(p))
    if not shp_files:
        return None
    preferred = [p for p in shp_files if any(token in p.name.upper() for token in ("LANDESGRENZE", "LANDESGEBIET"))]
    return sorted(preferred or shp_files)[0]


def _default_lake_shp() -> Optional[Path]:
    shp_files: List[Path] = []
    for root in _geometry_search_roots():
        if root.exists():
            shp_files.extend(p for p in root.rglob("*.shp") if p.is_file() and _is_lake_shapefile(p))
    if not shp_files:
        return None
    preferred = [p for p in shp_files if "LAKE_POLYGONS" in p.name.upper()]
    return sorted(preferred or shp_files)[0]


def _default_river_shp() -> Optional[Path]:
    shp_files: List[Path] = []
    for root in _geometry_search_roots():
        if root.exists():
            shp_files.extend(p for p in root.rglob("*.shp") if p.is_file() and _is_river_shapefile(p))
    if not shp_files:
        return None
    return sorted(shp_files)[0]


def _default_bridge_shps() -> List[Path]:
    shp_files: List[Path] = []
    for root in _geometry_search_roots():
        if root.exists():
            shp_files.extend(p for p in root.rglob("*.shp") if p.is_file() and _is_bridge_candidate_shapefile(p))
    preferred = [
        p for p in shp_files
        if any(token in p.name.upper() for token in ("BRIDGE_PROTECTION", "BRUECKEN", "BRUCKEN"))
    ]
    return sorted(preferred or shp_files)


def _parse_border_scale(raw: str, tiles_dir: Path) -> float:
    value = raw.strip().lower()
    if value in {"", "auto"}:
        scale = _read_last_scale_info(tiles_dir / "scale_info.json")
        if scale is None:
            print("[WARN] Border scale set to auto but no scale_info.json found; using 1.0")
            return 1.0
        print(f"[BORDER] Using stored tile scale {scale:.6f}")
        return float(scale)
    if ":" in value or "/" in value:
        ratio = _parse_scale_ratio(value)
        return 1000.0 / float(ratio)
    return float(value)


def _read_last_z_scale(path: Path) -> float:
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1.0
    try:
        return float(data.get("z_scale", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _parse_border_keep_list(raw: str) -> List[str]:
    if not raw:
        return []
    items = [part.strip() for part in raw.split(",")]
    return [item for item in items if item]


def _parse_water_features(raw: str) -> Tuple[bool, bool]:
    value = raw.strip().lower()
    if value in {"", "all", "both", "lakes,rivers", "rivers,lakes"}:
        return (True, True)
    if value in {"none", "off"}:
        return (False, False)

    tokens = {part.strip().lower() for part in value.split(",") if part.strip()}
    aliases = {
        "lake": "lakes",
        "lakes": "lakes",
        "standing": "lakes",
        "standing-water": "lakes",
        "standing_water": "lakes",
        "river": "rivers",
        "rivers": "rivers",
        "flowing": "rivers",
        "flowing-water": "rivers",
        "flowing_water": "rivers",
    }
    normalized = set()
    invalid = []
    for token in tokens:
        mapped = aliases.get(token)
        if mapped is None:
            invalid.append(token)
        else:
            normalized.add(mapped)
    if invalid:
        raise ValueError(f"Unknown water feature value(s): {', '.join(sorted(invalid))}")
    return ("lakes" in normalized, "rivers" in normalized)


def _parse_water_feature_ids(raw: str) -> Optional[set[str]]:
    value = raw.strip()
    if not value or value.lower() in {"all", "*"}:
        return None
    ids = {part.strip().lower() for part in value.split(",") if part.strip()}
    invalid = [item for item in ids if ":" not in item]
    if invalid:
        raise ValueError(f"Water feature IDs must look like lake:0 or river:3. Invalid: {', '.join(sorted(invalid))}")
    return ids


def _guess_border_label_field(field_defs, *, border_hint: str = "") -> Optional[str]:
    field_names = [name for name, _ftype, _len, _dec in field_defs]
    string_fields = [name for name, ftype, _len, _dec in field_defs if ftype in {"C", "M"}]
    upper_names = {name.upper(): name for name in field_names}
    hint = border_hint.upper()

    if "BEZIRK" in hint:
        preferred = [
            "BEZIRKSNAME",
            "BEZIRK",
            "NAME",
            "NAME_DE",
            "NAME_FR",
            "NAME_IT",
            "NAME_EN",
        ]
    elif "KANTON" in hint:
        preferred = [
            "KANTONSNAME",
            "KANTON",
            "NAME",
            "NAME_DE",
            "NAME_FR",
            "NAME_IT",
            "NAME_EN",
        ]
    else:
        preferred = [
            "NAME",
            "NAME_DE",
            "NAME_FR",
            "NAME_IT",
            "NAME_EN",
        ]

    for candidate in preferred:
        if candidate in upper_names:
            return upper_names[candidate]

    for name in field_names:
        if "NAME" in name.upper():
            return name

    if string_fields:
        return string_fields[0]

    return None


def _load_border_geometry(
    shp_path: Path,
    *,
    keep_values: Optional[List[str]] = None,
    keep_field: Optional[str] = None,
):
    try:
        import shapefile  # pyshp
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Border clipping requires 'pyshp' and 'shapely'. "
            "Install with: pip install pyshp shapely"
        ) from exc

    if not shp_path.exists():
        raise FileNotFoundError(f"Border shapefile not found: {shp_path}")

    reader = shapefile.Reader(str(shp_path))
    field_defs = [f for f in reader.fields if f[0] != "DeletionFlag"]
    field_names = [name for name, _ftype, _len, _dec in field_defs]

    keep_set = set(v.strip().lower() for v in (keep_values or []) if v.strip())
    label_field = None
    if keep_set:
        if keep_field:
            matches = [name for name in field_names if name.upper() == keep_field.strip().upper()]
            if not matches:
                raise ValueError(
                    f"Border field '{keep_field}' not found in {shp_path.name}. "
                    f"Available fields: {', '.join(field_names)}"
                )
            label_field = matches[0]
        else:
            label_field = _guess_border_label_field(field_defs, border_hint=shp_path.name)
        if not label_field:
            raise ValueError(f"Could not find a name field in {shp_path.name} for --border-keep.")

    geoms = []
    for shape_record in reader.iterShapeRecords():
        geom = shapely_shape(shape_record.shape.__geo_interface__)
        if geom.is_empty:
            continue
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        if keep_set:
            record_dict = dict(zip(field_names, shape_record.record))
            value = record_dict.get(label_field)
            if value is None:
                continue
            if str(value).strip().lower() not in keep_set:
                continue
        geoms.append(geom)

    if not geoms:
        if keep_set:
            raise ValueError(
                f"No polygon geometries matched --border-keep in {shp_path}. "
                f"Field: {label_field}"
            )
        raise ValueError(f"No polygon geometries found in {shp_path}")

    return unary_union(geoms)


def _scale_border_geometry(geom, scale: float):
    if scale == 1.0:
        return geom
    try:
        from shapely.affinity import scale as shapely_scale
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Border clipping requires 'shapely'. Install with: pip install shapely"
        ) from exc
    return shapely_scale(geom, xfact=scale, yfact=scale, origin=(0.0, 0.0))


def _normalize_token(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip().lower()


def _bounds_intersect(
    a_min_x: float,
    a_min_y: float,
    a_max_x: float,
    a_max_y: float,
    b_min_x: float,
    b_min_y: float,
    b_max_x: float,
    b_max_y: float,
) -> bool:
    return not (
        a_max_x < b_min_x or
        a_min_x > b_max_x or
        a_max_y < b_min_y or
        a_min_y > b_max_y
    )


def _lake_shape_parts(shape_obj) -> Tuple[List[object], List[object]]:
    try:
        from shapely.geometry import LineString, shape as shapely_shape
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Lake lowering requires 'shapely'. Install with: pip install shapely"
        ) from exc

    shape_type = int(getattr(shape_obj, "shapeType", 0))
    if shape_type in {5, 15, 25, 31}:  # Polygon / PolygonZ / PolygonM / MultiPatch-ish polygonal records
        geom = shapely_shape(shape_obj.__geo_interface__)
        if geom.is_empty:
            return ([], [])
        if geom.geom_type == "Polygon":
            return ([geom], [])
        if geom.geom_type == "MultiPolygon":
            return ([g for g in geom.geoms if not g.is_empty], [])
        return ([], [])

    if shape_type not in {3, 13, 23}:  # PolyLine / PolyLineZ / PolyLineM
        return ([], [])

    points = list(getattr(shape_obj, "points", []) or [])
    if not points:
        return ([], [])
    parts = list(getattr(shape_obj, "parts", []) or [0])
    if not parts:
        parts = [0]
    parts = parts + [len(points)]

    lines: List[object] = []
    for start, end in zip(parts, parts[1:]):
        segment = [(float(pt[0]), float(pt[1])) for pt in points[start:end]]
        if len(segment) < 2:
            continue
        line = LineString(segment)
        if line.is_empty or line.length <= 0.0:
            continue
        lines.append(line)
    return ([], lines)


def _water_record_name(field_names: List[str], record: object) -> str:
    record_dict = dict(zip(field_names, record))
    for field in ("NAME", "GEW_NAME"):
        value = record_dict.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _water_record_fallback_label(field_names: List[str], record: object, prefix: str) -> str:
    record_dict = dict(zip(field_names, record))
    for field in ("GEWISS_NR", "GEW_LAUF_U", "GEW_NAME_U", "UUID", "OBJEKTART"):
        value = record_dict.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f"Unnamed {prefix} {text}"
    return f"Unnamed {prefix}"


def _best_water_name(names: List[str], prefix: str) -> str:
    named = [name for name in names if name and not name.lower().startswith(f"unnamed {prefix}")]
    if named:
        return Counter(named).most_common(1)[0][0]
    unnamed = [name for name in names if name]
    if unnamed:
        return Counter(unnamed).most_common(1)[0][0]
    return f"Unnamed {prefix}"


def _water_label_fields(reader) -> List[str]:
    wanted = {"GROUP_KEY", "NAME", "GEW_NAME", "GEW_NAME_U", "GEW_LAUF_U", "GEWISS_NR", "OBJEKTART", "UUID"}
    return [field[0] for field in reader.fields if field[0] != "DeletionFlag" and field[0] in wanted]


def _water_group_key(field_names: List[str], record: object) -> str:
    record_dict = dict(zip(field_names, record))
    for field in ("GROUP_KEY", "GEWISS_NR", "GEW_LAUF_U", "GEW_NAME_U", "NAME", "UUID"):
        value = record_dict.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f"{field}:{text}"
    return ""


def _name_for_polygonized_lake(geom, line_features: List[Tuple[object, str]]) -> str:
    names = []
    boundary = geom.boundary.buffer(0.001)
    for line, name in line_features:
        if name and line.intersects(boundary):
            names.append(name)
    if not names:
        names = [name for _line, name in line_features if name]
    return _best_water_name(names, "lake")


def _load_lake_features_for_bounds(
    shp_path: Path,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    keep_ids: Optional[set[str]] = None,
):
    try:
        import shapefile  # pyshp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Lake lowering requires 'pyshp' and 'shapely'. "
            "Install with: pip install pyshp shapely"
        ) from exc

    if not shp_path.exists():
        raise FileNotFoundError(f"Lake shapefile not found: {shp_path}")

    reader = shapefile.Reader(str(shp_path), encoding="latin1", encodingErrors="replace")
    field_names = _water_label_fields(reader)
    features = []
    polygon_groups: Dict[str, Dict[str, object]] = {}
    touched_keys = set()
    for shape_record in reader.iterShapeRecords(fields=field_names, bbox=(min_x, min_y, max_x, max_y)):
        shape_bbox = getattr(shape_record.shape, "bbox", None)
        if shape_bbox and len(shape_bbox) >= 4:
            if not _bounds_intersect(
                float(shape_bbox[0]),
                float(shape_bbox[1]),
                float(shape_bbox[2]),
                float(shape_bbox[3]),
                min_x,
                min_y,
                max_x,
                max_y,
            ):
                continue
        name = _water_record_name(field_names, shape_record.record) or _water_record_fallback_label(
            field_names,
            shape_record.record,
            "lake",
        )
        group_key = _water_group_key(field_names, shape_record.record) or f"name:{name}"
        shape_polys, shape_lines = _lake_shape_parts(shape_record.shape)
        if shape_lines and group_key:
            touched_keys.add(group_key)
        for geom in shape_polys:
            geom_bounds = geom.bounds
            if not _bounds_intersect(
                float(geom_bounds[0]),
                float(geom_bounds[1]),
                float(geom_bounds[2]),
                float(geom_bounds[3]),
                min_x,
                min_y,
                max_x,
                max_y,
            ):
                continue
            if group_key not in polygon_groups:
                polygon_groups[group_key] = {"geoms": [], "names": []}
            polygon_groups[group_key]["geoms"].append(geom)  # type: ignore[union-attr]
            polygon_groups[group_key]["names"].append(name)  # type: ignore[union-attr]
    if polygon_groups:
        try:
            from shapely.ops import unary_union
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Lake lowering requires 'shapely'. Install with: pip install shapely"
            ) from exc
        for data in polygon_groups.values():
            geoms = data["geoms"]
            names = [name for name in data["names"] if name]  # type: ignore[index]
            geom = geoms[0] if len(geoms) == 1 else unary_union(geoms)  # type: ignore[index]
            features.append((geom, _best_water_name(names, "lake")))
        selected = []
        for idx, feature in enumerate(features):
            if keep_ids is None or f"lake:{idx}" in keep_ids:
                selected.append(feature)
        return selected
    if not touched_keys:
        return []

    try:
        from shapely.ops import polygonize, unary_union
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Lake lowering requires 'shapely'. Install with: pip install shapely"
        ) from exc

    def add_polygonized(line_features: List[Tuple[object, str]]) -> None:
        nonlocal features
        polygon_idx = 0
        lines = [line for line, _name in line_features]
        merged_lines = unary_union(lines)
        for geom in polygonize(merged_lines):
            geom_bounds = geom.bounds
            if not _bounds_intersect(
                float(geom_bounds[0]),
                float(geom_bounds[1]),
                float(geom_bounds[2]),
                float(geom_bounds[3]),
                min_x,
                min_y,
                max_x,
                max_y,
            ):
                continue
            if keep_ids is None or f"lake:{polygon_idx}" in keep_ids:
                features.append((geom, _name_for_polygonized_lake(geom, line_features)))
            polygon_idx += 1

    full_lake_line_features = []
    for shape_record in reader.iterShapeRecords(fields=field_names):
        group_key = _water_group_key(field_names, shape_record.record)
        if group_key not in touched_keys:
            continue
            name = _water_record_name(field_names, shape_record.record) or _water_record_fallback_label(
                field_names,
                shape_record.record,
                "lake",
            )
        _shape_polys, shape_lines = _lake_shape_parts(shape_record.shape)
        full_lake_line_features.extend((line, name) for line in shape_lines)

    if full_lake_line_features:
        add_polygonized(full_lake_line_features)
    return features


def _load_lake_geometries_for_bounds(
    shp_path: Path,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    keep_ids: Optional[set[str]] = None,
):
    return [
        geom
        for geom, _name in _load_lake_features_for_bounds(
            shp_path,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            keep_ids=keep_ids,
        )
    ]


def _load_all_lake_geometries(shp_path: Path):
    try:
        import shapefile  # pyshp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Lake lowering requires 'pyshp' and 'shapely'. "
            "Install with: pip install pyshp shapely"
        ) from exc

    if not shp_path.exists():
        raise FileNotFoundError(f"Lake shapefile not found: {shp_path}")

    reader = shapefile.Reader(str(shp_path))
    geoms = []
    lines = []
    for shape_obj in reader.iterShapes():
        shape_polys, shape_lines = _lake_shape_parts(shape_obj)
        geoms.extend(shape_polys)
        lines.extend(shape_lines)
    if geoms or not lines:
        return geoms

    try:
        from shapely.ops import polygonize, unary_union
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Lake lowering requires 'shapely'. Install with: pip install shapely"
        ) from exc

    merged_lines = unary_union(lines)
    geoms.extend(list(polygonize(merged_lines)))
    return geoms


def _load_all_water_lines(shp_path: Path):
    try:
        import shapefile  # pyshp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "River detection requires 'pyshp' and 'shapely'. "
            "Install with: pip install pyshp shapely"
        ) from exc

    if not shp_path.exists():
        raise FileNotFoundError(f"River shapefile not found: {shp_path}")

    reader = shapefile.Reader(str(shp_path))
    lines = []
    for shape_obj in reader.iterShapes():
        _shape_polys, shape_lines = _lake_shape_parts(shape_obj)
        lines.extend(shape_lines)
    return lines


def _load_water_line_features_for_bounds(
    shp_path: Path,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    keep_ids: Optional[set[str]] = None,
):
    try:
        import shapefile  # pyshp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "River detection requires 'pyshp' and 'shapely'. "
            "Install with: pip install pyshp shapely"
        ) from exc

    if not shp_path.exists():
        raise FileNotFoundError(f"River shapefile not found: {shp_path}")

    reader = shapefile.Reader(str(shp_path), encoding="latin1", encodingErrors="replace")
    field_names = _water_label_fields(reader)
    grouped: Dict[str, Dict[str, object]] = {}
    for shape_record in reader.iterShapeRecords(fields=field_names, bbox=(min_x, min_y, max_x, max_y)):
        shape_bbox = getattr(shape_record.shape, "bbox", None)
        if shape_bbox and len(shape_bbox) >= 4:
            if not _bounds_intersect(
                float(shape_bbox[0]),
                float(shape_bbox[1]),
                float(shape_bbox[2]),
                float(shape_bbox[3]),
                min_x,
                min_y,
                max_x,
                max_y,
            ):
                continue
        group_key = _water_group_key(field_names, shape_record.record)
        name = _water_record_name(field_names, shape_record.record) or _water_record_fallback_label(
            field_names,
            shape_record.record,
            "river",
        )
        _shape_polys, shape_lines = _lake_shape_parts(shape_record.shape)
        if not shape_lines:
            continue
        if not group_key:
            group_key = f"record:{len(grouped)}"
        if group_key not in grouped:
            grouped[group_key] = {"lines": [], "names": []}
        grouped[group_key]["lines"].extend(shape_lines)  # type: ignore[union-attr]
        grouped[group_key]["names"].append(name)  # type: ignore[union-attr]
    features = []
    for group_idx, data in enumerate(grouped.values()):
        if keep_ids is not None and f"river:{group_idx}" not in keep_ids:
            continue
        lines = data["lines"]
        names = [name for name in data["names"] if name]  # type: ignore[index]
        name = _best_water_name(names, "river")
        if len(lines) == 1:
            geom = lines[0]
        else:
            try:
                from shapely.ops import linemerge, unary_union
                geom = linemerge(unary_union(lines))
            except Exception:
                from shapely.geometry import MultiLineString
                geom = MultiLineString(lines)
        features.append((geom, name))
    return features


def _load_water_lines_for_bounds(
    shp_path: Path,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    keep_ids: Optional[set[str]] = None,
):
    return [
        line
        for line, _name in _load_water_line_features_for_bounds(
            shp_path,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            keep_ids=keep_ids,
        )
    ]


def _is_bridge_record(record_dict: Dict[str, object]) -> bool:
    for key, value in record_dict.items():
        key_norm = _normalize_token(key)
        value_norm = _normalize_token(value)
        if key_norm == "kunstbaute" and any(token in value_norm for token in ("bruecke", "brucke", "bridge")):
            return True
    return False


def _load_bridge_geometries(
    shp_paths: List[Path],
    *,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    max_x: Optional[float] = None,
    max_y: Optional[float] = None,
):
    try:
        import shapefile  # pyshp
        from shapely.geometry import shape as shapely_shape
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Bridge protection requires 'pyshp' and 'shapely'. "
            "Install with: pip install pyshp shapely"
        ) from exc

    geoms = []
    for shp_path in shp_paths:
        if not shp_path.exists():
            continue
        reader = shapefile.Reader(str(shp_path), encoding="latin1", encodingErrors="replace")
        field_names = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
        for shape_record in reader.iterShapeRecords():
            if min_x is not None and min_y is not None and max_x is not None and max_y is not None:
                shape_bbox = getattr(shape_record.shape, "bbox", None)
                if shape_bbox and len(shape_bbox) >= 4:
                    if not _bounds_intersect(
                        float(shape_bbox[0]),
                        float(shape_bbox[1]),
                        float(shape_bbox[2]),
                        float(shape_bbox[3]),
                        float(min_x),
                        float(min_y),
                        float(max_x),
                        float(max_y),
                    ):
                        continue
            record_dict = dict(zip(field_names, shape_record.record))
            if not _is_bridge_record(record_dict):
                continue
            geom = shapely_shape(shape_record.shape.__geo_interface__)
            if not geom.is_empty:
                geoms.append(geom)
    return geoms


def _source_bounds_for_model_bounds(model_bounds: Tuple[float, float, float, float], scale: float) -> Tuple[float, float, float, float]:
    if scale <= 0.0:
        raise ValueError("Water scale must be > 0.")
    min_x, min_y, max_x, max_y = model_bounds
    if scale == 1.0:
        return (float(min_x), float(min_y), float(max_x), float(max_y))
    return (
        float(min_x) / float(scale),
        float(min_y) / float(scale),
        float(max_x) / float(scale),
        float(max_y) / float(scale),
    )


def _load_riverbank_polygons_for_lines(
    geojson_path: Path,
    river_lines: List[object],
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> List[object]:
    try:
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("River outlines require 'shapely'.") from exc

    try:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read riverbank GeoJSON {geojson_path}: {exc}") from exc

    selected_lines = unary_union(river_lines)
    polygons: List[object] = []
    for feature in payload.get("features", []):
        geometry_data = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry_data:
            continue
        geom = shapely_shape(geometry_data)
        if geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        bounds = geom.bounds
        if not _bounds_intersect(bounds[0], bounds[1], bounds[2], bounds[3], min_x, min_y, max_x, max_y):
            continue
        if not geom.intersects(selected_lines):
            continue
        if geom.geom_type == "Polygon":
            polygons.append(geom)
        else:
            polygons.extend(part for part in geom.geoms if not part.is_empty)
    return polygons


def _prepare_water_geometries(
    *,
    lake_shp: Optional[Path],
    river_shp: Optional[Path],
    riverbank_geojson: Optional[Path],
    bridge_shps: List[Path],
    water_scale: float,
    bridge_buffer_mm: float,
    model_bounds: Tuple[float, float, float, float],
    water_feature_ids: Optional[set[str]] = None,
):
    if water_scale <= 0.0:
        raise ValueError("Water scale must be > 0.")

    water_geoms = []
    bridge_geoms = []
    src_min_x, src_min_y, src_max_x, src_max_y = _source_bounds_for_model_bounds(model_bounds, float(water_scale))

    if lake_shp is not None:
        lake_geoms = _load_lake_geometries_for_bounds(
            lake_shp,
            min_x=src_min_x,
            min_y=src_min_y,
            max_x=src_max_x,
            max_y=src_max_y,
            keep_ids=water_feature_ids,
        )
        for geom in lake_geoms:
            water_geoms.append(_scale_border_geometry(geom, water_scale) if water_scale != 1.0 else geom)
        print(f"[WATER] Loaded {len(lake_geoms):,} lake polygon feature(s) for model bounds")

    if river_shp is not None:
        river_lines = _load_water_lines_for_bounds(
            river_shp,
            min_x=src_min_x,
            min_y=src_min_y,
            max_x=src_max_x,
            max_y=src_max_y,
            keep_ids=water_feature_ids,
        )
        if riverbank_geojson is None or not riverbank_geojson.exists():
            raise FileNotFoundError(
                "Actual river outlines are required but no riverbank GeoJSON was found. "
                "Run download_riverbanks.py with the SwissALTI CSV first."
            )
        river_polygons = _load_riverbank_polygons_for_lines(
            riverbank_geojson,
            river_lines,
            min_x=src_min_x,
            min_y=src_min_y,
            max_x=src_max_x,
            max_y=src_max_y,
        )
        if not river_polygons:
            raise RuntimeError(
                "No actual riverbank polygons matched the selected river centerlines. "
                "Leave rivers unchecked or choose an area with mapped OSM river outlines."
            )
        for geom in river_polygons:
            water_geoms.append(_scale_border_geometry(geom, water_scale) if water_scale != 1.0 else geom)
        print(
            f"[WATER] Loaded {len(river_polygons):,} actual riverbank polygon(s) "
            f"intersecting {len(river_lines):,} selected river centerline feature(s)"
        )

    if river_shp is not None and bridge_shps:
        bridge_buffer = max(float(bridge_buffer_mm), 0.0) / 2.0
        if bridge_buffer > 0.0:
            raw_bridges = _load_bridge_geometries(
                bridge_shps,
                min_x=src_min_x,
                min_y=src_min_y,
                max_x=src_max_x,
                max_y=src_max_y,
            )
            for geom in raw_bridges:
                geom_scaled = _scale_border_geometry(geom, water_scale) if water_scale != 1.0 else geom
                bridge_geoms.append(geom_scaled.buffer(bridge_buffer, cap_style=2, join_style=2))
            print(
                f"[WATER] Loaded {len(raw_bridges):,} bridge protection feature(s) "
                f"with {float(bridge_buffer_mm):.3f} mm buffer"
            )

    return water_geoms, bridge_geoms


def _default_building_paths() -> List[Path]:
    paths: List[Path] = []
    for root in _geometry_search_roots():
        if not root.exists():
            continue
        for suffix in ("*.gml", "*.xml"):
            paths.extend(p for p in root.rglob(suffix) if p.is_file())
    return sorted(paths)


def _building_input_paths(raw_paths: Optional[List[Path]]) -> List[Path]:
    if not raw_paths:
        return _default_building_paths()
    out: List[Path] = []
    for path in raw_paths:
        if path.is_dir():
            for suffix in ("*.gml", "*.xml"):
                out.extend(p for p in path.rglob(suffix) if p.is_file())
        elif path.is_file():
            out.append(path)
    return sorted(set(out))


def _parse_gml_poslist(text: str, dimension: int) -> List[Tuple[float, float, float]]:
    values = [float(part) for part in text.split()]
    if dimension < 2:
        dimension = 3
    coords = []
    for idx in range(0, len(values) - dimension + 1, dimension):
        x = values[idx]
        y = values[idx + 1]
        z = values[idx + 2] if dimension >= 3 else 0.0
        coords.append((float(x), float(y), float(z)))
    return coords


def _polygon_coords_from_gml(poly) -> List[Tuple[float, float, float]]:
    pos_lists = []
    for elem in poly.iter():
        if elem.tag.endswith("posList") and elem.text and elem.text.strip():
            pos_lists.append(elem)
    if pos_lists:
        elem = pos_lists[0]
        try:
            dimension = int(elem.attrib.get("srsDimension", "3"))
        except ValueError:
            dimension = 3
        coords = _parse_gml_poslist(elem.text or "", dimension)
    else:
        coords = []
        for elem in poly.iter():
            if elem.tag.endswith("pos") and elem.text and elem.text.strip():
                parsed = _parse_gml_poslist(elem.text, 3)
                if parsed:
                    coords.append(parsed[0])
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def _coords_intersect_bounds(coords: List[Tuple[float, float, float]], bounds: Tuple[float, float, float, float]) -> bool:
    if not coords:
        return False
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    return _bounds_intersect(min(xs), min(ys), max(xs), max(ys), *bounds)


def _clip_polygon_to_xy_bounds(
    coords: np.ndarray,
    bounds: Tuple[float, float, float, float],
) -> np.ndarray:
    """Clip a 3D polygon to an XY rectangle while interpolating boundary heights."""
    min_x, min_y, max_x, max_y = bounds

    def clip_edge(vertices: np.ndarray, axis: int, limit: float, keep_greater: bool) -> np.ndarray:
        if vertices.shape[0] == 0:
            return vertices
        result: List[np.ndarray] = []
        previous = vertices[-1]
        previous_inside = previous[axis] >= limit if keep_greater else previous[axis] <= limit
        for current in vertices:
            current_inside = current[axis] >= limit if keep_greater else current[axis] <= limit
            if current_inside != previous_inside:
                delta = current[axis] - previous[axis]
                if delta != 0.0:
                    ratio = (limit - previous[axis]) / delta
                    result.append(previous + ratio * (current - previous))
            if current_inside:
                result.append(current)
            previous = current
            previous_inside = current_inside
        return np.asarray(result, dtype=np.float64)

    clipped = np.asarray(coords, dtype=np.float64)
    for axis, limit, keep_greater in (
        (0, min_x, True),
        (0, max_x, False),
        (1, min_y, True),
        (1, max_y, False),
    ):
        clipped = clip_edge(clipped, axis, limit, keep_greater)
        if clipped.shape[0] < 3:
            return np.empty((0, 3), dtype=np.float64)

    # Avoid zero-area fan triangles when a boundary vertex is repeated.
    unique_vertices = [clipped[0]]
    for vertex in clipped[1:]:
        if not np.allclose(vertex, unique_vertices[-1], rtol=0.0, atol=1e-9):
            unique_vertices.append(vertex)
    clipped = np.asarray(unique_vertices, dtype=np.float64)
    if clipped.shape[0] > 1 and np.allclose(clipped[0], clipped[-1], rtol=0.0, atol=1e-9):
        clipped = clipped[:-1]
    return clipped if clipped.shape[0] >= 3 else np.empty((0, 3), dtype=np.float64)


def _iter_citygml_polygon_poslists(path: Path):
    """Yield exterior CityGML polygon coordinate text without constructing XML nodes."""
    closing_tag = b"</gml:Polygon>"
    poslist_re = re.compile(
        rb"<gml:posList(?P<attrs>[^>]*)>(?P<coords>.*?)</gml:posList>",
        re.DOTALL,
    )
    dimension_re = re.compile(rb"srsDimension=[\"'](?P<dimension>\d+)[\"']")
    buffer = b""
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk and not buffer:
                return
            buffer += chunk
            cursor = 0
            while True:
                end = buffer.find(closing_tag, cursor)
                if end < 0:
                    break
                block_end = end + len(closing_tag)
                block = buffer[cursor:block_end]
                cursor = block_end
                start = block.rfind(b"<gml:Polygon")
                if start < 0:
                    continue
                match = poslist_re.search(block, start)
                if match is None:
                    continue
                dimension_match = dimension_re.search(match.group("attrs"))
                dimension = int(dimension_match.group("dimension")) if dimension_match else 3
                bytes_scanned = source.tell() - len(buffer) + cursor
                yield match.group("coords"), dimension, bytes_scanned
            buffer = buffer[cursor:]
            if not chunk:
                return


def _building_cache_path(
    paths: List[Path],
    source_bounds: Tuple[float, float, float, float],
    xy_scale: float,
    z_scale: float,
    max_files: int,
    printable: bool,
    nozzle_mm: float,
) -> Path:
    digest = hashlib.sha256()
    digest.update(b"building-rectangle-clip-v2")
    digest.update(repr(tuple(round(value, 4) for value in source_bounds)).encode("ascii"))
    digest.update(repr((round(xy_scale, 10), round(z_scale, 10), max_files, printable, round(nozzle_mm, 4))).encode("ascii"))
    for path in paths[:max_files] if max_files > 0 else paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(repr((stat.st_size, stat.st_mtime_ns)).encode("ascii"))
    return Path("output") / "cache" / "buildings" / f"{digest.hexdigest()[:20]}.npz"


def _simplify_building_polygon_for_print(coords: np.ndarray, nozzle_mm: float) -> np.ndarray:
    """Remove geometry that is smaller than the printer can reproduce."""
    grid = float(nozzle_mm) * 0.5
    snapped = np.round(coords / grid) * grid
    keep = np.ones(snapped.shape[0], dtype=bool)
    keep[1:] = np.any(np.abs(snapped[1:] - snapped[:-1]) > 1e-9, axis=1)
    simplified = snapped[keep]
    if simplified.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64)

    # Remove points that do not change a face at the selected print resolution.
    previous = np.roll(simplified, 1, axis=0)
    following = np.roll(simplified, -1, axis=0)
    segment = following - previous
    length_squared = np.einsum("ij,ij->i", segment, segment)
    relative = simplified - previous
    fraction = np.divide(
        np.einsum("ij,ij->i", relative, segment),
        length_squared,
        out=np.zeros_like(length_squared),
        where=length_squared > 1e-12,
    )
    nearest = previous + np.clip(fraction, 0.0, 1.0)[:, None] * segment
    distances = np.linalg.norm(simplified - nearest, axis=1)
    nonessential = (length_squared > 1e-12) & (distances <= grid * 0.35)
    if np.count_nonzero(~nonessential) >= 3:
        simplified = simplified[~nonessential]
    if simplified.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64)

    triangles = np.cross(simplified[1:-1] - simplified[0], simplified[2:] - simplified[0])
    surface_area = float(np.linalg.norm(triangles, axis=1).sum() * 0.5)
    if surface_area < (float(nozzle_mm) ** 2) * 0.25:
        return np.empty((0, 3), dtype=np.float64)
    return simplified


def _simplify_mesh_for_print(mesh: Mesh, nozzle_mm: float) -> Mesh:
    """Snap the final mesh to printable resolution and remove collapsed facets."""
    grid = float(nozzle_mm) * 0.5
    if grid <= 0.0 or mesh.faces.size == 0:
        return mesh

    quantized = np.round(mesh.vertices / grid).astype(np.int64)
    unique_vertices, remap = np.unique(quantized, axis=0, return_inverse=True)
    faces = remap[mesh.faces]
    valid = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[valid]
    vertices = unique_vertices.astype(np.float64) * grid
    if faces.size:
        triangle_vectors = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        triangle_areas = np.linalg.norm(triangle_vectors, axis=1) * 0.5
        faces = faces[triangle_areas >= (float(nozzle_mm) ** 2) * 0.01]
    if faces.size:
        canonical_faces = np.sort(faces, axis=1)
        _, keep_indices = np.unique(canonical_faces, axis=0, return_index=True)
        faces = faces[np.sort(keep_indices)]
    print(
        f"[PRINT] Simplified final mesh for a {float(nozzle_mm):.2f} mm nozzle: "
        f"{mesh.vertices.shape[0]:,} vertices / {mesh.faces.shape[0]:,} triangles -> "
        f"{vertices.shape[0]:,} vertices / {faces.shape[0]:,} triangles"
    )
    return Mesh(vertices=vertices, faces=faces.astype(np.int64, copy=False))


def _citygml_polygon_chunk_ranges(path: Path, target_bytes: int = 128 * 1024 * 1024) -> List[Tuple[int, int]]:
    """Split a CityGML file only after complete Polygon elements."""
    closing_tag = b"</gml:Polygon>"
    ranges: List[Tuple[int, int]] = []
    range_start = 0
    buffer = b""
    buffer_offset = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            search_from = 0
            while True:
                end = buffer.find(closing_tag, search_from)
                if end < 0:
                    break
                boundary = buffer_offset + end + len(closing_tag)
                if boundary - range_start >= target_bytes:
                    ranges.append((range_start, boundary))
                    range_start = boundary
                search_from = end + len(closing_tag)
            keep_bytes = min(len(closing_tag) - 1, len(buffer))
            buffer_offset += len(buffer) - keep_bytes
            buffer = buffer[-keep_bytes:]
    file_size = path.stat().st_size
    if range_start < file_size:
        ranges.append((range_start, file_size))
    return ranges


def _citygml_chunk_mesh_worker(task: tuple) -> tuple[str | None, int, int, int, int]:
    path_text, start, end, source_bounds, xy_scale, z_scale, printable, nozzle_mm, output_path = task
    path = Path(path_text)
    with path.open("rb") as source:
        source.seek(start)
        content = source.read(end - start)
    polygon_re = re.compile(rb"<gml:Polygon\b.*?</gml:Polygon>", re.DOTALL)
    poslist_re = re.compile(rb"<gml:posList(?P<attrs>[^>]*)>(?P<coords>.*?)</gml:posList>", re.DOTALL)
    dimension_re = re.compile(rb"srsDimension=[\"'](?P<dimension>\d+)[\"']")
    vertex_blocks: List[np.ndarray] = []
    face_blocks: List[np.ndarray] = []
    vertex_count = 0
    polygons_seen = polygons_used = polygons_clipped = polygons_print_filtered = 0
    for polygon in polygon_re.finditer(content):
        polygons_seen += 1
        match = poslist_re.search(polygon.group(0))
        if match is None:
            continue
        dimension_match = dimension_re.search(match.group("attrs"))
        dimension = int(dimension_match.group("dimension")) if dimension_match else 3
        values = np.fromstring(match.group("coords"), dtype=np.float64, sep=" ")
        if dimension < 2 or values.size < dimension * 3 or values.size % dimension:
            continue
        coords = values.reshape((-1, dimension))
        if dimension == 2:
            coords = np.column_stack((coords, np.zeros(coords.shape[0], dtype=np.float64)))
        else:
            coords = coords[:, :3]
        if coords.shape[0] > 1 and np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]
        if coords.shape[0] < 3 or not _bounds_intersect(
            float(coords[:, 0].min()), float(coords[:, 1].min()),
            float(coords[:, 0].max()), float(coords[:, 1].max()), *source_bounds,
        ):
            continue
        was_clipped = bool(
            np.any(coords[:, 0] < source_bounds[0]) or np.any(coords[:, 0] > source_bounds[2])
            or np.any(coords[:, 1] < source_bounds[1]) or np.any(coords[:, 1] > source_bounds[3])
        )
        coords = _clip_polygon_to_xy_bounds(coords, source_bounds)
        if coords.shape[0] < 3:
            continue
        if was_clipped:
            polygons_clipped += 1
        scaled = coords * np.asarray((xy_scale, xy_scale, z_scale), dtype=np.float64)
        if printable:
            scaled = _simplify_building_polygon_for_print(scaled, nozzle_mm)
            if scaled.shape[0] < 3:
                polygons_print_filtered += 1
                continue
        indices = np.arange(1, scaled.shape[0] - 1, dtype=np.int64)
        face_blocks.append(np.column_stack((
            np.full(indices.shape[0], vertex_count, dtype=np.int64),
            vertex_count + indices,
            vertex_count + indices + 1,
        )))
        vertex_blocks.append(scaled)
        vertex_count += scaled.shape[0]
        polygons_used += 1
    if not face_blocks:
        return None, polygons_seen, polygons_used, polygons_clipped, polygons_print_filtered
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vertices=np.vstack(vertex_blocks).astype(np.float32), faces=np.vstack(face_blocks).astype(np.int32))
    return str(out), polygons_seen, polygons_used, polygons_clipped, polygons_print_filtered


def _citygml_buildings_to_mesh_parallel(
    paths: List[Path],
    *,
    source_bounds: Tuple[float, float, float, float],
    xy_scale: float,
    z_scale: float,
    max_files: int,
    printable: bool,
    nozzle_mm: float,
    workers: int,
    temporary_dir: Path,
) -> Optional[Mesh]:
    selected_paths = paths[:max_files] if max_files > 0 else paths
    vertex_blocks: List[np.ndarray] = []
    face_blocks: List[np.ndarray] = []
    vertex_count = 0
    polygons_seen = polygons_used = polygons_clipped = polygons_print_filtered = 0
    temporary_dir.mkdir(parents=True, exist_ok=True)
    for file_index, path in enumerate(selected_paths, 1):
        ranges = _citygml_polygon_chunk_ranges(path)
        if len(ranges) < 2:
            return _citygml_buildings_to_mesh(
                paths,
                source_bounds=source_bounds,
                xy_scale=xy_scale,
                z_scale=z_scale,
                max_files=max_files,
                printable=printable,
                nozzle_mm=nozzle_mm,
                workers=1,
                temporary_dir=None,
            )
        print(f"[BUILDINGS] Parallel scan: {path.name}, {len(ranges)} polygon-safe chunks, {workers} workers")
        tasks = [
            (
                str(path), start, end, source_bounds, xy_scale, z_scale, printable, nozzle_mm,
                str(temporary_dir / f"{file_index:03d}_{chunk_index:04d}.npz"),
            )
            for chunk_index, (start, end) in enumerate(ranges, 1)
        ]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_citygml_chunk_mesh_worker, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                result_path, seen, used, clipped, print_filtered = future.result()
                polygons_seen += seen
                polygons_used += used
                polygons_clipped += clipped
                polygons_print_filtered += print_filtered
                if result_path:
                    chunk_path = Path(result_path)
                    with np.load(chunk_path, allow_pickle=False) as data:
                        vertices = np.asarray(data["vertices"], dtype=np.float64)
                        faces = np.asarray(data["faces"], dtype=np.int64)
                    face_blocks.append(faces + vertex_count)
                    vertex_blocks.append(vertices)
                    vertex_count += vertices.shape[0]
                    chunk_path.unlink(missing_ok=True)
                print(f"[BUILDINGS_PROGRESS] {completed}/{len(tasks)} {path.name} chunks complete")
    if not face_blocks:
        print(f"[BUILDINGS] No CityGML building polygons intersected the model bounds across {len(selected_paths):,} file(s).")
        return None
    mesh = Mesh(vertices=np.vstack(vertex_blocks), faces=np.vstack(face_blocks))
    print(
        f"[BUILDINGS] Loaded {polygons_used:,} CityGML polygon surface(s) from {len(selected_paths):,} file(s): "
        f"{mesh.faces.shape[0]:,} triangles; clipped {polygons_clipped:,} polygon(s) to model bounds"
    )
    if printable:
        print(
            f"[BUILDINGS] Print simplification used a {float(nozzle_mm):.2f} mm nozzle; "
            f"discarded {polygons_print_filtered:,} sub-nozzle surface(s)."
        )
    return mesh


def _load_building_mesh_cache(path: Path) -> Optional[Mesh]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            vertices = np.asarray(cached["vertices"], dtype=np.float64)
            faces = np.asarray(cached["faces"], dtype=np.int64)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[BUILDINGS] Ignoring invalid cached mesh {path.name}: {exc}")
        return None
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return None
    print(f"[BUILDINGS] Reusing clipped building cache: {path.name}")
    return Mesh(vertices=vertices, faces=faces)


def _save_building_mesh_cache(path: Path, mesh: Mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
    )
    for stale_path in path.parent.glob("*.npz"):
        if stale_path != path:
            try:
                stale_path.unlink()
            except OSError:
                pass
    print(f"[BUILDINGS] Saved clipped building cache: {path}")


def _citygml_buildings_to_mesh(
    paths: List[Path],
    *,
    source_bounds: Tuple[float, float, float, float],
    xy_scale: float,
    z_scale: float,
    max_files: int = 0,
    printable: bool = False,
    nozzle_mm: float = 0.4,
    workers: int = 1,
    temporary_dir: Optional[Path] = None,
) -> Optional[Mesh]:
    if not paths:
        return None
    if workers > 1 and temporary_dir is not None:
        return _citygml_buildings_to_mesh_parallel(
            paths,
            source_bounds=source_bounds,
            xy_scale=xy_scale,
            z_scale=z_scale,
            max_files=max_files,
            printable=printable,
            nozzle_mm=nozzle_mm,
            workers=workers,
            temporary_dir=temporary_dir,
        )

    vertex_blocks: List[np.ndarray] = []
    face_blocks: List[np.ndarray] = []
    vertex_count = 0
    files_used = 0
    polygons_used = 0
    polygons_clipped = 0
    polygons_print_filtered = 0
    selected_paths = paths[:max_files] if max_files > 0 else paths
    total_files = len(selected_paths)
    for path in selected_paths:
        print(f"[BUILDINGS_PROGRESS] {files_used}/{total_files} {path.name}")
        files_used += 1
        try:
            polygons_seen = 0
            file_size = max(1, path.stat().st_size)
            for raw_coords, dimension, bytes_scanned in _iter_citygml_polygon_poslists(path):
                polygons_seen += 1
                if polygons_seen % 50000 == 0:
                    fraction = min(1.0, bytes_scanned / file_size)
                    print(
                        f"[BUILDINGS_FILE_PROGRESS] {files_used}/{total_files} {fraction:.6f} "
                        f"{path.name} {polygons_seen:,} surfaces scanned, {polygons_used:,} kept"
                    )
                values = np.fromstring(raw_coords, dtype=np.float64, sep=" ")
                if dimension < 2 or values.size < dimension * 3 or values.size % dimension:
                    continue
                coords = values.reshape((-1, dimension))
                if dimension == 2:
                    coords = np.column_stack((coords, np.zeros(coords.shape[0], dtype=np.float64)))
                else:
                    coords = coords[:, :3]
                if coords.shape[0] > 1 and np.array_equal(coords[0], coords[-1]):
                    coords = coords[:-1]
                if coords.shape[0] < 3:
                    continue
                if not _bounds_intersect(
                    float(coords[:, 0].min()), float(coords[:, 1].min()),
                    float(coords[:, 0].max()), float(coords[:, 1].max()), *source_bounds,
                ):
                    continue
                was_clipped = bool(
                    np.any(coords[:, 0] < source_bounds[0])
                    or np.any(coords[:, 0] > source_bounds[2])
                    or np.any(coords[:, 1] < source_bounds[1])
                    or np.any(coords[:, 1] > source_bounds[3])
                )
                coords = _clip_polygon_to_xy_bounds(coords, source_bounds)
                if coords.shape[0] < 3:
                    continue
                if was_clipped:
                    polygons_clipped += 1
                scaled = coords * np.asarray((xy_scale, xy_scale, z_scale), dtype=np.float64)
                if printable:
                    scaled = _simplify_building_polygon_for_print(scaled, nozzle_mm)
                    if scaled.shape[0] < 3:
                        polygons_print_filtered += 1
                        continue
                indices = np.arange(1, scaled.shape[0] - 1, dtype=np.int64)
                face_blocks.append(
                    np.column_stack((
                        np.full(indices.shape[0], vertex_count, dtype=np.int64),
                        vertex_count + indices,
                        vertex_count + indices + 1,
                    ))
                )
                vertex_blocks.append(scaled)
                vertex_count += scaled.shape[0]
                polygons_used += 1
        except OSError as exc:
            print(f"[BUILDINGS] Skipping unreadable CityGML {path}: {exc}")
        print(f"[BUILDINGS_PROGRESS] {files_used}/{total_files} {path.name}")

    if not face_blocks:
        print(f"[BUILDINGS] No CityGML building polygons intersected the model bounds across {files_used:,} file(s).")
        return None
    mesh = Mesh(vertices=np.vstack(vertex_blocks), faces=np.vstack(face_blocks))
    print(
        f"[BUILDINGS] Loaded {polygons_used:,} CityGML polygon surface(s) "
        f"from {files_used:,}/{len(paths):,} file(s): {mesh.faces.shape[0]:,} triangles; "
        f"clipped {polygons_clipped:,} polygon(s) to model bounds"
    )
    if printable:
        print(
            f"[BUILDINGS] Print simplification used a {float(nozzle_mm):.2f} mm nozzle; "
            f"discarded {polygons_print_filtered:,} sub-nozzle surface(s)."
        )
    return mesh


def _combined_xy_mask(geoms: List[object], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    flat_x = np.asarray(xs, dtype=np.float64).ravel()
    flat_y = np.asarray(ys, dtype=np.float64).ravel()
    mask = np.zeros(flat_x.shape[0], dtype=bool)
    if not geoms or flat_x.size == 0:
        return mask.reshape(np.shape(xs))

    for geom in geoms:
        bounds = geom.bounds
        bbox_mask = (
            (flat_x >= bounds[0]) &
            (flat_x <= bounds[2]) &
            (flat_y >= bounds[1]) &
            (flat_y <= bounds[3])
        )
        candidate_idx = np.flatnonzero(bbox_mask & ~mask)
        if candidate_idx.size == 0:
            continue
        local_mask = _geometry_xy_mask(geom, flat_x[candidate_idx], flat_y[candidate_idx]).ravel()
        if np.any(local_mask):
            mask[candidate_idx[local_mask]] = True
    return mask.reshape(np.shape(xs))


def _triangle_intersection_mask(
    geoms: List[object],
    tri_vertices: np.ndarray,
    *,
    candidate_mask: Optional[np.ndarray] = None,
    label: str = "geometry",
) -> np.ndarray:
    mask = np.zeros(tri_vertices.shape[0], dtype=bool)
    if not geoms or tri_vertices.shape[0] == 0:
        return mask

    try:
        from shapely.geometry import Polygon
        from shapely.prepared import prep
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Water removal requires 'shapely'. Install with: pip install shapely"
        ) from exc

    tri_min_x = tri_vertices[:, :, 0].min(axis=1)
    tri_max_x = tri_vertices[:, :, 0].max(axis=1)
    tri_min_y = tri_vertices[:, :, 1].min(axis=1)
    tri_max_y = tri_vertices[:, :, 1].max(axis=1)
    edge_lengths = np.maximum.reduce((
        np.linalg.norm(tri_vertices[:, 1, :2] - tri_vertices[:, 0, :2], axis=1),
        np.linalg.norm(tri_vertices[:, 2, :2] - tri_vertices[:, 1, :2], axis=1),
        np.linalg.norm(tri_vertices[:, 0, :2] - tri_vertices[:, 2, :2], axis=1),
    ))

    for geom_index, geom in enumerate(geoms, 1):
        bounds = geom.bounds
        candidate_idx = np.flatnonzero(
            (candidate_mask if candidate_mask is not None else True)
            & ~mask
            & (tri_max_x >= bounds[0])
            & (tri_min_x <= bounds[2])
            & (tri_max_y >= bounds[1])
            & (tri_min_y <= bounds[3])
        )
        if candidate_idx.size == 0:
            continue

        candidates = tri_vertices[candidate_idx, :, :2]
        vertex_hits = _geometry_xy_mask(geom, candidates[:, :, 0], candidates[:, :, 1])
        centroid_hits = _geometry_xy_mask(
            geom,
            candidates[:, :, 0].mean(axis=1),
            candidates[:, :, 1].mean(axis=1),
        )
        certain_hits = np.any(vertex_hits, axis=1) | centroid_hits
        if np.any(certain_hits):
            mask[candidate_idx[certain_hits]] = True
        uncertain_idx = candidate_idx[~certain_hits]
        # A triangle with no covered vertex/centroid can only intersect the
        # polygon when it reaches its boundary. Limit costly exact checks to a
        # boundary band wide enough to cover the largest terrain triangle.
        if uncertain_idx.size:
            boundary_band = geom.boundary.buffer(float(edge_lengths[uncertain_idx].max()) + 1e-9)
            boundary_hits = _geometry_xy_mask(
                boundary_band,
                tri_vertices[uncertain_idx, :, 0].mean(axis=1),
                tri_vertices[uncertain_idx, :, 1].mean(axis=1),
            )
            exact_idx = uncertain_idx[boundary_hits]
        else:
            exact_idx = uncertain_idx
        print(
            f"[WATER] {label} {geom_index}/{len(geoms)}: {candidate_idx.size:,} candidate faces, "
            f"{exact_idx.size:,} exact boundary checks"
        )
        if exact_idx.size == 0:
            continue
        prepared = prep(geom)
        for position, idx in enumerate(exact_idx, 1):
            triangle = Polygon(
                (
                    (float(tri_vertices[idx, 0, 0]), float(tri_vertices[idx, 0, 1])),
                    (float(tri_vertices[idx, 1, 0]), float(tri_vertices[idx, 1, 1])),
                    (float(tri_vertices[idx, 2, 0]), float(tri_vertices[idx, 2, 1])),
                )
            )
            if triangle.is_valid and not triangle.is_empty and prepared.intersects(triangle):
                mask[idx] = True
            if position % 100_000 == 0:
                print(
                    f"[WATER] {label} {geom_index}/{len(geoms)}: "
                    f"{position:,}/{exact_idx.size:,} boundary faces checked"
                )
    return mask


def _compact_mesh(mesh: Mesh) -> Mesh:
    if mesh.faces.size == 0:
        return Mesh(vertices=np.empty((0, 3), dtype=np.float64), faces=np.empty((0, 3), dtype=np.int64))
    used = np.unique(mesh.faces.ravel())
    remap = np.full(mesh.vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    return Mesh(vertices=mesh.vertices[used], faces=remap[mesh.faces])


def _remove_duplicate_faces(mesh: Mesh) -> Mesh:
    """Remove repeated coincident faces after vertex welding."""
    if mesh.faces.size == 0:
        return mesh

    faces = mesh.faces.astype(np.int64, copy=False)
    valid = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    collapsed_count = int((~valid).sum())
    faces = faces[valid]
    if faces.size == 0:
        if collapsed_count:
            print(f"[CLEAN] Removed {collapsed_count:,} collapsed face(s)")
        return Mesh(vertices=mesh.vertices, faces=faces.reshape(0, 3))

    canonical = np.sort(faces, axis=1)
    _, first_indices, inverse, counts = np.unique(
        canonical,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    keep = np.zeros(faces.shape[0], dtype=bool)
    keep[first_indices[counts == 1]] = True

    duplicate_groups = np.flatnonzero(counts > 1)
    normals = compute_normals(mesh.vertices, faces)
    removed_opposing = 0
    kept_duplicates = 0
    for group_id in duplicate_groups:
        idx = np.flatnonzero(inverse == group_id)
        group_normals = normals[idx]
        opposing = False
        if idx.size > 1:
            dots = group_normals @ group_normals.T
            opposing = bool(np.any(dots < -0.95))
        if opposing:
            removed_opposing += int(idx.size)
            continue
        keep[int(idx[0])] = True
        kept_duplicates += int(idx.size - 1)

    removed_total = collapsed_count + int((~keep).sum())
    if removed_total:
        print(
            f"[CLEAN] Removed {removed_total:,} duplicate/collapsed face(s) "
            f"({removed_opposing:,} opposing coincident face(s), {kept_duplicates:,} repeated face(s))"
        )
    return Mesh(vertices=mesh.vertices, faces=faces[keep].astype(np.int64, copy=False))


def _remove_vertical_wall_faces(mesh: Mesh, *, normal_z_tol: float = 1e-6, z_span_tol: float = 1e-9) -> Mesh:
    """
    Strip artificial tile wall faces before global solidification.

    SwissTopo terrain tiles are height fields, so vertical faces in the merged
    terrain mesh come from prior per-tile bases or unwelded tile boundaries.
    The global base/walls are rebuilt later from the cleaned outer boundary.
    """
    if mesh.faces.size == 0:
        return mesh

    tri_vertices = mesh.vertices[mesh.faces]
    normals = compute_normals(mesh.vertices, mesh.faces)
    z_span = tri_vertices[:, :, 2].max(axis=1) - tri_vertices[:, :, 2].min(axis=1)
    vertical = (np.abs(normals[:, 2]) <= float(normal_z_tol)) & (z_span > float(z_span_tol))
    removed = int(vertical.sum())
    if removed == 0:
        return mesh
    print(f"[CLEAN] Removed {removed:,} vertical tile wall face(s)")
    return Mesh(vertices=mesh.vertices, faces=mesh.faces[~vertical].astype(np.int64, copy=False))


def _remove_tile_base_faces(mesh: Mesh, *, normal_z_tol: float = 1e-6, z_span_tol: float = 1e-9) -> Mesh:
    """Remove flat bottom caps that came from previously solidified tile STLs."""
    if mesh.faces.size == 0:
        return mesh

    tri_vertices = mesh.vertices[mesh.faces]
    normals = compute_normals(mesh.vertices, mesh.faces)
    z_span = tri_vertices[:, :, 2].max(axis=1) - tri_vertices[:, :, 2].min(axis=1)
    vertical = (np.abs(normals[:, 2]) <= float(normal_z_tol)) & (z_span > float(z_span_tol))
    if not np.any(vertical):
        return mesh

    bottom_vertex_mask = np.zeros(mesh.vertices.shape[0], dtype=bool)
    vertical_faces = mesh.faces[vertical]
    vertical_vertices = tri_vertices[vertical]
    vertical_min_z = vertical_vertices[:, :, 2].min(axis=1)
    is_bottom = np.abs(vertical_vertices[:, :, 2] - vertical_min_z[:, None]) <= float(z_span_tol)
    bottom_vertex_mask[vertical_faces[is_bottom]] = True

    flat = z_span <= float(z_span_tol)
    bottom_cap = flat & np.all(bottom_vertex_mask[mesh.faces], axis=1)
    removed = int(bottom_cap.sum())
    if removed == 0:
        return mesh
    print(f"[CLEAN] Removed {removed:,} old tile bottom-cap face(s)")
    return Mesh(vertices=mesh.vertices, faces=mesh.faces[~bottom_cap].astype(np.int64, copy=False))


def _clean_merged_terrain_mesh(mesh: Mesh) -> Mesh:
    """Clean tile-boundary artifacts before water handling and global base creation."""
    before_vertices = int(mesh.vertices.shape[0])
    before_faces = int(mesh.faces.shape[0])
    mesh = _remove_duplicate_faces(mesh)
    mesh = _remove_tile_base_faces(mesh)
    mesh = _remove_vertical_wall_faces(mesh)
    mesh = _remove_duplicate_faces(mesh)
    mesh = _compact_mesh(mesh)
    after_vertices = int(mesh.vertices.shape[0])
    after_faces = int(mesh.faces.shape[0])
    if before_vertices != after_vertices or before_faces != after_faces:
        print(
            f"[CLEAN] Merged terrain mesh: "
            f"{before_vertices:,} vertices / {before_faces:,} triangles -> "
            f"{after_vertices:,} vertices / {after_faces:,} triangles"
        )
    return mesh


def _apply_water_adjustment_to_mesh(
    mesh: Mesh,
    *,
    mode: str,
    water_geoms: List[object],
    bridge_geoms: List[object],
    amount_mm: float,
) -> Mesh:
    mode = mode.strip().lower()
    if mode == "off" or not water_geoms:
        return mesh
    if mode not in {"lower", "remove"}:
        raise ValueError("Water mode must be one of: off, lower, remove.")

    vertices = mesh.vertices
    if mode == "lower":
        water_mask = _combined_xy_mask(water_geoms, vertices[:, 0], vertices[:, 1]).ravel()
        if bridge_geoms:
            water_mask &= ~_combined_xy_mask(bridge_geoms, vertices[:, 0], vertices[:, 1]).ravel()
        lowered_count = int(water_mask.sum())
        if lowered_count == 0:
            print("[WATER] No lake or river vertices intersect the merged model bounds.")
            return mesh
        out_vertices = vertices.copy()
        out_vertices[water_mask, 2] -= float(amount_mm)
        print(f"[WATER] Lowered {lowered_count:,} vertex/vertices by {float(amount_mm):.3f} mm")
        return Mesh(vertices=out_vertices, faces=mesh.faces)

    tri_vertices = vertices[mesh.faces]
    remove_mask = _triangle_intersection_mask(water_geoms, tri_vertices, label="Water")
    if bridge_geoms:
        bridge_mask = _triangle_intersection_mask(
            bridge_geoms,
            tri_vertices,
            candidate_mask=remove_mask,
            label="Bridge protection",
        )
        remove_mask &= ~bridge_mask
    remove_count = int(remove_mask.sum())
    if remove_count == 0:
        print("[WATER] No lake or river faces intersect the merged model bounds.")
        return mesh
    kept_faces = mesh.faces[~remove_mask]
    print(f"[WATER] Removed {remove_count:,}/{mesh.faces.shape[0]:,} face(s) for water cutouts")
    return _compact_mesh(Mesh(vertices=mesh.vertices, faces=kept_faces))


def _geometry_xy_mask(geom, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    flat_x = np.asarray(xs, dtype=np.float64).ravel()
    flat_y = np.asarray(ys, dtype=np.float64).ravel()
    mask = np.zeros(flat_x.shape[0], dtype=bool)

    try:
        from shapely import intersects_xy as shapely_intersects_xy  # type: ignore
    except Exception:
        shapely_intersects_xy = None

    if shapely_intersects_xy is not None:
        chunk_size = 500_000
        for start in range(0, flat_x.shape[0], chunk_size):
            end = min(start + chunk_size, flat_x.shape[0])
            mask[start:end] = np.asarray(
                shapely_intersects_xy(geom, flat_x[start:end], flat_y[start:end]),
                dtype=bool,
            )
        return mask.reshape(np.shape(xs))

    try:
        from shapely.geometry import Point
        from shapely.prepared import prep
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Lake lowering requires 'shapely'. Install with: pip install shapely"
        ) from exc

    prepared = prep(geom)
    for idx, (x, y) in enumerate(zip(flat_x, flat_y)):
        mask[idx] = prepared.covers(Point(float(x), float(y)))
    return mask.reshape(np.shape(xs))


def _apply_lake_lowering_to_mesh(mesh: Mesh, lake_shp: Path, *, lake_scale: float, amount_mm: float) -> Mesh:
    if amount_mm <= 0.0:
        return mesh
    if lake_scale <= 0.0:
        raise ValueError("Lake scale must be > 0.")

    vertices = mesh.vertices
    model_min_x = float(vertices[:, 0].min())
    model_max_x = float(vertices[:, 0].max())
    model_min_y = float(vertices[:, 1].min())
    model_max_y = float(vertices[:, 1].max())

    lake_geoms = _load_all_lake_geometries(lake_shp)
    if not lake_geoms:
        print("[LAKES] No lake polygons found in the lake dataset.")
        return mesh

    scales_to_try: List[float] = [float(lake_scale)]
    if abs(float(lake_scale) - 1.0) > 1e-12:
        scales_to_try.append(1.0)

    for scale_try in scales_to_try:
        tested_mask = np.zeros(vertices.shape[0], dtype=bool)
        lowered_mask = np.zeros(vertices.shape[0], dtype=bool)
        intersecting_feature_count = 0

        for geom in lake_geoms:
            geom_scaled = _scale_border_geometry(geom, scale_try) if scale_try != 1.0 else geom
            bounds = geom_scaled.bounds
            if not _bounds_intersect(
                float(bounds[0]),
                float(bounds[1]),
                float(bounds[2]),
                float(bounds[3]),
                model_min_x,
                model_min_y,
                model_max_x,
                model_max_y,
            ):
                continue

            intersecting_feature_count += 1
            bbox_mask = (
                (vertices[:, 0] >= bounds[0]) &
                (vertices[:, 0] <= bounds[2]) &
                (vertices[:, 1] >= bounds[1]) &
                (vertices[:, 1] <= bounds[3])
            )
            candidate_idx = np.flatnonzero(bbox_mask)
            if candidate_idx.size == 0:
                continue
            tested_mask[candidate_idx] = True
            local_mask = _geometry_xy_mask(geom_scaled, vertices[candidate_idx, 0], vertices[candidate_idx, 1])
            if np.any(local_mask):
                lowered_mask[candidate_idx[local_mask]] = True

        tested_count = int(tested_mask.sum())
        lowered_count = int(lowered_mask.sum())
        if lowered_count > 0:
            out_vertices = vertices.copy()
            out_vertices[lowered_mask, 2] -= float(amount_mm)
            if scale_try != float(lake_scale):
                print(
                    f"[LAKES] Stored scale {float(lake_scale):.6f} found no lake hits; "
                    f"used fallback XY scale {scale_try:.6f}"
                )
            print(
                f"[LAKES] Lowered {lowered_count:,} merged vertex/vertices by {float(amount_mm):.3f} mm "
                f"(tested {tested_count:,}/{vertices.shape[0]:,}, features {intersecting_feature_count:,})"
            )
            return Mesh(vertices=out_vertices, faces=mesh.faces)

    print("[LAKES] No lake polygons intersect the merged model bounds.")
    return mesh


def _interp_z_from_triangle(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
    x: float,
    y: float,
) -> float:
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-12:
        return (az + bz + cz) / 3.0
    w1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denom
    w2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denom
    w3 = 1.0 - w1 - w2
    return w1 * az + w2 * bz + w3 * cz


def _triangulate_intersection(geom):
    try:
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        from shapely.ops import triangulate
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Border clipping requires 'shapely'. Install with: pip install shapely"
        ) from exc

    if geom.is_empty:
        return []

    polys = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]

    triangles_2d = []
    for poly in polys:
        if poly.is_empty or poly.area == 0.0:
            continue
        for tri in triangulate(poly):
            if not poly.covers(tri.representative_point()):
                continue
            coords = list(tri.exterior.coords)
            if len(coords) < 4:
                continue
            triangles_2d.append(coords[:3])
    return triangles_2d


def _clip_triangle_to_border(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
    border_geom,
    border_prep,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    try:
        from shapely.geometry import Polygon
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Border clipping requires 'shapely'. Install with: pip install shapely"
        ) from exc

    tri_poly = Polygon([(a[0], a[1]), (b[0], b[1]), (c[0], c[1])])
    if border_prep.covers(tri_poly):
        return [(a, b, c)]
    if not border_prep.intersects(tri_poly):
        return []
    inter = tri_poly.intersection(border_geom)
    clipped_tris: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []
    for tri2d in _triangulate_intersection(inter):
        pts3d = []
        for x, y in tri2d:
            z = _interp_z_from_triangle(a, b, c, float(x), float(y))
            pts3d.append((float(x), float(y), float(z)))
        if len(pts3d) == 3:
            clipped_tris.append((pts3d[0], pts3d[1], pts3d[2]))
    return clipped_tris


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
    water_mode: str = "off",
    water_lower_mm: float = 0.0,
    lake_shp: Optional[Path] = None,
    river_shp: Optional[Path] = None,
    riverbank_geojson: Optional[Path] = None,
    bridge_shps: Optional[List[Path]] = None,
    water_scale: float = 1.0,
    bridge_buffer_mm: float = 1.2,
    water_feature_ids: Optional[set[str]] = None,
    building_paths: Optional[List[Path]] = None,
    buildings_xy_scale: float = 1.0,
    buildings_z_scale: float = 1.0,
    buildings_max_files: int = 0,
    printable_simplification: bool = False,
    printer_nozzle_mm: float = 0.4,
    buildings_workers: int = 1,
    clean_tiles_after_merge: bool = False,
    clip_border: bool = False,
    border_geom=None,
) -> None:
    """
    Full merge that streams tiles, welds vertices on the fly, and can solidify globally.
    This avoids holding all tile meshes in memory at once.
    """
    stl_files = _list_stl_files()
    print(f"Found {len(stl_files)} STL file(s) under ./output/tiles")
    print(f"[MERGE-MESH] Output: {out_stl}")
    _validate_no_enclosed_tile_holes(stl_files)

    use_weld = float(weld_tol) > 0.0
    if use_weld:
        print(f"[MERGE-MESH] Welding on the fly with tol={float(weld_tol)}")

    border_prep = None
    if clip_border:
        if border_geom is None:
            raise ValueError("clip_border is True but no border geometry was provided.")
        try:
            from shapely.prepared import prep
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Border clipping requires 'shapely'. Install with: pip install shapely"
            ) from exc
        border_prep = prep(border_geom)
        print("[MERGE-MESH] Clipping triangles to border geometry")

    mapping: Dict[Tuple[int, int, int], int] = {} if use_weld else {}
    new_verts: List[Tuple[float, float, float]] = []
    new_faces: List[Tuple[int, int, int]] = []
    raw_vertices = 0
    raw_tris = 0

    for i, p in enumerate(stl_files, 1):
        print("\n" + "=" * 80)
        print(f"[MERGE-MESH] ({i}/{len(stl_files)}) Reading {p}")
        print("=" * 80)
        for a, b, c in _iter_stl_triangles(p):
            raw_tris += 1
            raw_vertices += 3

            tri_list = [(a, b, c)]
            if clip_border:
                assert border_prep is not None
                tri_list = _clip_triangle_to_border(a, b, c, border_geom, border_prep)
                if not tri_list:
                    continue

            for ta, tb, tc in tri_list:
                if use_weld:
                    ia = mapping.get(
                        (int(round(ta[0] / weld_tol)), int(round(ta[1] / weld_tol)), int(round(ta[2] / weld_tol)))
                    )
                    if ia is None:
                        ia = len(new_verts)
                        mapping[
                            (int(round(ta[0] / weld_tol)), int(round(ta[1] / weld_tol)), int(round(ta[2] / weld_tol)))
                        ] = ia
                        new_verts.append(ta)

                    ib = mapping.get(
                        (int(round(tb[0] / weld_tol)), int(round(tb[1] / weld_tol)), int(round(tb[2] / weld_tol)))
                    )
                    if ib is None:
                        ib = len(new_verts)
                        mapping[
                            (int(round(tb[0] / weld_tol)), int(round(tb[1] / weld_tol)), int(round(tb[2] / weld_tol)))
                        ] = ib
                        new_verts.append(tb)

                    ic = mapping.get(
                        (int(round(tc[0] / weld_tol)), int(round(tc[1] / weld_tol)), int(round(tc[2] / weld_tol)))
                    )
                    if ic is None:
                        ic = len(new_verts)
                        mapping[
                            (int(round(tc[0] / weld_tol)), int(round(tc[1] / weld_tol)), int(round(tc[2] / weld_tol)))
                        ] = ic
                        new_verts.append(tc)
                else:
                    ia = len(new_verts)
                    new_verts.append(ta)
                    ib = len(new_verts)
                    new_verts.append(tb)
                    ic = len(new_verts)
                    new_verts.append(tc)

                new_faces.append((ia, ib, ic))
        print(f"[PROGRESS] {i}/{len(stl_files)} {p.name}")

    merged = Mesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
    )
    print(
        f"[MERGE-MESH] Combined raw vertices: {raw_vertices:,} -> {merged.vertices.shape[0]:,} | triangles: {merged.faces.shape[0]:,}"
    )

    if z_scale != 1.0:
        merged = scale_mesh_z(merged, float(z_scale))

    merged = _clean_merged_terrain_mesh(merged)

    if water_mode != "off":
        model_bounds = (
            float(merged.vertices[:, 0].min()),
            float(merged.vertices[:, 1].min()),
            float(merged.vertices[:, 0].max()),
            float(merged.vertices[:, 1].max()),
        )
        water_geoms, bridge_geoms = _prepare_water_geometries(
            lake_shp=lake_shp,
            river_shp=river_shp,
            riverbank_geojson=riverbank_geojson,
            bridge_shps=list(bridge_shps or []),
            water_scale=float(water_scale),
            bridge_buffer_mm=float(bridge_buffer_mm),
            model_bounds=model_bounds,
            water_feature_ids=water_feature_ids,
        )
        if not water_geoms:
            print("[WATER] No lake or river geometry found.")
        print(f"[WATER] Applying merged water mode: {water_mode}")
        merged = _apply_water_adjustment_to_mesh(
            merged,
            mode=str(water_mode),
            water_geoms=water_geoms,
            bridge_geoms=bridge_geoms,
            amount_mm=float(water_lower_mm),
        )

    model_bounds_for_overlays = (
        float(merged.vertices[:, 0].min()),
        float(merged.vertices[:, 1].min()),
        float(merged.vertices[:, 0].max()),
        float(merged.vertices[:, 1].max()),
    )

    if make_solid_flag:
        if base_z_value is not None:
            base_z = float(base_z_value)
        elif base_mode == "sealevel":
            base_z = 0.0
        else:
            base_z = float(merged.vertices[:, 2].min() - float(base_thickness_value))
        merged = make_solid(merged, base_z)

    if building_paths:
        source_bounds = _source_bounds_for_model_bounds(model_bounds_for_overlays, float(buildings_xy_scale))
        cache_path = _building_cache_path(
            building_paths,
            source_bounds,
            float(buildings_xy_scale),
            float(buildings_z_scale),
            int(buildings_max_files),
            bool(printable_simplification),
            float(printer_nozzle_mm),
        )
        building_mesh = _load_building_mesh_cache(cache_path)
        if building_mesh is None:
            print("[BUILDINGS] Building clipped mesh cache; this is only needed once for this model area.")
            chunk_dir = cache_path.parent / f"{cache_path.stem}.chunks"
            try:
                building_mesh = _citygml_buildings_to_mesh(
                    building_paths,
                    source_bounds=source_bounds,
                    xy_scale=float(buildings_xy_scale),
                    z_scale=float(buildings_z_scale),
                    max_files=int(buildings_max_files),
                    printable=bool(printable_simplification),
                    nozzle_mm=float(printer_nozzle_mm),
                    workers=max(1, int(buildings_workers)),
                    temporary_dir=chunk_dir,
                )
            finally:
                shutil.rmtree(chunk_dir, ignore_errors=True)
            if building_mesh is not None:
                _save_building_mesh_cache(cache_path, building_mesh)
        if building_mesh is not None:
            merged = concat_meshes([merged, building_mesh])
            if float(weld_tol) > 0.0:
                merged = weld_vertices(merged, float(weld_tol))
            print(
                f"[BUILDINGS] Appended buildings. Final mesh: "
                f"{merged.vertices.shape[0]:,} vertices, {merged.faces.shape[0]:,} triangles"
            )

    if printable_simplification:
        merged = _simplify_mesh_for_print(merged, float(printer_nozzle_mm))

    if binary_out:
        write_binary_stl(merged, out_stl, solid_name=solid_name)
    else:
        write_ascii_stl(merged, out_stl, solid_name=solid_name)

    print(f"[MERGE-MESH] Done. Wrote {out_stl}")
    if clean_tiles_after_merge:
        _clear_tiles_dir()


# ----------------------------
# Main CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert XYZ/GeoTIFF terrain data to STL surface(s).")
    input_tile_edge_units = 1000.0  # 1 km tiles, input units are meters by default.
    ap.add_argument(
        "--all",
        action="store_true",
        help="Convert all .xyz and .tif/.tiff files in ./work/terrain into ./output/tiles.",
    )
    ap.add_argument(
        "--merge-stl",
        type=Path,
        default=None,
        help="Merge existing .stl files found under ./output/tiles into one combined STL (no XYZ processing).",
    )
    ap.add_argument(
        "--clip-border",
        action="store_true",
        help="Clip merged STL to the Swiss border (requires --merge-stl).",
    )
    ap.add_argument(
        "--border-shp",
        type=Path,
        default=None,
        help="Path to a Swiss border .shp file (default: auto-detect from ./reference_data).",
    )
    ap.add_argument(
        "--border-scale",
        type=str,
        default="auto",
        help="Scale factor to apply to border coordinates to match STL units. "
             "Use a number (e.g. 10) or 'auto' to reuse the last conversion scale.",
    )
    ap.add_argument(
        "--border-keep",
        type=str,
        default="",
        help="Comma-separated list of border feature names to keep (case-insensitive).",
    )
    ap.add_argument(
        "--border-field",
        type=str,
        default="",
        help="DBF field name to match when using --border-keep (optional).",
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
        "--lake-lower-mm",
        type=float,
        default=0.0,
        help="Backward-compatible alias for --water-mode lower --water-lower-mm.",
    )
    ap.add_argument(
        "--lake-shp",
        type=Path,
        default=None,
        help="Path to a lake polygon .shp file used during merge (default: auto-detect standing water in ./reference_data).",
    )
    ap.add_argument(
        "--water-mode",
        type=str,
        default="off",
        choices=["off", "lower", "remove"],
        help="Water treatment during merge: off, lower lake/river surfaces, or remove them as cutouts.",
    )
    ap.add_argument(
        "--water-features",
        type=str,
        default="lakes,rivers",
        help="Comma-separated water feature classes to affect: lakes, rivers, or lakes,rivers (default).",
    )
    ap.add_argument(
        "--water-feature-ids",
        type=str,
        default="",
        help="Optional comma-separated selected water feature IDs from the GUI, e.g. lake:0,river:3. Default: all.",
    )
    ap.add_argument(
        "--water-lower-mm",
        type=float,
        default=0.0,
        help="Lower lake and river surfaces by this amount in final STL units (mm). Requires --water-mode lower.",
    )
    ap.add_argument(
        "--river-shp",
        type=Path,
        default=None,
        help="Path to a river centerline .shp file (default: auto-detect TLM_FLIESSGEWAESSER in ./reference_data).",
    )
    ap.add_argument(
        "--riverbank-geojson",
        type=Path,
        default=DEFAULT_RIVERBANK_GEOJSON,
        help="LV95 GeoJSON with actual OSM riverbank polygons (default: ./work/water/riverbanks.geojson).",
    )
    ap.add_argument(
        "--bridge-shp",
        type=Path,
        action="append",
        default=[],
        help="Transport .shp file with KUNSTBAUTE bridge attributes to protect from water lowering/removal. "
             "May be passed more than once; defaults to auto-detected road/rail layers.",
    )
    ap.add_argument(
        "--bridge-buffer-mm",
        type=float,
        default=1.2,
        help="Final-model width used to protect bridge features from water lowering/removal (default: 1.2 mm).",
    )
    ap.add_argument(
        "--buildings",
        action="store_true",
        help="Append swissBUILDINGS3D 3.0 Beta CityGML building surfaces during --merge-stl.",
    )
    ap.add_argument(
        "--buildings-path",
        type=Path,
        action="append",
        default=[],
        help="CityGML file or directory for swissBUILDINGS3D 3.0 Beta. May be passed more than once; default: auto-detect .gml/.xml under reference_data.",
    )
    ap.add_argument(
        "--buildings-max-files",
        type=int,
        default=0,
        help="Optional limit on CityGML files to read, useful for testing. Default 0 means no limit.",
    )
    ap.add_argument(
        "--buildings-workers",
        type=int,
        default=4,
        help="Worker processes for scanning large CityGML files (default: 4).",
    )
    ap.add_argument(
        "--printable-simplification",
        action="store_true",
        help="Simplify the final terrain, water, base, and building mesh for the printer nozzle resolution.",
    )
    ap.add_argument(
        "--printer-nozzle-mm",
        type=float,
        default=0.4,
        help="Printer nozzle diameter used by --printable-simplification (default: 0.4 mm).",
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
        "--tile-size-mm",
        type=float,
        default=None,
        help="Target side length (mm) for a single 1 km input tile. "
             "Sets a fixed scale (assumes 1 km = 1000 input units).",
    )
    ap.add_argument(
        "--scale-ratio",
        type=str,
        default=None,
        help="Map scale ratio like '100' or '1:100' (meaning 1 model unit = 100 real units). "
             "Sets a fixed scale (assumes input units are meters).",
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
        "--input-resolution",
        action="store_true",
        help="Keep every source grid sample in the STL (equivalent to step=1 after crop/scale). "
             "This can create very large STL files.",
    )
    ap.add_argument(
        "--crop-rect",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help="Crop conversion to this source-coordinate rectangle before scaling/triangulation. "
             "Use the SwissTopo order WEST SOUTH EAST NORTH, "
             "e.g. --crop-rect 2600000 1200000 2600500 1200400.",
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
    ap.add_argument(
        "--clean-tiles",
        action="store_true",
        help="Delete existing files in ./output/tiles before converting.",
    )
    ap.add_argument(
        "--clean-tiles-after-merge",
        action="store_true",
        help="Delete intermediate files in ./output/tiles after a successful --merge-stl run.",
    )

    args = ap.parse_args()

    if args.target_size_mm is not None and not args.all:
        ap.error("--target-size-mm can only be used with --all.")
    if args.tile_size_mm is not None and not args.all:
        ap.error("--tile-size-mm can only be used with --all.")
    if args.scale_ratio is not None and not args.all:
        ap.error("--scale-ratio can only be used with --all.")
    if args.input_resolution and not args.all:
        ap.error("--input-resolution can only be used with --all.")

    scale_mode_count = sum(
        1 for v in (args.target_size_mm, args.tile_size_mm, args.scale_ratio) if v is not None
    )
    if scale_mode_count > 1:
        ap.error("Use only one of --target-size-mm, --tile-size-mm, or --scale-ratio.")

    if float(args.lake_lower_mm) > 0.0:
        if args.water_mode != "off" and args.water_mode != "lower":
            ap.error("--lake-lower-mm cannot be combined with --water-mode remove.")
        args.water_mode = "lower"
        if float(args.water_lower_mm) <= 0.0:
            args.water_lower_mm = float(args.lake_lower_mm)

    try:
        include_lakes, include_rivers = _parse_water_features(str(args.water_features))
        water_feature_ids = _parse_water_feature_ids(str(args.water_feature_ids))
    except ValueError as exc:
        ap.error(str(exc))

    if args.water_mode != "off" and args.merge_stl is None:
        ap.error("--water-mode can only be used with --merge-stl.")
    if args.water_mode != "off" and not (include_lakes or include_rivers):
        ap.error("--water-mode requires at least one --water-features value: lakes or rivers.")
    if args.water_mode == "lower" and float(args.water_lower_mm) <= 0.0:
        ap.error("--water-mode lower requires --water-lower-mm > 0.")
    if args.water_mode != "off" and float(args.bridge_buffer_mm) < 0.0:
        ap.error("--bridge-buffer-mm must be >= 0.")
    if args.buildings and args.merge_stl is None:
        ap.error("--buildings can only be used with --merge-stl.")
    if args.buildings_max_files < 0:
        ap.error("--buildings-max-files must be >= 0.")
    if args.buildings_workers < 1:
        ap.error("--buildings-workers must be >= 1.")
    if args.printable_simplification and float(args.printer_nozzle_mm) <= 0.0:
        ap.error("--printer-nozzle-mm must be > 0 when --printable-simplification is used.")

    try:
        crop_rect = _parse_crop_rect(args.crop_rect)
    except ValueError as exc:
        ap.error(str(exc))

    if crop_rect is not None and args.merge_stl is not None:
        ap.error("--crop-rect can only be used with --all conversion.")

    if args.merge_stl is not None:
        border_geom = None
        lake_shp = None
        river_shp = None
        riverbank_geojson = None
        bridge_shps: List[Path] = []
        water_scale = 1.0
        building_paths: List[Path] = []
        buildings_xy_scale = 1.0
        buildings_z_scale = 1.0
        if args.clip_border:
            border_path = args.border_shp or _default_border_shp()
            if border_path is None:
                ap.error("--clip-border requested but no border shapefile was found in ./reference_data.")
            border_scale = _parse_border_scale(str(args.border_scale), Path("./output/tiles"))
            print(f"[BORDER] Loading border from {border_path}")
            keep_values = _parse_border_keep_list(str(args.border_keep))
            keep_field = str(args.border_field).strip() or None
            if keep_values:
                print(f"[BORDER] Keeping {len(keep_values)} feature(s)")
            border_geom = _load_border_geometry(
                border_path,
                keep_values=keep_values or None,
                keep_field=keep_field,
            )
            if border_scale != 1.0:
                print(f"[BORDER] Scaling border by {border_scale:.6f}")
                border_geom = _scale_border_geometry(border_geom, float(border_scale))

        if args.water_mode != "off":
            if include_lakes:
                lake_shp = args.lake_shp or _default_lake_shp()
            if include_lakes and lake_shp is None:
                ap.error("--water-mode requested but no standing-water shapefile was found in ./reference_data.")
            if include_rivers:
                river_shp = args.river_shp or _default_river_shp()
                riverbank_geojson = args.riverbank_geojson
            if include_rivers and river_shp is None:
                ap.error("--water-mode requested but no river shapefile was found in ./reference_data.")
            bridge_shps = list(args.bridge_shp or []) or _default_bridge_shps()
            water_scale = _parse_border_scale("auto", Path("./output/tiles"))
            if lake_shp is not None:
                print(f"[WATER] Using lake polygons from {lake_shp}")
            else:
                print("[WATER] Lake features disabled")
            if river_shp is not None:
                print(f"[WATER] Using river centerlines from {river_shp}")
                print(f"[WATER] Using actual river outlines from {riverbank_geojson}")
            else:
                print("[WATER] River features disabled")
            if bridge_shps:
                print(f"[WATER] Protecting bridges from {len(bridge_shps)} shapefile(s)")
            else:
                print("[WATER] No bridge shapefiles found; river/lake treatment will not preserve bridges.")
            print(f"[WATER] Using XY scale {water_scale:.6f} for merged water matching")

        if args.buildings:
            building_paths = _building_input_paths(list(args.buildings_path or []))
            if not building_paths:
                ap.error("--buildings requested but no .gml or .xml building files were found.")
            buildings_xy_scale = _parse_border_scale("auto", Path("./output/tiles"))
            tile_z_scale = _read_last_z_scale(Path("./output/tiles") / "scale_info.json")
            buildings_z_scale = buildings_xy_scale * tile_z_scale * float(args.merge_z_scale)
            print(f"[BUILDINGS] Using {len(building_paths):,} CityGML building file(s)")
            print(
                f"[BUILDINGS] XY scale {buildings_xy_scale:.6f}; "
                f"Z scale {buildings_z_scale:.6f}"
            )

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
            water_mode=str(args.water_mode),
            water_lower_mm=float(args.water_lower_mm),
            lake_shp=lake_shp,
            river_shp=river_shp,
            riverbank_geojson=riverbank_geojson,
            bridge_shps=bridge_shps,
            water_scale=float(water_scale),
            bridge_buffer_mm=float(args.bridge_buffer_mm),
            water_feature_ids=water_feature_ids,
            building_paths=building_paths,
            buildings_xy_scale=float(buildings_xy_scale),
            buildings_z_scale=float(buildings_z_scale),
            buildings_max_files=int(args.buildings_max_files),
            printable_simplification=bool(args.printable_simplification),
            printer_nozzle_mm=float(args.printer_nozzle_mm),
            buildings_workers=int(args.buildings_workers),
            clean_tiles_after_merge=bool(args.clean_tiles_after_merge),
            clip_border=bool(args.clip_border),
            border_geom=border_geom,
        )
        return

    if args.all:
        if bool(args.make_solid):
            print("[WARN] --make-solid is ignored in --all mode. "
                  "Use --merge-stl --make-solid to add a global base after merging.")

        input_files = _list_input_files()
        xyz_count = len([p for p in input_files if p.suffix.lower() == ".xyz"])
        tif_count = len([p for p in input_files if p.suffix.lower() in {".tif", ".tiff"}])
        print(
            f"Found {len(input_files)} input file(s) "
            f"({xyz_count} XYZ, {tif_count} TIF) under ./work/terrain"
        )
        if crop_rect is not None:
            print(
                f"[CROP] Source coordinate rectangle: "
                f"X {crop_rect.min_x:.3f}..{crop_rect.max_x:.3f}, "
                f"Y {crop_rect.min_y:.3f}..{crop_rect.max_y:.3f}"
            )

        target_size_mm = args.target_size_mm
        tile_size_mm = args.tile_size_mm
        scale_ratio = args.scale_ratio
        target_edge = str(args.target_edge)
        if target_size_mm is None and tile_size_mm is None and scale_ratio is None and args.step == 1 and sys.stdin.isatty():
            target_edge = _prompt_edge_mode()
            edge_label = "shortest" if target_edge == "shortest" else "longest"
            target_size_mm = _prompt_optional_float(
                f"Target {edge_label} edge length in mm (blank to keep --step 1): "
            )

        auto_scale = 1.0
        auto_step = int(args.step)
        if target_size_mm is not None:
            auto_scale, auto_step = _auto_scale_and_step(
                input_files,
                target_size_mm=float(target_size_mm),
                target_resolution_mm=float(args.target_resolution_mm),
                edge_mode=target_edge,
                crop_rect=crop_rect,
            )
        elif tile_size_mm is not None:
            auto_scale = float(tile_size_mm) / float(input_tile_edge_units)
            print(
                f"[SCALE] Tile size: {float(tile_size_mm):.2f} mm per 1 km tile "
                f"-> scale {auto_scale:.6f} (input units -> mm)"
            )
            auto_step = _auto_step_from_scale(
                input_files,
                scale=float(auto_scale),
                target_resolution_mm=float(args.target_resolution_mm),
                crop_rect=crop_rect,
            )
        elif scale_ratio is not None:
            ratio = _parse_scale_ratio(str(scale_ratio))
            auto_scale = 1000.0 / float(ratio)
            print(
                f"[SCALE] Map scale 1:{ratio:.6f} "
                f"-> scale {auto_scale:.6f} (input units -> mm)"
            )
            auto_step = _auto_step_from_scale(
                input_files,
                scale=float(auto_scale),
                target_resolution_mm=float(args.target_resolution_mm),
                crop_rect=crop_rect,
            )

        if bool(args.input_resolution):
            auto_step = 1
            print("[DETAIL] Using full input resolution: step=1 (no downsampling).")

        output_tiles_dir = Path("./output/tiles")
        if args.clean_tiles and output_tiles_dir.exists():
            for existing in output_tiles_dir.iterdir():
                try:
                    if existing.is_file():
                        existing.unlink()
                    else:
                        shutil.rmtree(existing)
                except OSError:
                    print(f"[WARN] Failed to remove: {existing.name}")
        output_tiles_dir.mkdir(parents=True, exist_ok=True)
        scale_info_path = output_tiles_dir / "scale_info.json"
        try:
            scale_info_path.write_text(
                json.dumps(
                    {
                        "scale_xy": float(auto_scale),
                        "z_scale": float(args.z_scale),
                        "crop_rect": (
                            {
                                "min_x": crop_rect.min_x,
                                "min_y": crop_rect.min_y,
                                "max_x": crop_rect.max_x,
                                "max_y": crop_rect.max_y,
                            }
                            if crop_rect is not None
                            else None
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            print(f"[WARN] Failed to write {scale_info_path}")

        tasks: List[Tuple[Path, Path, str, float, float, float, int, float, Optional[float], str, bool, Optional[CropRect]]] = []
        model_name = args.model_name.strip()
        for input_path in input_files:
            if model_name:
                tile_stem = f"{model_name}_{input_path.stem}"
            else:
                tile_stem = input_path.stem
            stl_path = output_tiles_dir / f"{tile_stem}.stl"
            per_file_name = tile_stem
            tasks.append(
                (
                    input_path,
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
                    crop_rect,
                )
            )

        workers = max(1, int(args.workers))
        if workers == 1 or len(tasks) == 1:
            failures = 0
            for task in tasks:
                input_path = task[0]
                try:
                    _convert_worker(task)
                except Exception as e:
                    failures += 1
                    print(f"ERROR converting {input_path}: {e}")
            if failures:
                raise SystemExit(f"{failures} tile(s) failed during conversion; aborting before merge.")
            return

        total = len(tasks)
        completed = 0
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_convert_worker, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                input_path = task[0]
                completed += 1
                try:
                    future.result()
                except Exception as e:
                    failures += 1
                    print(f"ERROR converting {input_path}: {e}")
                print(f"[PROGRESS] {completed}/{total} {input_path.name}")

        if failures:
            raise SystemExit(f"{failures} tile(s) failed during conversion; aborting before merge.")
        return

    ap.error("Use --all to convert tiles or --merge-stl <out.stl> to merge.")


if __name__ == "__main__":
    main()

    

