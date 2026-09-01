#!/usr/bin/env python3
from __future__ import annotations

import shutil
import os
import stat
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import shapefile
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "reference_data"
BRIDGE_BASE = "swissTLM3D_TLM_BRIDGE_PROTECTION"
LAKE_SOURCE_BASE = "swissTLM3D_TLM_STEHENDES_GEWAESSER"
LAKE_POLYGON_BASE = "swissTLM3D_TLM_LAKE_POLYGONS"

KEEP_DIRS = {"TLM_GEWAESSER", "TLM_BRIDGES"}
KEEP_BASES = {
    LAKE_SOURCE_BASE,
    LAKE_POLYGON_BASE,
    "swissTLM3D_TLM_FLIESSGEWAESSER",
    BRIDGE_BASE,
}
KEEP_SUFFIXES = {
    LAKE_SOURCE_BASE: {".shp", ".shx", ".dbf", ".prj", ".cpg"},
    LAKE_POLYGON_BASE: {".shp", ".shx", ".dbf", ".prj", ".cpg"},
    "swissTLM3D_TLM_FLIESSGEWAESSER": {".shp", ".shx", ".dbf", ".prj", ".cpg"},
    BRIDGE_BASE: {".shp", ".shx", ".dbf", ".prj", ".cpg"},
}
TRANSPORT_SOURCES = [
    ("TLM_STRASSEN", "swissTLM3D_TLM_STRASSE.shp"),
    ("TLM_OEV", "swissTLM3D_TLM_EISENBAHN.shp"),
]


def _tlm_root() -> Path:
    roots = sorted(path for path in GEOMETRY.glob("swisstlm3d_*_2056_5728.shp") if path.is_dir())
    if not roots:
        raise FileNotFoundError("No extracted SwissTLM3D shapefile package was found in reference_data.")
    return roots[-1]


def _normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip().lower()


def _is_bridge(value: object) -> bool:
    token = _normalize(value)
    return any(part in token for part in ("bruecke", "brucke", "bridge"))


def _record_dict(field_names: list[str], record) -> dict[str, object]:
    return dict(zip(field_names, record))


def _text_value(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    return "" if value is None else str(value).strip()


def _water_group_key(record: dict[str, object]) -> str:
    for field in ("GEWISS_NR", "GEW_LAUF_U", "GEW_NAME_U", "NAME", "UUID"):
        text = _text_value(record, field)
        if text:
            return f"{field}:{text}"
    return ""


def _lake_lines(shape_obj):
    from shapely.geometry import LineString

    points = list(getattr(shape_obj, "points", []) or [])
    if not points:
        return []
    parts = list(getattr(shape_obj, "parts", []) or [0])
    parts = parts + [len(points)]
    lines = []
    for start, end in zip(parts, parts[1:]):
        segment = [(float(pt[0]), float(pt[1])) for pt in points[start:end]]
        if len(segment) >= 2:
            line = LineString(segment)
            if not line.is_empty and line.length > 0.0:
                lines.append(line)
    return lines


def _assert_inside(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved != resolved and parent_resolved not in resolved.parents:
        raise RuntimeError(f"Refusing to operate outside {parent_resolved}: {resolved}")
    return resolved


def _handle_remove_readonly(func, path, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove_tree(path: Path, parent: Path) -> None:
    shutil.rmtree(_assert_inside(path, parent), onerror=_handle_remove_readonly)


def _polygon_parts(poly: Polygon) -> list[list[tuple[float, float]]]:
    parts = [[(float(x), float(y)) for x, y in poly.exterior.coords]]
    parts.extend([[(float(x), float(y)) for x, y in interior.coords] for interior in poly.interiors])
    return parts


def _write_lake_polygons() -> Path:
    tlm_root = _tlm_root()
    source = tlm_root / "TLM_GEWAESSER" / f"{LAKE_SOURCE_BASE}.shp"
    out_base = tlm_root / "TLM_GEWAESSER" / LAKE_POLYGON_BASE
    out_shp = out_base.with_suffix(".shp")
    if out_shp.exists():
        print(f"Keeping existing lake polygon shapefile: {out_shp}")
        return out_shp
    if not source.exists():
        raise FileNotFoundError(f"No lake source shapefile found: {source}")

    reader = shapefile.Reader(str(source), encoding="latin1", encodingErrors="replace")
    field_names = [field[0] for field in reader.fields if field[0] != "DeletionFlag"]
    grouped = defaultdict(list)
    names = defaultdict(list)
    object_types = defaultdict(list)
    numbers = defaultdict(list)

    for shape_record in reader.iterShapeRecords():
        record = _record_dict(field_names, shape_record.record)
        group_key = _water_group_key(record)
        if not group_key:
            continue
        grouped[group_key].extend(_lake_lines(shape_record.shape))
        for field, target in (("NAME", names), ("OBJEKTART", object_types), ("GEWISS_NR", numbers)):
            text = _text_value(record, field)
            if text:
                target[group_key].append(text)

    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        path = out_base.with_suffix(suffix)
        if path.exists():
            path.unlink()

    writer = shapefile.Writer(str(out_base), shapeType=shapefile.POLYGON, encoding="latin1")
    writer.autoBalance = 1
    writer.field("GROUP_KEY", "C", size=120)
    writer.field("NAME", "C", size=254)
    writer.field("GEWISS_NR", "C", size=20)
    writer.field("OBJEKTART", "C", size=50)

    polygon_count = 0
    for group_key, lines in grouped.items():
        if not lines:
            continue
        for geom in polygonize(unary_union(lines)):
            if geom.is_empty or geom.area <= 0.0:
                continue
            if geom.geom_type != "Polygon":
                continue
            writer.poly(_polygon_parts(geom))
            writer.record(
                group_key,
                Counter(names[group_key]).most_common(1)[0][0] if names[group_key] else "",
                Counter(numbers[group_key]).most_common(1)[0][0] if numbers[group_key] else "",
                Counter(object_types[group_key]).most_common(1)[0][0] if object_types[group_key] else "",
            )
            polygon_count += 1
    writer.close()

    if source.with_suffix(".prj").exists():
        shutil.copy2(source.with_suffix(".prj"), out_base.with_suffix(".prj"))
    if source.with_suffix(".cpg").exists():
        shutil.copy2(source.with_suffix(".cpg"), out_base.with_suffix(".cpg"))

    print(f"Created lake polygon shapefile with {polygon_count:,} feature(s): {out_shp}")
    return out_shp


def _write_bridge_protection() -> Path:
    tlm_root = _tlm_root()
    source_paths = [tlm_root / folder / filename for folder, filename in TRANSPORT_SOURCES]
    source_paths = [path for path in source_paths if path.exists()]
    out_base = tlm_root / "TLM_BRIDGES" / BRIDGE_BASE
    existing_out = out_base.with_suffix(".shp")
    if existing_out.exists():
        print(f"Keeping existing bridge protection shapefile: {existing_out}")
        return existing_out
    if not source_paths:
        raise FileNotFoundError("No transport source shapefiles found for bridge protection export.")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        path = out_base.with_suffix(suffix)
        if path.exists():
            path.unlink()

    writer = shapefile.Writer(str(out_base), shapeType=shapefile.POLYLINEZ)
    writer.autoBalance = 1
    writer.field("SOURCE", "C", size=16)
    writer.field("KUNSTBAUTE", "C", size=80)

    bridge_count = 0
    copied_prj = False
    copied_cpg = False
    for shp_path in source_paths:
        reader = shapefile.Reader(str(shp_path), encoding="latin1", encodingErrors="replace")
        fields = [field[0] for field in reader.fields if field[0] != "DeletionFlag"]
        if "KUNSTBAUTE" not in fields:
            continue
        kunst_idx = fields.index("KUNSTBAUTE")
        for idx, record in enumerate(reader.iterRecords()):
            kunstbaute = record[kunst_idx]
            if not _is_bridge(kunstbaute):
                continue
            writer.shape(reader.shape(idx))
            writer.record(shp_path.stem, str(kunstbaute))
            bridge_count += 1

        if not copied_prj and shp_path.with_suffix(".prj").exists():
            shutil.copy2(shp_path.with_suffix(".prj"), out_base.with_suffix(".prj"))
            copied_prj = True
        if not copied_cpg and shp_path.with_suffix(".cpg").exists():
            shutil.copy2(shp_path.with_suffix(".cpg"), out_base.with_suffix(".cpg"))
            copied_cpg = True

    writer.close()
    print(f"Created bridge protection shapefile with {bridge_count:,} feature(s): {out_base.with_suffix('.shp')}")
    return out_base.with_suffix(".shp")


def _prune_geometry() -> None:
    _assert_inside(GEOMETRY, ROOT)

    try:
        tlm_root = _tlm_root()
    except FileNotFoundError:
        return

    for child in tlm_root.iterdir():
        if child.is_dir() and child.name not in KEEP_DIRS:
            _remove_tree(child, tlm_root)

    for keep_dir in KEEP_DIRS:
        path = tlm_root / keep_dir
        if not path.exists():
            continue
        for file_path in path.iterdir():
            if not file_path.is_file():
                continue
            keep_suffixes = KEEP_SUFFIXES.get(file_path.stem, set())
            if file_path.name.endswith(".lock") or file_path.stem not in KEEP_BASES or file_path.suffix.lower() not in keep_suffixes:
                _assert_inside(file_path, path).unlink()


def main() -> None:
    _write_lake_polygons()
    _write_bridge_protection()
    _prune_geometry()
    total = sum(path.stat().st_size for path in GEOMETRY.rglob("*") if path.is_file())
    count = sum(1 for path in GEOMETRY.rglob("*") if path.is_file())
    print(f"reference_data now contains {count:,} file(s), {total / (1024 ** 2):.1f} MiB")


if __name__ == "__main__":
    main()
