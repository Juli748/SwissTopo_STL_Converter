import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from download_buildings import _lv95_bounds_to_wgs84, bounds_from_csv


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_OUTPUT = Path("work") / "water" / "riverbanks.geojson"


def _wgs84_to_lv95(lon: float, lat: float) -> tuple[float, float]:
    lat_aux = (lat * 3600.0 - 169028.66) / 10000.0
    lon_aux = (lon * 3600.0 - 26782.5) / 10000.0
    east = 2600072.37 + 211455.93 * lon_aux - 10938.51 * lon_aux * lat_aux - 0.36 * lon_aux * lat_aux**2 - 44.54 * lon_aux**3
    north = 1200147.07 + 308807.95 * lat_aux + 3745.25 * lon_aux**2 + 76.63 * lat_aux**2 - 194.56 * lon_aux**2 * lat_aux + 119.79 * lat_aux**3
    return east, north


def _line_coordinates(geometry: list[dict]) -> list[tuple[float, float]]:
    return [_wgs84_to_lv95(float(point["lon"]), float(point["lat"])) for point in geometry]


def _query_river_areas(bounds_wgs84: tuple[float, float, float, float]) -> dict:
    west, south, east, north = bounds_wgs84
    bbox = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    query = f"""
[out:json][timeout:180];
(
  way[\"natural\"=\"water\"][\"water\"=\"river\"]({bbox});
  way[\"waterway\"=\"riverbank\"]({bbox});
  relation[\"type\"=\"multipolygon\"][\"natural\"=\"water\"][\"water\"=\"river\"]({bbox});
  relation[\"type\"=\"multipolygon\"][\"waterway\"=\"riverbank\"]({bbox});
);
out geom;
"""
    request = Request(
        OVERPASS_URL,
        data=urlencode({"data": query}).encode("utf-8"),
        headers={"User-Agent": "SwissTopo-STL-Converter/1.0"},
    )
    with urlopen(request, timeout=240) as response:
        return json.loads(response.read().decode("utf-8"))


def _feature_collection(overpass: dict) -> dict:
    features = []
    for element in overpass.get("elements", []):
        if element.get("type") == "way":
            coords = _line_coordinates(element.get("geometry") or [])
            if len(coords) >= 4 and coords[0] == coords[-1]:
                features.append({
                    "type": "Feature",
                    "properties": {"osm_id": element.get("id"), "source": "OpenStreetMap"},
                    "geometry": {"type": "Polygon", "coordinates": [[list(point) for point in coords]]},
                })
    try:
        from shapely.geometry import LineString, mapping
        from shapely.ops import polygonize, unary_union
    except Exception:
        LineString = None
    for element in overpass.get("elements", []):
        if element.get("type") != "relation" or LineString is None:
            continue
        outer_lines = []
        inner_lines = []
        for member in element.get("members") or []:
            coords = _line_coordinates(member.get("geometry") or [])
            if len(coords) < 2:
                continue
            if member.get("role") == "inner":
                inner_lines.append(LineString(coords))
            else:
                outer_lines.append(LineString(coords))
        if not outer_lines:
            continue
        outer_polygons = list(polygonize(unary_union(outer_lines)))
        inner_union = unary_union(list(polygonize(unary_union(inner_lines)))) if inner_lines else None
        for polygon in outer_polygons:
            geometry = polygon.difference(inner_union) if inner_union is not None else polygon
            if geometry.is_empty:
                continue
            features.append({
                "type": "Feature",
                "properties": {"osm_relation": element.get("id"), "source": "OpenStreetMap"},
                "geometry": mapping(geometry),
            })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:2056"}},
        "features": features,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download actual OpenStreetMap riverbank polygons for a SwissALTI CSV area.")
    parser.add_argument("--csv", required=True, help="SwissALTI CSV with download URLs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"LV95 riverbank GeoJSON output (default: {DEFAULT_OUTPUT}).")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1
    try:
        bounds_lv95 = bounds_from_csv(csv_path)
    except ValueError as exc:
        print(exc)
        return 1
    bounds_wgs84 = _lv95_bounds_to_wgs84(bounds_lv95)
    print(f"Querying OpenStreetMap riverbank polygons for LV95 bounds: {bounds_lv95}")
    try:
        overpass = _query_river_areas(bounds_wgs84)
    except Exception as exc:
        print(f"Could not download riverbank polygons: {exc}")
        return 1
    collection = _feature_collection(overpass)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection, separators=(",", ":")), encoding="utf-8")
    print(f"[PROGRESS] 1/1 riverbank polygons ({len(collection['features'])} feature(s))")
    print(f"Riverbank outlines ready in: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
