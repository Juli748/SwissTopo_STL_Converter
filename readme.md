# XYZ to STL Terrain Converter

This tool converts terrain point clouds stored as .xyz files into STL surface meshes.
It is designed for large, gridded terrain datasets and supports batch conversion and seamless merging.

---

## Quick Start (recommended workflow)

GUI (guided workflow):
```
python gui.py
```

1) Download XYZ tiles
- Get terrain tiles as XYZ (X Y Z per line).
- Put your CSV of download URLs in `./data` and run:

```
python download_tiles.py
```

- This downloads ZIPs into temp folders and copies all XYZ files into `./data/xyz`.

Swisstopo sources:
- swissALTI3D (DTM): https://www.swisstopo.admin.ch/de/hoehenmodell-swissalti3d
- swissSURFACE3D Raster (DSM): https://www.swisstopo.admin.ch/de/hoehenmodell-swisssurface3d-raster

2) Convert each XYZ to STL (one STL per tile)

```
python build_stl.py --all --step 10
```

Or auto-scale/downsample for a desired print size (smallest edge in mm):

```
python build_stl.py --all --target-size-mm 150
```

3) Merge all STL tiles into one seamless STL

```
python build_stl.py --merge-stl output/terrain.stl --weld-tol 0.001
```

4) (Optional) Make the merged STL printable with a global base

```
python build_stl.py --merge-stl output/terrain.stl --make-solid --base-thickness 10 --weld-tol 0.001
```

The base is added at the lowest point of the merged terrain minus the extra thickness.

---

## Folder Structure

```
project/
  build_stl.py
  download_tiles.py
  data/
    urls.csv
    xyz/
      tile_001.xyz
      tile_002.xyz
  output/
    tiles/
      tile_001.stl
      tile_002.stl
    terrain.stl
```

All batch operations read from `./data/xyz` and write to `./output`.

---

## Input Format

Each .xyz file must be plain text with one point per line:

```
X Y Z
793483.25 1139000.25 3602.40
793483.75 1139000.25 3602.23
...
```

Notes:
- A header line like X Y Z is allowed
- Blank lines and # comments are ignored
- Coordinates should form a complete X/Y grid for best results

---

## Good Command Suggestions

Fast and manageable for large areas:

```
python build_stl.py --all --step 10
python build_stl.py --merge-stl output/terrain.stl --weld-tol 0.001
```

Higher detail (bigger files):

```
python build_stl.py --all --step 4
python build_stl.py --merge-stl output/terrain.stl --weld-tol 0.001
```

Printable with a global base:

```
python build_stl.py --merge-stl output/terrain.stl --make-solid --base-thickness 10 --weld-tol 0.001
```

---

## Modes and Arguments (when to use what)

Batch conversion (many XYZ files):
```
python build_stl.py --all [options]
```
Arguments:
- --all (required): convert every .xyz under ./data/xyz into ./output/tiles
Options you can use here:
- --step N: downsample grid by keeping every Nth point in X and Y
- --tol T: snap X/Y values to a grid of size T for noisy coordinates
- --z-scale S: multiply all Z values (e.g. 2 for exaggeration)
- --target-size-mm M: auto-compute scale and step from all XYZs so the smallest edge is M mm
- --target-resolution-mm R: target XY spacing in the final STL when using --target-size-mm
Notes:
- --make-solid is ignored in batch mode (global base is added after merging).

Merge STL tiles:
```
python build_stl.py --merge-stl output/terrain.stl [options]
```
Arguments:
- --merge-stl (required): output STL path for the merged file; merges all .stl under ./output/tiles
Options you can use here:
- --weld-tol T: weld shared vertices within tolerance to remove seams
- --make-solid: add a global base and side walls to the merged terrain
- --base-thickness T: set base plane to minZ - T for the merged terrain
- --base-z Z: set an explicit base Z (overrides --base-thickness)
- --merge-z-scale S: multiply Z by this factor for the merged STL

---

## Merge Behavior (seamless joins)

Merging always loads all STL tiles, welds shared vertices, and writes a single STL.
This removes tiny gaps at tile borders and is required for a clean global base.

---

## Troubleshooting

Grid not detected:
- Try --tol 0.001
- Check for missing points in the XYZ file

Very large STL:
- Increase --step

---

## License

Free to use, modify, and adapt for terrain processing, GIS, CAD, or 3D printing workflows.
