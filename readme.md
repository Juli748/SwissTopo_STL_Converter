# SwissTopo STL Converter

This project is now GUI-first and optimized around a simple path:

1. Select a SwissTopo CSV or place terrain tiles into `data/`
2. Choose the final model size and a detail preset
3. Click `Run Full Pipeline`

The app can still handle advanced border clipping, lake lowering, scale overrides, and manual downsampling, but those settings are hidden by default.

## Start

Run:

```bash
python gui.py
```

The GUI now shows:

- A pipeline overview with detected inputs, tile STLs, and final STL status
- A simplified conversion section with `Draft`, `Balanced`, `Fine`, and `Custom`
- A simplified merge section focused on the final STL path and printable base
- A `Run Full Pipeline` button that can download, convert, and merge in sequence

## Recommended Workflow

### Option A: Start from a SwissTopo CSV

1. Export the SwissTopo download CSV
2. Open `gui.py`
3. Select the CSV
4. Set the model name
5. Choose the final model size in mm
6. Leave the detail preset on `Balanced` unless you need faster or finer output
7. Leave `Add printable base` enabled for 3D printing
8. Click `Run Full Pipeline`

### Option B: Start from already-downloaded files

Put your terrain files here:

- `data/xyz/*.xyz`
- `data/tif/*.tif`
- `data/tif/*.tiff`

Then open the GUI and run:

1. `Create STL Tiles`
2. `Build Final STL`

## What The Presets Mean

- `Draft`: faster conversion, lighter STL files
- `Balanced`: default choice for most prints
- `Fine`: keeps more terrain detail, slower and larger output
- `Custom`: unlocks manual step size, alternate scaling modes, tolerance, and worker count

## Advanced Features

Open the advanced panels only if you need them.

Available advanced conversion controls:

- Manual `--step`
- Fixed tile size scaling
- Scale ratio input like `1:100`
- Grid tolerance for noisy XYZ data
- Worker count

Available advanced merge controls:

- Weld tolerance
- Merge-only Z scaling
- Lake lowering
- Base mode and base Z override
- Border clipping using Swiss boundary shapefiles
- Canton/bezirk filtering

## SwissTopo CSV

Typical flow for SwissTopo:

1. Open the swissALTI3D page
2. Select the tiles you want
3. Export the CSV of download URLs
4. Use that CSV in the GUI

Reference:

```text
https://www.swisstopo.admin.ch/de/hoehenmodell-swissalti3d
```

## Folders

```text
project/
  gui.py
  build_stl.py
  download_tiles.py
  data/
    your_urls.csv
    xyz/
    tif/
  output/
    tiles/
    terrain.stl
  geometry_data/
```

## Dependencies

Minimum:

```bash
pip install numpy
```

Optional:

```bash
pip install rasterio
pip install scipy
pip install shapely pyshp
```

Use these only if needed:

- `rasterio` for GeoTIFF/COG input
- `scipy` for triangulation fallback on non-grid XYZ data
- `shapely` and `pyshp` for border clipping and lake lowering

## CLI

The CLI still works if you want direct control:

```bash
python download_tiles.py --csv path/to/urls.csv
python build_stl.py --all --target-size-mm 150
python build_stl.py --merge-stl output/terrain.stl --make-solid
```

## Notes

- `Balanced` plus a final model size is the intended default for most users
- Keep advanced settings hidden unless you actually need them
- Border clipping and lake lowering depend on the shapefiles in `geometry_data/`
