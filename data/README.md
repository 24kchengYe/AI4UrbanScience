# Data directory

This folder holds inputs and intermediate outputs for AI4UrbanScience
experiments. Large files are **not committed** (see `.gitignore`);
download them yourself and set the paths in `.env`.

## Structure

```
data/
├── generated/                         # Outputs of the generate* scripts
│   ├── scaling_law/<model>/<prompt>/run_NNN.xlsx
│   ├── distance_decay/<model>/<prompt>/run_NNN.xlsx
│   ├── vitality_attributes/<model>/<prompt>/run_NNN.xlsx
│   ├── vitality_scored/<model>/<scoring_prompt>/run_NNN.xlsx
│   ├── perception_images/baseline/*.jpg
│   ├── perception_images/variants/<category>/*.jpg
│   ├── perception_alignment/<model>/<dim>.csv
│   ├── prompt_engineering_symbolic/{with,without}_blueprint/run_NNN.xlsx
│   └── prompt_engineering_perceptual/<dim>_with_blueprint.csv
│
├── real_world/                        # Experiments 06 / 07
│   └── chinese_cities.csv             # produced by extract_real_data.py
│
└── place_pulse/                       # Experiment 04
    ├── votes.csv
    └── images/<image_id>.jpg
```

## External datasets and where to get them

| Dataset | Used by | Source |
|---|---|---|
| **MIT Place Pulse 2.0**                  | Exp. 04 | https://www.kaggle.com/datasets/ericpdavies/place-pulse-2-0 |
| **BCL China City Boundary**              | Exp. 06 | https://www.beijingcitylab.com/data-released-1/ |
| **PopSE China 2020 100m**                | Exp. 06 | https://doi.org/10.11888/Geogra.tpdc.271936 |
| **Global real GDP 1992-2019**            | Exp. 06 | https://doi.org/10.6084/m9.figshare.17004523 |
| **OSM 2018 China roads**                 | Exp. 06 | http://download.geofabrik.de/asia/china.html |
| **CMAB 2024 building footprints**        | Exp. 06 | https://zenodo.org/record/11170381 |
| **GAIA impervious surface (China)**      | Exp. 06 | http://data.ess.tsinghua.edu.cn/ |
| **geohey 2020 natural blocks**           | Exp. 06 | https://geohey.com/ |

## Setting environment variables

After downloading, add absolute paths to `.env`:

```
PLACE_PULSE_DIR=/abs/path/to/place_pulse
REAL_DATA_POP_RASTER_PATH=/abs/path/to/PopSE_China2020_100m.tif
REAL_DATA_GDP_RASTER_PATH=/abs/path/to/global_gdp_2019.tif
REAL_DATA_ROAD_GDB_PATH=/abs/path/to/gis_osm_roads_free_2018.shp
REAL_DATA_BUILDING_BASE_PATH=/abs/path/to/CMAB_postprocess
REAL_DATA_IMPERVIOUS_RASTER_PATH=/abs/path/to/GAIA_china.tif
REAL_DATA_BLOCK_VECTOR_PATH=/abs/path/to/geohey_natural_block_wgs84.shp
REAL_DATA_CITY_BOUNDARY_PATH=/abs/path/to/city_boundary.shp
```

If a variable is missing, the relevant script will either fall back to
the placeholder default (and print a warning) or refuse to start with a
clear message pointing to this README.
