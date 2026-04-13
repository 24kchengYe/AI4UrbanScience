# Data directory

This folder holds all inputs and outputs of the AI4UrbanScience experiments.
Different sub-directories are distributed differently:

| Sub-directory | Where it lives | Size |
|---|---|---|
| `examples/`   | **Committed to GitHub** — a small curated subset of replicates so reviewers can run the analysis scripts without incurring API cost | < 5 MB |
| `generated/`  | **Figshare only** — the full generated datasets (all models, all prompts, all replicates). Downloadable from the Figshare record below | ~370 MB |
| `real_world/` | **Not distributed** — external GIS/statistics rasters and shapefiles. Users configure paths to their own copies via `.env` (see below) | 6+ GB |
| `place_pulse/`| **Not distributed** — MIT Place Pulse 2.0 dataset, download from the original source | ~3 GB |

### Full generated data archive

```
https://doi.org/10.6084/m9.figshare.28910084
```

Download the `AI4UrbanScience-data-v1.zip` file from the Figshare record
above and unzip it **into this directory**. The archive's top-level folder
is `generated/` so after unzip you should have `data/generated/scaling_law/...`
and similar — which is exactly where every experiment script expects its
inputs.

Shell example:

```bash
# from the repository root
cd data/
wget https://ndownloader.figshare.com/files/<file_id> -O AI4UrbanScience-data-v1.zip
unzip AI4UrbanScience-data-v1.zip
```

### Structure of `generated/` after unzip

```
data/generated/
├── scaling_law/<model>/<prompt_slug>/run_NNN.xlsx
│     e.g. gpt-4o/scaling_law.baseline/run_001.xlsx
│           gpt-4o/scaling_law.china_2020/run_001.xlsx
│           gpt-4o/scaling_law.usa/run_001.xlsx
│
├── distance_decay/<model>/<prompt_slug>/<variant?>/run_NNN.xlsx
│     e.g. gpt-4o/distance_decay.baseline_100/run_001.xlsx
│           gpt-4o/distance_decay.city_150rings_2010/shanghai_1990_50km/run_001.xlsx
│           gpt-4o/distance_decay.city_150rings_2010/portland_2000_150km/run_001.xlsx
│     (the `<variant>` sub-directory stores Supplementary Experiment 3
│      city/year/radius combinations — Fig. S10 in the manuscript)
│
├── vitality_attributes/<model>/<prompt_slug>/run_NNN.xlsx
├── vitality_scored/<model>/<prompt_slug>/run_NNN.xlsx
│
├── perception_images/
│   ├── baseline/*.jpg
│   ├── variants/<category>/*.jpg
│   ├── correlational/...       # Supplementary: perception correlational study
│   ├── element_addition/...    # Synthetic intervention (Fig. 3c / Fig. 6)
│   └── expert_evaluation/...   # Blind realism rating study
│
├── perception_alignment/<source>/<file>.csv
│
├── prompt_engineering_symbolic/<experiment_type>/<bp>_<sampling>/
│     SynAlign four-stage pipeline outputs for symbolic domains
│     (scaling law, distance decay, urban vitality)
│
└── prompt_engineering_perceptual/stage<N>/
      SynAlign pipeline outputs for perceptual domain + shared knowledge base
```

A complete manifest mapping every new file back to its original
research-time location is in `MANIFEST.md` (produced at the same time as
the generated data archive).

## External benchmark datasets (not redistributed)

These datasets are required by Experiments 04, 06 and 07 but are not
re-distributed here because they are already openly available from their
original publishers:

| Dataset | Used by | Source |
|---|---|---|
| MIT Place Pulse 2.0               | Exp. 04 | https://www.kaggle.com/datasets/ericpdavies/place-pulse-2-0 |
| BCL China City Boundary           | Exp. 06 | https://www.beijingcitylab.com/data-released-1/ |
| PopSE China 2020 100 m            | Exp. 06 | https://doi.org/10.11888/Geogra.tpdc.271936 |
| Global real GDP 1992-2019         | Exp. 06 | https://doi.org/10.6084/m9.figshare.17004523 |
| OSM 2018 China roads              | Exp. 06 | http://download.geofabrik.de/asia/china.html |
| CMAB 2024 building footprints     | Exp. 06 | https://zenodo.org/record/11170381 |
| GAIA impervious surface (China)   | Exp. 06 | http://data.ess.tsinghua.edu.cn/ |
| geohey 2020 natural blocks        | Exp. 06 | https://geohey.com/ |

After downloading the ones you need, declare their absolute paths in the
repository's `.env` file (copy `.env.example` to `.env` first):

```
PLACE_PULSE_DIR=/abs/path/to/place_pulse_v2
REAL_DATA_CITY_BOUNDARY_PATH=/abs/path/to/city_boundaries.shp
REAL_DATA_POP_RASTER_PATH=/abs/path/to/PopSE_China2020_100m.tif
REAL_DATA_GDP_RASTER_PATH=/abs/path/to/global_real_gdp_2019.tif
REAL_DATA_ROAD_GDB_PATH=/abs/path/to/gis_osm_roads_free_2018.shp
REAL_DATA_BUILDING_BASE_PATH=/abs/path/to/CMAB_postprocess
REAL_DATA_IMPERVIOUS_RASTER_PATH=/abs/path/to/GAIA_china.tif
REAL_DATA_BLOCK_VECTOR_PATH=/abs/path/to/geohey_natural_block_wgs84.shp
```

Scripts that need these paths will fail with a clear message pointing
back to this README if the variable is missing, so you can set them
incrementally as you run specific experiments.

## Regenerating the data from scratch

If you want to regenerate instead of downloading, every
`experiments/*/generate*.py` script accepts the same flags and will write
its output into `data/generated/<experiment>/...` — the exact layout
above. Running all symbolic experiments end-to-end takes roughly 6-12
hours and 20-60 USD in API spending, depending on your provider.
