# Experiment 06 — Data Distribution Divergence (Fig. 4)

Quantify how much the GenAI-produced "mirage cities" depart from real-world
Chinese city statistics, using three metrics:

* **MAE** on bin-level relative frequencies
* **Overlap Ratio** (OR)
* **Jensen-Shannon Divergence** (JSD)

## Pipeline

```
extract_real_data.py  → data/real_world/chinese_cities.csv
compute_divergence.py → results/distribution_divergence/metrics.csv
figure4.py            → results/figures/figure4.png + .pdf
```

## Data prerequisites

Several external rasters/shapefiles are required. Declare their local paths
in `.env` (see `data/README.md` for download instructions):

```
REAL_DATA_CITY_BOUNDARY_PATH=/abs/path/to/city_boundaries.shp
REAL_DATA_POP_RASTER_PATH=/abs/path/to/PopSE_China2020_100m.tif
REAL_DATA_GDP_RASTER_PATH=/abs/path/to/global_real_gdp_2019.tif
REAL_DATA_IMPERVIOUS_RASTER_PATH=/abs/path/to/GAIA_china.tif
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `10RealData.py`                                   | `extract_real_data.py` |
| `10RealData2.py`                                  | `extract_real_data.py` (consolidated) |
| `11datadifference.py`                             | `compute_divergence.py` + `figure4.py` |
| `11datadifference2.py`                            | `compute_divergence.py` (variant collapsed) |
| `11数据降维可视化.py`                              | ad-hoc visualization, not essential to Fig. 4 |
| `12realdata-TheoryValidMultitime(Zipf-Scaling).py`| superseded by `analyze.py` in Exp 01 |
| `12真实 生成和提示生成的数据分布可视化.py`          | superseded by `figure4.py` |
| `11datadifference-naturecities改稿用-图4.py`       | superseded by `figure4.py` |
