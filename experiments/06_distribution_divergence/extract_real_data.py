"""Experiment 06 — extract real-world city attributes from shapefiles.

Collapses the two legacy scripts ``10RealData.py`` and ``10RealData2.py``
into a single CLI. The external geospatial datasets are declared via
environment variables in ``.env`` (see ``data/README.md``).

The output is a CSV with columns ``city, Population, GDP, Infrastructure``.
"""

from __future__ import annotations

# --- path bootstrap so `python experiments/.../foo.py` works without pip install ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# --- end bootstrap ---

import argparse
import logging
from pathlib import Path

import pandas as pd

from ai4us import config

log = logging.getLogger("divergence.extract_real_data")


def _require(name: str) -> Path:
    value = config.env(name)
    if not value:
        raise SystemExit(
            f"Environment variable {name} is not set. Configure it in .env "
            f"(see data/README.md for where to obtain the dataset)."
        )
    return Path(value)


def extract() -> pd.DataFrame:
    """Extract per-city attributes from the configured external datasets.

    This is a thin wrapper around ``geopandas`` + ``rasterstats`` that
    performs zonal statistics for each city boundary using population, GDP
    and road/building rasters. The original implementation in
    ``10RealData.py`` / ``10RealData2.py`` has been collapsed here without
    changing the geospatial logic — the only differences are:

    * paths come from environment variables rather than being hard-coded,
    * the output is a tidy DataFrame with a consistent schema.
    """
    try:
        import geopandas as gpd
        import rasterio  # noqa: F401
        from rasterstats import zonal_stats
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            f"Geospatial dependencies missing ({e}). "
            f"Install with: pip install geopandas rasterio rasterstats"
        )

    boundary_path = _require("REAL_DATA_CITY_BOUNDARY_PATH")
    pop_raster = _require("REAL_DATA_POP_RASTER_PATH")
    gdp_raster = _require("REAL_DATA_GDP_RASTER_PATH")

    log.info("reading city boundaries from %s", boundary_path)
    cities = gpd.read_file(boundary_path)
    log.info("loaded %d city polygons", len(cities))

    log.info("computing population zonal stats")
    pop = zonal_stats(cities, str(pop_raster), stats=["sum"])

    log.info("computing GDP zonal stats")
    gdp = zonal_stats(cities, str(gdp_raster), stats=["sum"])

    df = pd.DataFrame({
        "city": cities.get("name", cities.index.astype(str)).astype(str),
        "Population": [r["sum"] for r in pop],
        "GDP": [r["sum"] for r in gdp],
    })

    # Infrastructure: fall back to placeholder if the optional raster is missing
    infra_env = config.env("REAL_DATA_IMPERVIOUS_RASTER_PATH")
    if infra_env:
        log.info("computing impervious-surface zonal stats")
        infra = zonal_stats(cities, infra_env, stats=["sum"])
        df["Infrastructure"] = [r["sum"] for r in infra]
    else:
        log.warning("no impervious raster configured; Infrastructure column left empty")
        df["Infrastructure"] = pd.NA

    return df.dropna(how="any", subset=["Population", "GDP"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path,
                        default=config.paths.real_world / "chinese_cities.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    df = extract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("extracted %d cities -> %s", len(df), args.output)


if __name__ == "__main__":
    main()
