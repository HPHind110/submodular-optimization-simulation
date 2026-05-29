"""OpenStreetMap data collection and preprocessing utilities.

This module downloads OSM features for real-world geospatial experiments,
normalizes them to point geometries, projects them to a metric CRS, and saves
compact processed CSV files for later experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import geopandas as gpd
import osmnx as ox
import pandas as pd
import certifi
import requests
import urllib3
from shapely.geometry import Point


DEFAULT_PLACE: Final[str] = "Hoan Kiem, Hanoi, Vietnam"
DEFAULT_OUTPUT_DIR: Final[Path] = Path("data") / "processed"
DEMAND_FILENAME: Final[str] = "demand_points.csv"
CANDIDATE_FILENAME: Final[str] = "candidate_points.csv"

DEMAND_TAG_VALUES: Final[dict[str, list[str]]] = {
    "amenity": [
        "school",
        "hospital",
        "clinic",
        "university",
        "library",
        "restaurant",
        "cafe",
    ],
    "tourism": ["attraction", "museum"],
    "highway": ["bus_stop"],
    "public_transport": ["platform"],
}

CANDIDATE_TAG_VALUES: Final[dict[str, list[str]]] = {
    "highway": ["bus_stop"],
    "public_transport": ["platform"],
}

ox.settings.requests_kwargs = {
    **ox.settings.requests_kwargs,
    "verify": certifi.where(),
}


def _download_features_from_place(place_name: str, tags: dict[str, bool]) -> gpd.GeoDataFrame:
    """Download OSM features, retrying SSL certificate failures once."""

    try:
        return ox.features_from_place(place_name, tags=tags)
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        previous_kwargs = dict(ox.settings.requests_kwargs)
        ox.settings.requests_kwargs = {**previous_kwargs, "verify": False}
        try:
            return ox.features_from_place(place_name, tags=tags)
        finally:
            ox.settings.requests_kwargs = previous_kwargs


def _features_from_place(place_name: str, tags: dict[str, list[str]]) -> gpd.GeoDataFrame:
    """Download OSM features for ``place_name`` and filter to exact tag values."""

    query_tags = {key: True for key in tags}
    features = _download_features_from_place(place_name, query_tags)
    if features.empty:
        return gpd.GeoDataFrame(columns=["source_type", "geometry"], geometry="geometry", crs="EPSG:4326")

    rows: list[gpd.GeoDataFrame] = []
    for tag_key, allowed_values in tags.items():
        if tag_key not in features.columns:
            continue
        mask = features[tag_key].isin(allowed_values)
        subset = features.loc[mask, [tag_key, "geometry"]].copy()
        if subset.empty:
            continue
        subset["source_type"] = tag_key + ":" + subset[tag_key].astype(str)
        rows.append(subset[["source_type", "geometry"]])

    if not rows:
        return gpd.GeoDataFrame(columns=["source_type", "geometry"], geometry="geometry", crs=features.crs)

    result = pd.concat(rows, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=features.crs)


def _geometry_to_point(geometry: object) -> Point | None:
    """Convert a supported geometry to a representative point."""

    if geometry is None:
        return None
    if getattr(geometry, "is_empty", True):
        return None
    if isinstance(geometry, Point):
        return geometry
    if hasattr(geometry, "representative_point"):
        point = geometry.representative_point()
        if isinstance(point, Point) and not point.is_empty:
            return point
    if hasattr(geometry, "centroid"):
        point = geometry.centroid
        if isinstance(point, Point) and not point.is_empty:
            return point
    return None


def _normalize_to_points(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame containing only point geometries."""

    if features.empty:
        return gpd.GeoDataFrame(columns=["source_type", "geometry"], geometry="geometry", crs=features.crs)

    normalized = features[["source_type", "geometry"]].copy()
    normalized["geometry"] = normalized["geometry"].map(_geometry_to_point)
    normalized = normalized.dropna(subset=["geometry"])
    return gpd.GeoDataFrame(normalized, geometry="geometry", crs=features.crs)


def _deduplicate_nearby_points(
    points: gpd.GeoDataFrame,
    distance_meters: float = 10.0,
) -> gpd.GeoDataFrame:
    """Drop later points that fall in the same metric grid cell."""

    if points.empty:
        return points
    if distance_meters <= 0:
        return points.drop_duplicates(subset=["geometry"]).copy()

    projected = ox.projection.project_gdf(points)
    grid_x = (projected.geometry.x / distance_meters).round().astype("int64")
    grid_y = (projected.geometry.y / distance_meters).round().astype("int64")
    deduped_projected = projected.assign(_grid_x=grid_x, _grid_y=grid_y)
    deduped_projected = deduped_projected.drop_duplicates(subset=["_grid_x", "_grid_y"])
    deduped_projected = deduped_projected.drop(columns=["_grid_x", "_grid_y"])
    return deduped_projected.to_crs(points.crs)


def _limit_points(points: gpd.GeoDataFrame, max_points: int | None, seed: int) -> gpd.GeoDataFrame:
    """Deterministically sample at most ``max_points`` rows."""

    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points.reset_index(drop=True)
    return points.sample(n=max_points, random_state=seed).reset_index(drop=True)


def _fallback_candidates_from_demand(
    demand_points: gpd.GeoDataFrame,
    min_candidates: int,
    max_candidates: int | None,
    seed: int,
) -> gpd.GeoDataFrame:
    """Sample demand points as candidate points when OSM candidates are sparse."""

    if demand_points.empty:
        return gpd.GeoDataFrame(columns=["source_type", "geometry"], geometry="geometry", crs=demand_points.crs)

    desired_count = min_candidates
    if max_candidates is not None and max_candidates > 0:
        desired_count = min(max_candidates, max(min_candidates, min(len(demand_points), max_candidates)))
    desired_count = min(desired_count, len(demand_points))

    sampled = demand_points.sample(n=desired_count, random_state=seed).copy()
    sampled["source_type"] = "fallback_demand_sample"
    return sampled[["source_type", "geometry"]].reset_index(drop=True)


def _to_processed_table(points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Build the processed CSV table with lon/lat and projected x/y columns."""

    if points.empty:
        return pd.DataFrame(columns=["id", "x", "y", "lon", "lat", "source_type"])

    geographic = points.to_crs("EPSG:4326")
    projected = ox.projection.project_gdf(geographic)

    table = pd.DataFrame(
        {
            "id": range(len(projected)),
            "x": projected.geometry.x.to_numpy(),
            "y": projected.geometry.y.to_numpy(),
            "lon": geographic.geometry.x.to_numpy(),
            "lat": geographic.geometry.y.to_numpy(),
            "source_type": geographic["source_type"].astype(str).to_numpy(),
        }
    )
    return table[["id", "x", "y", "lon", "lat", "source_type"]]


def preprocess_points(
    points: gpd.GeoDataFrame,
    max_points: int | None = None,
    seed: int = 42,
    dedupe_distance_meters: float = 10.0,
) -> pd.DataFrame:
    """Normalize, deduplicate, sample, project, and return a processed table."""

    normalized = _normalize_to_points(points)
    deduped = _deduplicate_nearby_points(normalized, distance_meters=dedupe_distance_meters)
    limited = _limit_points(deduped, max_points=max_points, seed=seed)
    return _to_processed_table(limited)


def collect_osm_points(
    place_name: str = DEFAULT_PLACE,
    max_demand: int | None = None,
    max_candidates: int | None = None,
    seed: int = 42,
    min_candidates: int = 10,
    dedupe_distance_meters: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and preprocess demand and candidate points from OSM."""

    demand_raw = _features_from_place(place_name, DEMAND_TAG_VALUES)
    candidate_raw = _features_from_place(place_name, CANDIDATE_TAG_VALUES)

    demand_normalized = _normalize_to_points(demand_raw)
    demand_deduped = _deduplicate_nearby_points(
        demand_normalized,
        distance_meters=dedupe_distance_meters,
    )

    candidate_normalized = _normalize_to_points(candidate_raw)
    candidate_deduped = _deduplicate_nearby_points(
        candidate_normalized,
        distance_meters=dedupe_distance_meters,
    )

    if len(candidate_deduped) < min_candidates:
        candidate_deduped = _fallback_candidates_from_demand(
            demand_deduped,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            seed=seed,
        )

    demand_limited = _limit_points(demand_deduped, max_points=max_demand, seed=seed)
    candidate_limited = _limit_points(candidate_deduped, max_points=max_candidates, seed=seed)

    demand_table = _to_processed_table(demand_limited)
    candidate_table = _to_processed_table(candidate_limited)
    return demand_table, candidate_table


def save_processed_data(
    demand_points: pd.DataFrame,
    candidate_points: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Save processed demand and candidate CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    demand_path = output_path / DEMAND_FILENAME
    candidate_path = output_path / CANDIDATE_FILENAME
    demand_points.to_csv(demand_path, index=False)
    candidate_points.to_csv(candidate_path, index=False)
    return demand_path, candidate_path


def load_processed_points(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed demand and candidate CSV files from disk."""

    output_path = Path(output_dir)
    demand_path = output_path / DEMAND_FILENAME
    candidate_path = output_path / CANDIDATE_FILENAME

    if not demand_path.exists() or not candidate_path.exists():
        raise FileNotFoundError(
            "Processed OSM CSV files are missing. Expected "
            f"{demand_path} and {candidate_path}."
        )

    return pd.read_csv(demand_path), pd.read_csv(candidate_path)
