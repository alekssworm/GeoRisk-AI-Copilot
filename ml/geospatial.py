from pathlib import Path

import pandas as pd


def points_to_geodataframe(
    frame: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    crs: str = "EPSG:4326",
):
    """Convert tabular location records to a GeoPandas GeoDataFrame.

    GeoPandas is imported lazily so the core ML tests can run even when a user is
    only exercising tabular features. Install project requirements to enable this.
    """

    import geopandas as gpd

    geometry = gpd.points_from_xy(frame[longitude_col], frame[latitude_col])
    return gpd.GeoDataFrame(frame.copy(), geometry=geometry, crs=crs)


def enrich_with_nearest_distance_km(
    frame: pd.DataFrame,
    layer_path: str | Path,
    output_column: str = "distance_to_water_km",
    projected_crs: str = "EPSG:3857",
) -> pd.DataFrame:
    """Attach nearest-feature distance from a vector layer to point records.

    This is useful when replacing the synthetic demo inputs with real monitoring
    points and real hydrology, facility, or exclusion-zone layers.
    """

    import geopandas as gpd

    points = points_to_geodataframe(frame).to_crs(projected_crs)
    layer = gpd.read_file(layer_path).to_crs(projected_crs)
    nearest = gpd.sjoin_nearest(points, layer[["geometry"]], how="left", distance_col="_distance_m")
    enriched = pd.DataFrame(nearest.drop(columns=["geometry", "index_right"], errors="ignore"))
    enriched[output_column] = enriched["_distance_m"].fillna(0) / 1000.0
    return enriched.drop(columns=["_distance_m"], errors="ignore")
