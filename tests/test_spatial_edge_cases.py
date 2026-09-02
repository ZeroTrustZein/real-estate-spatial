"""Edge cases and boundary tests for spatial computations, indexing, caching, and serialization."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from real_estate_spatial.spatial.coordinates import (
    EARTH_RADIUS_KM,
    bounding_box,
    haversine_distance,
    haversine_distance_vectorized,
    point_in_bbox,
)
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.spatial.index import GridIndex, KDTreeIndex
from real_estate_spatial.spatial.persistence import SpatialIndexSerializer, SpatialQueryCache
from real_estate_spatial.types import BoundingBox, Coordinates, Property


class TestSpatialCoordinatesEdgeCases:
    """Test boundary conditions, antipodal coordinates, and edge cases in spatial math."""

    def test_antipodal_points_distance(self) -> None:
        """Distance between exact antipodal points should be half Earth circumference."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 180.0
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        expected = math.pi * EARTH_RADIUS_KM
        assert dist == pytest.approx(expected, rel=1e-4)

    def test_north_pole_to_south_pole(self) -> None:
        """Distance from North Pole to South Pole."""
        dist = haversine_distance(90.0, 0.0, -90.0, 0.0)
        expected = math.pi * EARTH_RADIUS_KM
        assert dist == pytest.approx(expected, rel=1e-4)

    def test_cross_antimeridian_distance(self) -> None:
        """Distance crossing the 180-degree meridian."""
        dist = haversine_distance(0.0, 179.9, 0.0, -179.9)
        # Difference in longitude is 0.2 degrees at equator
        expected = (0.2 / 360.0) * (2 * math.pi * EARTH_RADIUS_KM)
        assert dist == pytest.approx(expected, rel=1e-3)

    def test_vectorized_haversine_broadcasting_and_empty(self) -> None:
        """Vectorized haversine handles empty arrays and scalar-to-array broadcasting."""
        # Empty arrays
        empty_res = haversine_distance_vectorized(37.77, -122.42, np.array([]), np.array([]))
        assert len(empty_res) == 0

        # Incompatible lengths raise ValueError
        with pytest.raises(ValueError):
            haversine_distance_vectorized(
                37.77, -122.42, np.array([37.1, 37.2]), np.array([-122.1, -122.2, -122.3])
            )

    def test_bounding_box_at_equator_and_poles(self) -> None:
        """Bounding box computation at equator vs extreme high latitudes."""
        min_lat, min_lon, max_lat, max_lon = bounding_box(0.0, 0.0, 111.0)  # ~1 degree
        assert min_lat < 0.0 < max_lat
        assert min_lon < 0.0 < max_lon

        bbox_tuple = (min_lat, min_lon, max_lat, max_lon)
        assert point_in_bbox(0.0, 0.0, bbox_tuple) is True
        assert point_in_bbox(50.0, 50.0, bbox_tuple) is False

    def test_bounding_box_negative_radius(self) -> None:
        """Negative radius should be clamped to zero."""
        min_lat, min_lon, max_lat, max_lon = bounding_box(37.77, -122.41, -50.0)
        assert min_lat == pytest.approx(37.77)
        assert max_lat == pytest.approx(37.77)
        assert min_lon == pytest.approx(-122.41)
        assert max_lon == pytest.approx(-122.41)


class TestSpatialIndexEdgeCases:
    """Test spatial indexing with k > n, zero radius, empty datasets, and duplicate coordinates."""

    def test_k_greater_than_n_properties(self, sample_properties: List[Property]) -> None:
        """Querying k > len(dataset) returns all available properties sorted by distance."""
        n = len(sample_properties)
        query_coord = Coordinates(lat=37.7749, lon=-122.4194)

        kdtree = KDTreeIndex(sample_properties)
        kdtree_results = kdtree.query_knn(query_coord, k=n + 50)
        assert len(kdtree_results) == n

        grid = GridIndex(sample_properties)
        grid_results = grid.query_knn(query_coord, k=n + 50)
        assert len(grid_results) == n

        # Check distances are ascending
        distances = [d for _, d in kdtree_results]
        assert distances == sorted(distances)

    def test_radius_zero_and_very_large(self, sample_properties: List[Property]) -> None:
        """Radius=0.0 returns exact match; very large radius returns all points."""
        exact_coord = sample_properties[0].coordinates
        kdtree = KDTreeIndex(sample_properties)

        # Exact match with tiny radius
        res_tiny = kdtree.query_radius(exact_coord, radius_km=0.0001)
        assert len(res_tiny) >= 1
        assert res_tiny[0].id == sample_properties[0].id

        # Huge radius
        res_huge = kdtree.query_radius(exact_coord, radius_km=50000.0)
        assert len(res_huge) == len(sample_properties)

    def test_identical_coordinates_multiple_properties(self) -> None:
        """Handling multiple distinct properties (e.g. condo units) sharing identical lat/lon."""
        coord = Coordinates(lat=37.7800, lon=-122.4000)
        units = [
            Property(
                id=f"CONDO-{i}",
                address=f"100 Unit #{i}",
                coordinates=coord,
                price=500000.0 + i * 10000,
                sqft=800.0,
            )
            for i in range(10)
        ]
        index = KDTreeIndex(units)
        assert index.size() == 10

        # Query KNN for all 10
        results = index.query_knn(coord, k=10)
        assert len(results) == 10
        for _, dist in results:
            assert dist == pytest.approx(0.0, abs=1e-5)

    def test_grid_index_dynamic_bounding_box(self, sample_dataset: List[Property]) -> None:
        """Grid index bbox queries accurately capture points inside box."""
        grid = GridIndex(sample_dataset)
        bbox = BoundingBox(min_lat=37.75, min_lon=-122.45, max_lat=37.80, max_lon=-122.38)
        found = grid.query_bbox(bbox)

        for p in found:
            assert bbox.min_lat <= p.coordinates.lat <= bbox.max_lat
            assert bbox.min_lon <= p.coordinates.lon <= bbox.max_lon


class TestSpatialFeaturesAndExtractor:
    """Test SpatialFeatureExtractor edge cases and isolation."""

    def test_spatial_lag_no_neighbors_returns_default(self, sample_property: Property) -> None:
        """When no valid properties exist, spatial_lag returns 0.0."""
        extractor = SpatialFeatureExtractor()

        # Target coordinate
        far_coord = Coordinates(lat=45.0, lon=-93.0)
        # Empty list of background properties
        lag_val = extractor.spatial_lag(far_coord, properties=[], value_attr="price")
        assert lag_val == 0.0

    def test_local_density_calculations(self, sample_properties: List[Property]) -> None:
        """Verify local density counts properties within various search radii."""
        extractor = SpatialFeatureExtractor()
        center = sample_properties[0].coordinates

        density_close = extractor.local_density(
            center, sample_properties, radius_km=0.5, exclude_id=sample_properties[0].id
        )
        density_wide = extractor.local_density(
            center, sample_properties, radius_km=10.0, exclude_id=sample_properties[0].id
        )

        assert 0 <= density_close <= len(sample_properties) - 1
        assert density_close <= density_wide
        assert density_wide == len(sample_properties) - 1

    def test_batch_feature_extraction(self, sample_dataset: List[Property]) -> None:
        """Extract spatial feature matrices with density, lag, and landmark distance columns."""
        extractor = SpatialFeatureExtractor(
            landmarks={"CBD": Coordinates(lat=37.7749, lon=-122.4194)},
        )
        features_list = [extractor.extract_all(p, sample_dataset) for p in sample_dataset[:10]]
        assert len(features_list) == 10
        for f in features_list:
            assert "dist_CBD_km" in f
            assert "spatial_lag_price_k5" in f
            assert "density_1km" in f


class TestSpatialPersistenceAndCacheEdgeCases:
    """Test index caching, eviction, and file persistence failure modes."""

    def test_lru_cache_hit_rate_and_capacity(self) -> None:
        """Verify query cache hit rates, miss tracking, and size capping."""
        cache = SpatialQueryCache(max_size=3)
        coord = Coordinates(lat=37.77, lon=-122.42)

        # Miss
        assert cache.get_knn(coord, k=5) is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Put and Hit
        fake_result = [(Property(id="P1", address="A1", coordinates=coord, sqft=1000.0), 0.5)]
        cache.set_knn(coord, k=5, results=fake_result)
        hit_res = cache.get_knn(coord, k=5)
        assert hit_res is not None
        assert cache.hits == 1
        assert cache.hit_rate == pytest.approx(0.5)

        # Fill cache past max_size=3
        c1 = Coordinates(lat=37.1, lon=-122.1)
        c2 = Coordinates(lat=37.2, lon=-122.2)
        c3 = Coordinates(lat=37.3, lon=-122.3)
        cache.set_knn(c1, k=5, results=fake_result)
        cache.set_knn(c2, k=5, results=fake_result)
        cache.set_knn(c3, k=5, results=fake_result)

        assert len(cache) <= 3
        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_persistence_serializer_roundtrip(
        self, tmp_path: Path, sample_dataset: List[Property]
    ) -> None:
        """Verify KDTree and Grid indices persist and reload cleanly in JSON and Binary formats."""
        kdtree = KDTreeIndex(sample_dataset)
        grid = GridIndex(sample_dataset)

        # KDTree JSON
        kd_json_path = tmp_path / "kdtree.json"
        SpatialIndexSerializer.save_index(kdtree, kd_json_path, format="json")
        assert kd_json_path.exists()
        loaded_kd_json = SpatialIndexSerializer.load_index(kd_json_path)
        assert len(loaded_kd_json) == len(sample_dataset)

        # Grid JSON
        grid_json_path = tmp_path / "grid.json"
        SpatialIndexSerializer.save_index(grid, grid_json_path, format="json")
        assert grid_json_path.exists()
        loaded_grid_json = SpatialIndexSerializer.load_index(grid_json_path)
        assert len(loaded_grid_json) == len(sample_dataset)

        # Binary format
        kd_bin_path = tmp_path / "kdtree.bin"
        SpatialIndexSerializer.save_index(kdtree, kd_bin_path, format="binary")
        assert kd_bin_path.exists()
        loaded_kd_bin = SpatialIndexSerializer.load_index(kd_bin_path)
        assert len(loaded_kd_bin) == len(sample_dataset)
