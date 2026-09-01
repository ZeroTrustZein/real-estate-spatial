"""Comprehensive tests for spatial indexing, coordinates, and feature extraction."""

from __future__ import annotations

import math
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
from real_estate_spatial.types import BoundingBox, Coordinates, Property


class TestCoordinateCalculations:
    """Test geodesic distance and bounding box calculations."""

    def test_haversine_same_point(self) -> None:
        """Distance between identical coordinates is zero."""
        assert haversine_distance(37.7749, -122.4194, 37.7749, -122.4194) == 0.0
        assert haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0
        assert haversine_distance(90.0, 0.0, 90.0, 0.0) == 0.0

    def test_haversine_symmetry(self) -> None:
        """Distance from A to B equals distance from B to A."""
        lat1, lon1 = 37.7749, -122.4194  # San Francisco
        lat2, lon2 = 34.0522, -118.2437  # Los Angeles
        d1 = haversine_distance(lat1, lon1, lat2, lon2)
        d2 = haversine_distance(lat2, lon2, lat1, lon1)
        assert d1 == pytest.approx(d2, rel=1e-9)
        # SF to LA is approximately 559 km
        assert 550.0 < d1 < 570.0

    def test_haversine_poles_and_antimeridian(self) -> None:
        """Test edge coordinates like North Pole to South Pole and antimeridian crossing."""
        # Pole to pole along meridian = pi * R ≈ 20,015 km
        pole_to_pole = haversine_distance(90.0, 0.0, -90.0, 0.0)
        assert pole_to_pole == pytest.approx(math.pi * EARTH_RADIUS_KM, rel=1e-4)

        # Antimeridian: 179°E to 179°W across 180° = 2 degrees along equator
        d_antimeridian = haversine_distance(0.0, 179.0, 0.0, -179.0)
        expected = (2.0 / 360.0) * (2.0 * math.pi * EARTH_RADIUS_KM)
        assert d_antimeridian == pytest.approx(expected, rel=1e-4)

    def test_haversine_vectorized(self) -> None:
        """Vectorized haversine returns identical results to scalar loop."""
        ref_lat, ref_lon = 37.7749, -122.4194
        lats = np.array([37.7750, 37.7800, 37.8000, 34.0522, -33.8688])
        lons = np.array([-122.4190, -122.4100, -122.4000, -118.2437, 151.2093])

        vec_dists = haversine_distance_vectorized(ref_lat, ref_lon, lats, lons)
        assert len(vec_dists) == len(lats)

        for i in range(len(lats)):
            scalar_d = haversine_distance(ref_lat, ref_lon, float(lats[i]), float(lons[i]))
            assert vec_dists[i] == pytest.approx(scalar_d, rel=1e-9)

    def test_bounding_box_and_point_in_bbox(self) -> None:
        """Test bounding box expansion and containment checks."""
        lat, lon = 37.7749, -122.4194
        bbox = bounding_box(lat, lon, radius_km=10.0)
        assert len(bbox) == 4
        min_lat, min_lon, max_lat, max_lon = bbox

        assert min_lat < lat < max_lat
        assert min_lon < lon < max_lon

        # Point itself is inside
        assert point_in_bbox(lat, lon, bbox) is True

        # Point far away is outside
        assert point_in_bbox(lat + 2.0, lon, bbox) is False
        assert point_in_bbox(lat, lon + 2.0, bbox) is False


class TestKDTreeIndex:
    """Test KDTree spatial index functionality and edge cases."""

    def test_empty_kdtree(self) -> None:
        """Empty KDTree handles queries gracefully."""
        tree = KDTreeIndex()
        assert tree.size() == 0
        assert len(tree) == 0

        target = Coordinates(lat=37.7749, lon=-122.4194)
        assert tree.query_radius(target, radius_km=5.0) == []
        assert tree.query_knn(target, k=3) == []
        assert tree.query_bbox((37.0, -123.0, 38.0, -122.0)) == []

    def test_single_property_kdtree(self, sample_property: Property) -> None:
        """KDTree with one property answers queries correctly."""
        tree = KDTreeIndex([sample_property])
        assert tree.size() == 1

        target = sample_property.coordinates
        # Radius contains the property
        res_r = tree.query_radius(target, radius_km=1.0)
        assert len(res_r) == 1
        assert res_r[0].id == sample_property.id

        # Radius 0 at exact coordinate includes property (dist=0.0 <= 0.0)
        res_exact = tree.query_radius(target, radius_km=0.0)
        assert len(res_exact) == 1

        # Negative radius returns empty
        assert tree.query_radius(target, radius_km=-1.0) == []

        # kNN with k=1 and k=5
        knn_1 = tree.query_knn(target, k=1)
        assert len(knn_1) == 1
        assert knn_1[0][0].id == sample_property.id
        assert knn_1[0][1] == pytest.approx(0.0, abs=1e-5)

        knn_5 = tree.query_knn(target, k=5)
        assert len(knn_5) == 1

        # k <= 0
        assert tree.query_knn(target, k=0) == []
        assert tree.query_knn(target, k=-2) == []

    def test_kdtree_knn_accuracy(self, sample_properties: List[Property]) -> None:
        """KDTree kNN results match brute force haversine distances in order."""
        tree = KDTreeIndex(sample_properties)
        assert tree.size() == len(sample_properties)

        target = Coordinates(lat=37.7750, lon=-122.4190)

        # Brute-force calculate all distances
        all_dists = [
            (p, haversine_distance(target.lat, target.lon, p.coordinates.lat, p.coordinates.lon))
            for p in sample_properties
        ]
        all_dists.sort(key=lambda x: x[1])

        # Test k=1, k=3, k=len
        for k in (1, 3, len(sample_properties)):
            knn_res = tree.query_knn(target, k=k)
            assert len(knn_res) == k
            for i in range(k):
                assert knn_res[i][0].id == all_dists[i][0].id
                assert knn_res[i][1] == pytest.approx(all_dists[i][1], rel=1e-5)

    def test_kdtree_radius_query(self, sample_properties: List[Property]) -> None:
        """KDTree query_radius returns exactly properties within radius."""
        tree = KDTreeIndex(sample_properties)
        target = Coordinates(lat=37.7749, lon=-122.4194)

        radius_km = 0.5
        res = tree.query_radius(target, radius_km=radius_km)

        # Verify all returned properties are <= radius_km
        for p in res:
            d = haversine_distance(target.lat, target.lon, p.coordinates.lat, p.coordinates.lon)
            assert d <= radius_km

        # Verify no properties <= radius_km were missed
        returned_ids = {p.id for p in res}
        for p in sample_properties:
            d = haversine_distance(target.lat, target.lon, p.coordinates.lat, p.coordinates.lon)
            if d <= radius_km:
                assert p.id in returned_ids

    def test_kdtree_bbox_query(self, sample_properties: List[Property]) -> None:
        """KDTree query_bbox correctly filters properties inside bounding box."""
        tree = KDTreeIndex(sample_properties)

        # Box covering Market St properties
        bbox = BoundingBox(min_lat=37.7740, min_lon=-122.4200, max_lat=37.7765, max_lon=-122.4160)
        props_in_box = tree.query_bbox(bbox)
        assert len(props_in_box) >= 2

        for p in props_in_box:
            assert bbox.contains(p.coordinates) is True

        # Inverted box returns empty
        assert tree.query_bbox((38.0, -122.0, 37.0, -123.0)) == []

    def test_kdtree_duplicate_coordinates(self) -> None:
        """KDTree handles multiple properties at identical coordinates."""
        props = [
            Property(
                id=f"PROP-DUP-{i}",
                address=f"Unit {i}",
                coordinates=Coordinates(lat=37.7749, lon=-122.4194),
                price=500000.0 + i * 10000,
                sqft=1000.0 + i * 100,
            )
            for i in range(5)
        ]
        tree = KDTreeIndex(props)
        assert tree.size() == 5

        knn = tree.query_knn(Coordinates(lat=37.7749, lon=-122.4194), k=3)
        assert len(knn) == 3
        for _, dist in knn:
            assert dist == pytest.approx(0.0, abs=1e-5)


class TestGridIndex:
    """Test Spatial Grid Index functionality and edge cases."""

    def test_empty_grid_index(self) -> None:
        """Empty GridIndex handles queries cleanly."""
        grid = GridIndex()
        assert grid.size() == 0
        assert len(grid) == 0

        target = Coordinates(lat=37.7749, lon=-122.4194)
        assert grid.query_radius(target, radius_km=5.0) == []
        assert grid.query_knn(target, k=3) == []
        assert grid.query_bbox((37.0, -123.0, 38.0, -122.0)) == []

    def test_grid_radius_and_knn(self, sample_properties: List[Property]) -> None:
        """GridIndex accurately resolves radius and kNN queries."""
        grid = GridIndex(sample_properties, cell_size_deg=0.02)
        assert grid.size() == len(sample_properties)

        target = Coordinates(lat=37.7750, lon=-122.4190)

        # Radius query
        rad_res = grid.query_radius(target, radius_km=1.0)
        assert len(rad_res) >= 2
        for p in rad_res:
            d = haversine_distance(target.lat, target.lon, p.coordinates.lat, p.coordinates.lon)
            assert d <= 1.0

        # Negative radius returns empty
        assert grid.query_radius(target, radius_km=-0.5) == []

        # kNN query
        knn_res = grid.query_knn(target, k=3)
        assert len(knn_res) == 3
        assert knn_res[0][1] <= knn_res[1][1] <= knn_res[2][1]

        # k larger than index size
        knn_large = grid.query_knn(target, k=50)
        assert len(knn_large) == len(sample_properties)

        # k <= 0
        assert grid.query_knn(target, k=0) == []
        assert grid.query_knn(target, k=-1) == []

    def test_grid_bbox_query(self, sample_properties: List[Property]) -> None:
        """GridIndex query_bbox correctly filters by bounding box."""
        grid = GridIndex(sample_properties, cell_size_deg=0.01)
        bbox = BoundingBox(min_lat=37.7740, min_lon=-122.4200, max_lat=37.7765, max_lon=-122.4160)

        results = grid.query_bbox(bbox)
        assert len(results) >= 2
        for p in results:
            assert bbox.contains(p.coordinates) is True

        # Inverted bbox returns empty
        assert grid.query_bbox((38.0, -122.0, 37.0, -123.0)) == []


class TestSpatialFeatureExtractor:
    """Test spatial lag, density, and landmark feature extraction."""

    def test_spatial_lag_basic_and_decay(self, sample_properties: List[Property]) -> None:
        """Test spatial lag weighting and decay power variations."""
        extractor = SpatialFeatureExtractor()
        target = Coordinates(lat=37.7749, lon=-122.4194)

        lag_1 = extractor.spatial_lag(target, sample_properties, value_attr="price", k=3, decay_power=1.0)
        lag_2 = extractor.spatial_lag(target, sample_properties, value_attr="price", k=3, decay_power=2.0)

        assert lag_1 > 0.0
        assert lag_2 > 0.0

    def test_spatial_lag_exclude_id_prevents_leakage(self, sample_properties: List[Property]) -> None:
        """exclude_id must prevent a property from including its own price in spatial lag."""
        extractor = SpatialFeatureExtractor()
        subject = sample_properties[0]

        lag_with_self = extractor.spatial_lag(
            subject.coordinates, sample_properties, value_attr="price", k=5, exclude_id=None
        )
        lag_without_self = extractor.spatial_lag(
            subject.coordinates, sample_properties, value_attr="price", k=5, exclude_id=subject.id
        )

        # Self distance is ~0 so weight on self is dominant when not excluded
        assert lag_with_self != lag_without_self
        # With self excluded, lag is strictly derived from neighboring properties
        assert lag_without_self > 0.0

    def test_spatial_lag_empty_or_missing_attribute(self) -> None:
        """Spatial lag handles empty property list and missing attributes safely."""
        extractor = SpatialFeatureExtractor()
        target = Coordinates(lat=37.7749, lon=-122.4194)

        assert extractor.spatial_lag(target, [], value_attr="price") == 0.0

        prop_no_price = Property(
            id="P-NOPRICE",
            address="Nowhere",
            coordinates=target,
            sqft=1200.0,
            price=None,
        )
        assert extractor.spatial_lag(target, [prop_no_price], value_attr="price") == 0.0
        assert extractor.spatial_lag(target, [prop_no_price], value_attr="non_existent_field") == 0.0

    def test_local_density_and_exclude_id(self, sample_properties: List[Property]) -> None:
        """Local density counts properties within radius and respects exclude_id."""
        extractor = SpatialFeatureExtractor()
        subject = sample_properties[0]

        density_total = extractor.local_density(subject.coordinates, sample_properties, radius_km=1.0)
        density_excluded = extractor.local_density(
            subject.coordinates, sample_properties, radius_km=1.0, exclude_id=subject.id
        )

        assert density_total == density_excluded + 1

    def test_landmarks_and_extract_all(self, sample_properties: List[Property]) -> None:
        """Test adding landmarks and extracting complete feature dictionary."""
        extractor = SpatialFeatureExtractor(
            landmarks={"downtown": Coordinates(lat=37.7749, lon=-122.4194)}
        )
        extractor.add_landmark("airport", Coordinates(lat=37.6213, lon=-122.3790))

        subject = sample_properties[1]
        features = extractor.extract_all(subject, sample_properties)

        assert features["lat"] == subject.coordinates.lat
        assert features["lon"] == subject.coordinates.lon
        assert "spatial_lag_price_k5" in features
        assert "spatial_lag_price_k10" in features
        assert "density_1km" in features
        assert "density_3km" in features
        assert "density_5km" in features
        assert "dist_downtown_km" in features
        assert "dist_airport_km" in features
        assert features["dist_downtown_km"] >= 0.0
        assert features["dist_airport_km"] > 0.0
