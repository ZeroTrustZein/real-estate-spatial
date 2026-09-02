"""Tests for PropertyDataLoader (CSV, JSON, GeoJSON) and SyntheticRealEstateGenerator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader
from real_estate_spatial.types import Property


class TestPropertyDataLoader:
    """Test suite for PropertyDataLoader across JSON, CSV, and GeoJSON formats."""

    def test_json_roundtrip_and_structure(
        self, tmp_path: Path, sample_properties: List[Property]
    ) -> None:
        """Verify saving and loading JSON produces identical properties."""
        filepath = tmp_path / "subdir" / "properties.json"

        # Save properties to nested path (testing mkdir)
        PropertyDataLoader.save_json(sample_properties, filepath)
        assert filepath.exists()

        # Load back
        loaded = PropertyDataLoader.load_json(filepath)
        assert len(loaded) == len(sample_properties)
        for original, restored in zip(sample_properties, loaded):
            assert original.id == restored.id
            assert original.address == restored.address
            assert original.coordinates.lat == pytest.approx(restored.coordinates.lat)
            assert original.coordinates.lon == pytest.approx(restored.coordinates.lon)
            assert original.price == pytest.approx(restored.price)
            assert original.sqft == pytest.approx(restored.sqft)
            assert original.bedrooms == restored.bedrooms
            assert original.bathrooms == pytest.approx(restored.bathrooms)

    def test_json_load_wrapped_object(self, tmp_path: Path, sample_property: Property) -> None:
        """Verify load_json supports dict format with 'properties' root key."""
        filepath = tmp_path / "wrapped.json"
        data = {
            "metadata": {"version": "1.0", "city": "San Francisco"},
            "properties": [sample_property.to_dict()],
        }
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = PropertyDataLoader.load_json(filepath)
        assert len(loaded) == 1
        assert loaded[0].id == sample_property.id

    def test_json_load_invalid_format_raises(self, tmp_path: Path) -> None:
        """Verify ValueError is raised when JSON root is not a list or dictionary with properties."""
        filepath = tmp_path / "invalid.json"
        with filepath.open("w", encoding="utf-8") as f:
            json.dump({"data": 123}, f)

        with pytest.raises(ValueError, match="Expected JSON file containing a list of properties"):
            PropertyDataLoader.load_json(filepath)

    def test_csv_roundtrip_with_optional_fields(
        self, tmp_path: Path, sample_properties: List[Property]
    ) -> None:
        """Verify CSV saving and loading handles standard and optional property attributes."""
        filepath = tmp_path / "data" / "properties.csv"
        PropertyDataLoader.save_csv(sample_properties, filepath)
        assert filepath.exists()

        loaded = PropertyDataLoader.load_csv(filepath)
        assert len(loaded) == len(sample_properties)
        for orig, res in zip(sample_properties, loaded):
            assert orig.id == res.id
            assert orig.address == res.address
            assert orig.coordinates.lat == pytest.approx(res.coordinates.lat)
            assert orig.coordinates.lon == pytest.approx(res.coordinates.lon)
            assert orig.price == pytest.approx(res.price)
            assert orig.sqft == pytest.approx(res.sqft)
            assert orig.bedrooms == res.bedrooms
            assert orig.bathrooms == pytest.approx(res.bathrooms)
            assert orig.year_built == res.year_built
            assert orig.sale_date == res.sale_date

    def test_csv_load_latitude_longitude_headers(self, tmp_path: Path) -> None:
        """Verify load_csv supports alternative column names like 'latitude' and 'longitude'."""
        filepath = tmp_path / "alt_headers.csv"
        csv_content = (
            "id,address,latitude,longitude,price,sqft,bedrooms,bathrooms\n"
            "ALT-01,123 Main St,37.78,-122.40,950000,1500,2,2.0\n"
            "ALT-02,456 Oak St,37.79,-122.41,,1200,1,1.0\n"
        )
        filepath.write_text(csv_content, encoding="utf-8")

        loaded = PropertyDataLoader.load_csv(filepath)
        assert len(loaded) == 2
        assert loaded[0].id == "ALT-01"
        assert loaded[0].coordinates.lat == pytest.approx(37.78)
        assert loaded[0].coordinates.lon == pytest.approx(-122.40)
        assert loaded[0].price == pytest.approx(950000.0)

        # Property without price
        assert loaded[1].id == "ALT-02"
        assert loaded[1].price is None

    def test_csv_load_whitespace_and_blank_values(self, tmp_path: Path) -> None:
        """Verify load_csv parses blank and whitespace values as None instead of raising ValueError."""
        filepath = tmp_path / "whitespace.csv"
        csv_content = (
            "id,address,lat,lon,price,sqft,bedrooms,bathrooms,lot_size_sqft,year_built,property_type,sale_date\n"
            "PROP-WS1, 100 Main St , 37.77 , -122.41 ,  , 1500 , 3 , 2.0 ,   ,   , single_family ,   \n"
            "PROP-WS2, 200 Oak St , 37.78 , -122.42 , 500000 , 1200 , 2 , 1.0 , 4000 , 1990 , condo , 2023-01-01 \n"
        )
        filepath.write_text(csv_content, encoding="utf-8")

        loaded = PropertyDataLoader.load_csv(filepath)
        assert len(loaded) == 2
        p1 = loaded[0]
        assert p1.id == "PROP-WS1"
        assert p1.address == "100 Main St"
        assert p1.coordinates.lat == pytest.approx(37.77)
        assert p1.coordinates.lon == pytest.approx(-122.41)
        assert p1.price is None
        assert p1.lot_size_sqft is None
        assert p1.year_built is None
        assert p1.sale_date is None

    def test_geojson_roundtrip(self, tmp_path: Path, sample_properties: List[Property]) -> None:
        """Verify GeoJSON FeatureCollection serialization and loading."""
        filepath = tmp_path / "geo" / "properties.geojson"
        PropertyDataLoader.save_geojson(sample_properties, filepath)
        assert filepath.exists()

        # Check raw json structure
        with filepath.open("r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        assert geojson_data["type"] == "FeatureCollection"
        assert len(geojson_data["features"]) == len(sample_properties)
        first_feat = geojson_data["features"][0]
        assert first_feat["type"] == "Feature"
        assert first_feat["geometry"]["type"] == "Point"
        # In GeoJSON geometry.coordinates is [lon, lat]
        assert first_feat["geometry"]["coordinates"] == [
            sample_properties[0].coordinates.lon,
            sample_properties[0].coordinates.lat,
        ]

        # Load back
        loaded = PropertyDataLoader.load_geojson(filepath)
        assert len(loaded) == len(sample_properties)
        for orig, res in zip(sample_properties, loaded):
            assert orig.id == res.id
            assert orig.coordinates.lat == pytest.approx(res.coordinates.lat)
            assert orig.coordinates.lon == pytest.approx(res.coordinates.lon)
            assert orig.sqft == pytest.approx(res.sqft)


class TestSyntheticRealEstateGenerator:
    """Test suite for SyntheticRealEstateGenerator."""

    def test_reproducibility_with_seed(self) -> None:
        """Identical random seeds must generate exact identical property lists."""
        gen1 = SyntheticRealEstateGenerator(seed=42)
        props1 = gen1.generate(n_samples=20)

        gen2 = SyntheticRealEstateGenerator(seed=42)
        props2 = gen2.generate(n_samples=20)

        assert len(props1) == 20
        assert len(props2) == 20
        for p1, p2 in zip(props1, props2):
            assert p1.id == p2.id
            assert p1.address == p2.address
            assert p1.price == p2.price
            assert p1.sqft == p2.sqft
            assert p1.coordinates.lat == p2.coordinates.lat
            assert p1.coordinates.lon == p2.coordinates.lon

    def test_different_seeds_produce_distinct_data(self) -> None:
        """Different seeds must produce different data points."""
        gen = SyntheticRealEstateGenerator()
        p_a = gen.generate(n_samples=10, seed=101)
        p_b = gen.generate(n_samples=10, seed=202)

        assert p_a[0].price != p_b[0].price or p_a[0].coordinates.lat != p_b[0].coordinates.lat

    def test_property_attributes_validity(self) -> None:
        """Verify generated properties satisfy all physical and financial constraints."""
        gen = SyntheticRealEstateGenerator(
            center_lat=40.7128,
            center_lon=-74.0060,
            base_price_sqft=800.0,
            cbd_gradient=-30.0,
            seed=777,
        )
        props = gen.generate(n_samples=50)
        assert len(props) == 50

        for p in props:
            # Coordinates
            assert -90.0 <= p.coordinates.lat <= 90.0
            assert -180.0 <= p.coordinates.lon <= 180.0

            # Physical specs
            assert 1 <= p.bedrooms <= 6
            assert 1.0 <= p.bathrooms <= 6.0
            assert p.sqft >= 500.0
            assert p.lot_size_sqft is not None and p.lot_size_sqft >= p.sqft
            assert p.year_built is not None and 1900 <= p.year_built <= 2026
            assert p.property_type in ("single_family", "condo", "townhouse")

            # Financial specs
            assert p.price is not None and p.price >= 75000.0
            assert p.features is not None
            assert "dist_cbd_km" in p.features
            assert "cluster_premium" in p.features
            assert p.features["dist_cbd_km"] >= 0.0

    def test_generate_dataset_alias(self) -> None:
        """Verify generate_dataset acts as alias with correct n_properties."""
        gen = SyntheticRealEstateGenerator(seed=99)
        props = gen.generate_dataset(n_properties=15, noise_level=0.02)
        assert len(props) == 15
