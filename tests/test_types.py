"""Comprehensive tests for real_estate_spatial.types module."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from real_estate_spatial.types import (
    BoundingBox,
    CompCandidate,
    Coordinates,
    ModelMetrics,
    Property,
    PropertyType,
    ValuationMethodology,
    ValuationResult,
)


class TestCoordinates:
    """Test suite for Coordinates dataclass."""

    def test_valid_coordinates_and_bounds(self) -> None:
        """Test valid coordinates including exact boundary values."""
        c1 = Coordinates(lat=37.7749, lon=-122.4194)
        assert c1.lat == 37.7749
        assert c1.lon == -122.4194

        # Boundary extremes
        c_north = Coordinates(lat=90.0, lon=0.0)
        assert c_north.lat == 90.0

        c_south = Coordinates(lat=-90.0, lon=0.0)
        assert c_south.lat == -90.0

        c_east = Coordinates(lat=0.0, lon=180.0)
        assert c_east.lon == 180.0

        c_west = Coordinates(lat=0.0, lon=-180.0)
        assert c_west.lon == -180.0

        # Int conversion to float
        c_int = Coordinates(lat=10, lon=20)
        assert isinstance(c_int.lat, float)
        assert isinstance(c_int.lon, float)
        assert c_int.lat == 10.0
        assert c_int.lon == 20.0

    def test_invalid_bounds(self) -> None:
        """Test out-of-bounds latitude and longitude raise ValueError."""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            Coordinates(lat=90.001, lon=0.0)

        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            Coordinates(lat=-90.001, lon=0.0)

        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            Coordinates(lat=0.0, lon=180.001)

        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            Coordinates(lat=0.0, lon=-180.001)

    def test_nan_and_inf_rejection(self) -> None:
        """Test that NaN and Inf values are rejected with ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            Coordinates(lat=float("nan"), lon=0.0)

        with pytest.raises(ValueError, match="NaN"):
            Coordinates(lat=0.0, lon=float("nan"))

        with pytest.raises(ValueError, match="Infinite"):
            Coordinates(lat=float("inf"), lon=0.0)

        with pytest.raises(ValueError, match="Infinite"):
            Coordinates(lat=0.0, lon=float("-inf"))

    def test_type_errors(self) -> None:
        """Test non-numeric types raise TypeError."""
        with pytest.raises(TypeError):
            Coordinates(lat="37.77", lon=-122.41)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            Coordinates(lat=True, lon=10.0)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            Coordinates(lat=10.0, lon=False)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            Coordinates(lat=None, lon=0.0)  # type: ignore[arg-type]

    def test_distance_to(self) -> None:
        """Test distance_to calculation across various scenarios."""
        sf = Coordinates(lat=37.7749, lon=-122.4194)
        oakland = Coordinates(lat=37.8044, lon=-122.2712)

        # Distance to self should be zero
        assert sf.distance_to(sf) == pytest.approx(0.0, abs=1e-6)

        # SF to Oakland is ~13.4 km
        dist = sf.distance_to(oakland)
        assert 12.0 < dist < 15.0

        # Custom radius (e.g. Earth radius in miles: 3958.8)
        dist_miles = sf.distance_to(oakland, radius_km=3958.8)
        assert 7.5 < dist_miles < 9.5

        # Antipodal points (distance should be pi * R)
        p1 = Coordinates(lat=0.0, lon=0.0)
        p2 = Coordinates(lat=0.0, lon=180.0)
        assert p1.distance_to(p2, radius_km=1000.0) == pytest.approx(math.pi * 1000.0, rel=1e-5)

        # Invalid type passed to distance_to
        with pytest.raises(TypeError):
            sf.distance_to((37.8044, -122.2712))  # type: ignore[arg-type]

    def test_to_tuple_and_from_tuple(self) -> None:
        """Test tuple conversion roundtrips."""
        c = Coordinates(lat=40.7128, lon=-74.0060)
        tup = c.to_tuple()
        assert tup == (40.7128, -74.0060)

        c_from = Coordinates.from_tuple((40.7128, -74.0060))
        assert c_from == c

        c_from_list = Coordinates.from_tuple([40.7128, -74.0060])
        assert c_from_list == c

        with pytest.raises(ValueError, match="Expected sequence of 2 items"):
            Coordinates.from_tuple((40.7128,))

        with pytest.raises(ValueError, match="Expected sequence of 2 items"):
            Coordinates.from_tuple((40.7128, -74.0060, 100.0))

    def test_to_dict_and_from_dict(self) -> None:
        """Test dictionary serialization and robust deserialization."""
        c = Coordinates(lat=51.5074, lon=-0.1278)
        d = c.to_dict()
        assert d == {"lat": 51.5074, "lon": -0.1278}

        assert Coordinates.from_dict(d) == c
        assert Coordinates.from_dict({"latitude": 51.5074, "longitude": -0.1278}) == c
        assert Coordinates.from_dict({"latitude": 51.5074, "lng": -0.1278}) == c

        with pytest.raises(ValueError, match="Missing lat or lon"):
            Coordinates.from_dict({"foo": 12.0})

    def test_immutability_and_hashability(self) -> None:
        """Test frozen dataclass properties (immutability and hashing)."""
        c = Coordinates(lat=37.7749, lon=-122.4194)

        with pytest.raises(FrozenInstanceError):
            c.lat = 40.0  # type: ignore[misc]

        coord_set = {c, Coordinates(lat=37.7749, lon=-122.4194)}
        assert len(coord_set) == 1


class TestBoundingBox:
    """Test suite for BoundingBox dataclass."""

    def test_valid_bounding_box(self) -> None:
        """Test creation of valid BoundingBox."""
        bbox = BoundingBox(min_lat=37.0, min_lon=-123.0, max_lat=38.0, max_lon=-122.0)
        assert bbox.min_lat == 37.0
        assert bbox.min_lon == -123.0
        assert bbox.max_lat == 38.0
        assert bbox.max_lon == -122.0

    def test_invalid_bounds_and_order(self) -> None:
        """Test invalid coordinates and min > max bounds."""
        with pytest.raises(ValueError, match="min_lat .* cannot be greater than max_lat"):
            BoundingBox(min_lat=38.0, min_lon=-123.0, max_lat=37.0, max_lon=-122.0)

        with pytest.raises(ValueError, match="min_lon .* cannot be greater than max_lon"):
            BoundingBox(min_lat=37.0, min_lon=-121.0, max_lat=38.0, max_lon=-122.0)

        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            BoundingBox(min_lat=-95.0, min_lon=0.0, max_lat=0.0, max_lon=10.0)

        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            BoundingBox(min_lat=0.0, min_lon=-190.0, max_lat=10.0, max_lon=10.0)

        with pytest.raises(ValueError, match="NaN or Infinite"):
            BoundingBox(min_lat=float("nan"), min_lon=0.0, max_lat=10.0, max_lon=10.0)

        with pytest.raises(TypeError):
            BoundingBox(min_lat="0.0", min_lon=0.0, max_lat=10.0, max_lon=10.0)  # type: ignore[arg-type]

    def test_contains(self) -> None:
        """Test containment check with Coordinates and sequences."""
        bbox = BoundingBox(min_lat=37.0, min_lon=-123.0, max_lat=38.0, max_lon=-122.0)

        # Inside
        assert bbox.contains(Coordinates(lat=37.5, lon=-122.5)) is True
        assert bbox.contains((37.5, -122.5)) is True
        assert bbox.contains([37.5, -122.5]) is True

        # On borders
        assert bbox.contains(Coordinates(lat=37.0, lon=-123.0)) is True
        assert bbox.contains(Coordinates(lat=38.0, lon=-122.0)) is True

        # Outside
        assert bbox.contains(Coordinates(lat=36.9, lon=-122.5)) is False
        assert bbox.contains(Coordinates(lat=38.1, lon=-122.5)) is False
        assert bbox.contains(Coordinates(lat=37.5, lon=-123.1)) is False
        assert bbox.contains(Coordinates(lat=37.5, lon=-121.9)) is False

        # Invalid point
        with pytest.raises(TypeError):
            bbox.contains("invalid")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            bbox.contains("ab")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            bbox.contains("12")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            bbox.contains(b"12")  # type: ignore[arg-type]

    def test_to_tuple_from_tuple_to_dict_from_dict(self) -> None:
        """Test tuple and dictionary conversions for BoundingBox."""
        bbox = BoundingBox(min_lat=37.0, min_lon=-123.0, max_lat=38.0, max_lon=-122.0)

        tup = bbox.to_tuple()
        assert tup == (37.0, -123.0, 38.0, -122.0)
        assert BoundingBox.from_tuple(tup) == bbox

        with pytest.raises(ValueError, match="Expected sequence of 4 items"):
            BoundingBox.from_tuple((37.0, -123.0, 38.0))

        d = bbox.to_dict()
        assert d == {
            "min_lat": 37.0,
            "min_lon": -123.0,
            "max_lat": 38.0,
            "max_lon": -122.0,
        }
        assert BoundingBox.from_dict(d) == bbox


class TestEnums:
    """Test suite for PropertyType and ValuationMethodology enums."""

    def test_property_type_values(self) -> None:
        """Verify PropertyType members and string comparison."""
        assert PropertyType.SINGLE_FAMILY == "single_family"
        assert PropertyType.CONDO == "condo"
        assert PropertyType.TOWNHOUSE == "townhouse"
        assert PropertyType.MULTI_FAMILY == "multi_family"
        assert PropertyType.COMMERCIAL == "commercial"
        assert PropertyType.LAND == "land"

    def test_valuation_methodology_values(self) -> None:
        """Verify ValuationMethodology members and string comparison."""
        assert ValuationMethodology.COMPS == "comps"
        assert ValuationMethodology.HEDONIC == "hedonic"
        assert ValuationMethodology.GBM == "gbm"
        assert ValuationMethodology.ENSEMBLE == "ensemble"
        assert ValuationMethodology.FALLBACK == "fallback"


class TestProperty:
    """Test suite for Property dataclass."""

    def test_property_creation_and_price_per_sqft(self) -> None:
        """Test Property instantiation and price_per_sqft calculation."""
        p = Property(
            id="PROP-01",
            address="123 Main St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            price=1_000_000.0,
            sqft=2000.0,
            bedrooms=3,
            bathrooms=2.5,
            lot_size_sqft=5000.0,
            year_built=2000,
            property_type=PropertyType.SINGLE_FAMILY,
        )

        assert p.id == "PROP-01"
        assert p.property_type == "single_family"
        assert p.price_per_sqft == 500.0
        assert p.age(2026) == 26.0
        assert p.effective_lot_size() == 5000.0
        assert p.lot_sqft == 5000.0

        # Sqft = 0 or price = None -> price_per_sqft is None
        p_no_sqft = Property(
            id="PROP-02",
            address="456 Elm St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            price=500_000.0,
            sqft=0.0,
        )
        assert p_no_sqft.price_per_sqft is None
        assert p_no_sqft.age() == 25.0
        assert p_no_sqft.effective_lot_size() == 0.0

        p_no_price = Property(
            id="PROP-03",
            address="789 Oak St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            price=None,
            sqft=1500.0,
        )
        assert p_no_price.price_per_sqft is None
        assert p_no_price.effective_lot_size() == 4500.0

    def test_negative_validations(self) -> None:
        """Test negative structural or price values raise ValueError."""
        coords = Coordinates(lat=37.7749, lon=-122.4194)

        with pytest.raises(ValueError, match="sqft cannot be negative"):
            Property(id="P1", address="A", coordinates=coords, sqft=-10.0)

        with pytest.raises(ValueError, match="bedrooms cannot be negative"):
            Property(id="P1", address="A", coordinates=coords, bedrooms=-1)

        with pytest.raises(ValueError, match="bathrooms cannot be negative"):
            Property(id="P1", address="A", coordinates=coords, bathrooms=-0.5)

        with pytest.raises(ValueError, match="price cannot be negative"):
            Property(id="P1", address="A", coordinates=coords, price=-100.0)

        with pytest.raises(ValueError, match="lot_size_sqft cannot be negative"):
            Property(id="P1", address="A", coordinates=coords, lot_size_sqft=-50.0)

    def test_property_from_dict_variants(self) -> None:
        """Test parsing Property from various dictionary structures."""
        # Nested coordinates dict
        p1 = Property.from_dict(
            {
                "id": "P1",
                "address": "123 Main St",
                "coordinates": {"lat": 37.7749, "lon": -122.4194},
                "price": "750000",
                "sqft": "1500",
                "bedrooms": "3",
                "bathrooms": "2.0",
                "lot_size_sqft": "4000",
                "year_built": "2005",
                "sale_date": "2025-01-01",
            }
        )
        assert p1.coordinates.lat == 37.7749
        assert p1.price == 750000.0
        assert p1.sqft == 1500.0
        assert p1.bedrooms == 3
        assert p1.lot_size_sqft == 4000.0
        assert p1.year_built == 2005
        assert p1.sale_date == "2025-01-01"

        # Coordinates already Coordinates instance
        p_coord_inst = Property.from_dict(
            {
                "id": "P1B",
                "address": "123 Main St",
                "coordinates": Coordinates(lat=37.7749, lon=-122.4194),
            }
        )
        assert p_coord_inst.coordinates == Coordinates(lat=37.7749, lon=-122.4194)

        # Flat coordinates (lat, lon)
        p2 = Property.from_dict(
            {
                "id": "P2",
                "address": "456 Market St",
                "lat": 37.7749,
                "lon": -122.4194,
            }
        )
        assert p2.coordinates == Coordinates(lat=37.7749, lon=-122.4194)

        # Flat coordinates (latitude, longitude)
        p3 = Property.from_dict(
            {
                "id": "P3",
                "address": "789 Mission St",
                "latitude": 37.7749,
                "longitude": -122.4194,
            }
        )
        assert p3.coordinates == Coordinates(lat=37.7749, lon=-122.4194)

        # Coordinate tuple / sequence
        p4 = Property.from_dict(
            {
                "id": "P4",
                "address": "101 Howard St",
                "coordinates": [37.7749, -122.4194],
            }
        )
        assert p4.coordinates == Coordinates(lat=37.7749, lon=-122.4194)

        # Fallback when coordinates missing and lat/lon missing
        p_fallback = Property.from_dict({"id": "P5", "address": "Empty St"})
        assert p_fallback.coordinates == Coordinates(lat=0.0, lon=0.0)

    def test_property_init_coordinate_types_and_age_fallback(self) -> None:
        """Test Property __post_init__ coordinate conversions/errors and age fallback."""
        # List coordinates
        p_list = Property(id="P_L", address="A", coordinates=[37.77, -122.41])  # type: ignore[arg-type]
        assert isinstance(p_list.coordinates, Coordinates)
        assert p_list.coordinates.lat == 37.77

        # Tuple coordinates
        p_tup = Property(id="P_T", address="A", coordinates=(37.77, -122.41))  # type: ignore[arg-type]
        assert isinstance(p_tup.coordinates, Coordinates)

        # Dict coordinates
        p_dict = Property(id="P_D", address="A", coordinates={"lat": 37.77, "lon": -122.41})  # type: ignore[arg-type]
        assert isinstance(p_dict.coordinates, Coordinates)

        # Invalid coordinate type raises TypeError
        with pytest.raises(TypeError, match="Invalid coordinates type"):
            Property(id="P_ERR", address="A", coordinates="invalid_coord")  # type: ignore[arg-type]

        # Property age invalid/non-numeric year_built fallback to 25.0
        p_invalid_year = Property(
            id="P_Y",
            address="A",
            coordinates=Coordinates(lat=0.0, lon=0.0),
            year_built="invalid_year",  # type: ignore[arg-type]
        )
        assert p_invalid_year.age(2026) == 25.0

    def test_to_geojson_feature(self) -> None:
        """Test GeoJSON Feature conversion."""
        p = Property(
            id="PROP-GEO-1",
            address="1 Market St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            price=1_200_000.0,
            sqft=1800.0,
            bedrooms=2,
            bathrooms=2.0,
        )
        feat = p.to_geojson_feature()

        assert feat["type"] == "Feature"
        assert feat["id"] == "PROP-GEO-1"
        assert feat["geometry"]["type"] == "Point"
        # GeoJSON is [lon, lat]
        assert feat["geometry"]["coordinates"] == [-122.4194, 37.7749]
        assert feat["properties"]["id"] == "PROP-GEO-1"
        assert feat["properties"]["price"] == 1_200_000.0
        assert "coordinates" not in feat["properties"]


class TestCompCandidate:
    """Test suite for CompCandidate dataclass."""

    def test_adjustments_and_properties(self) -> None:
        """Test net, gross, and percentage adjustment calculations."""
        base_prop = Property(
            id="COMP-01",
            address="123 Comps Ave",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            price=1_000_000.0,
            sqft=1800.0,
        )
        candidate = CompCandidate(
            property=base_prop,
            similarity_score=0.92,
            distance_km=0.85,
            time_diff_days=15,
            adjusted_price=1_030_000.0,
            adjustments={"sqft_adj": 50_000.0, "time_adj": -20_000.0},
        )

        assert candidate.net_adjustment == 30_000.0
        assert candidate.gross_adjustment == 70_000.0
        assert candidate.net_adjustment_pct == pytest.approx(0.03, rel=1e-5)

        # Dictionary roundtrip
        d = candidate.to_dict()
        assert d["similarity_score"] == 0.92
        assert d["net_adjustment"] == 30_000.0
        assert d["gross_adjustment"] == 70_000.0
        assert d["net_adjustment_pct"] == 0.03

        reconstructed = CompCandidate.from_dict(d)
        assert reconstructed.property.id == base_prop.id
        assert reconstructed.similarity_score == candidate.similarity_score
        assert reconstructed.adjustments == candidate.adjustments

    def test_net_adjustment_pct_zero_price(self) -> None:
        """Test net_adjustment_pct returns 0.0 when property price is None or zero."""
        prop_zero = Property(
            id="COMP-ZERO",
            address="0 Free Rd",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            price=0.0,
        )
        cand = CompCandidate(
            property=prop_zero,
            similarity_score=0.8,
            distance_km=1.0,
            time_diff_days=0,
            adjusted_price=10_000.0,
            adjustments={"adj": 10_000.0},
        )
        assert cand.net_adjustment_pct == 0.0

    def test_comp_candidate_from_dict_variants_and_errors(self) -> None:
        """Test CompCandidate.from_dict with Property instances and invalid types."""
        base_prop = Property(
            id="PROP-COMP-1",
            address="123 Test St",
            coordinates=Coordinates(lat=37.77, lon=-122.41),
            price=500_000.0,
        )

        # property already a Property instance
        c1 = CompCandidate.from_dict(
            {
                "property": base_prop,
                "similarity_score": 0.85,
                "distance_km": 1.2,
                "time_diff_days": 10,
                "adjusted_price": 510_000.0,
                "adjustments": {"sqft": 10_000.0},
            }
        )
        assert c1.property.id == "PROP-COMP-1"
        assert c1.time_diff_days == 10

        # Invalid property type
        with pytest.raises(ValueError, match="CompCandidate data missing valid property"):
            CompCandidate.from_dict({"property": "not_a_property"})


class TestValuationResult:
    """Test suite for ValuationResult dataclass."""

    def test_spread_and_spread_pct(self) -> None:
        """Test spread, spread_pct, and serialization roundtrip."""
        res = ValuationResult(
            property_id="PROP-VAL-01",
            estimated_value=1_000_000.0,
            confidence_score=0.88,
            lower_bound=950_000.0,
            upper_bound=1_050_000.0,
            comps_used=["COMP-1", "COMP-2"],
            methodology=ValuationMethodology.GBM.value,
            feature_contributions={"sqft": 0.45, "distance_to_center": -0.15},
        )

        assert res.spread == 100_000.0
        assert res.spread_pct == pytest.approx(0.10, rel=1e-5)

        d = res.to_dict()
        assert d["spread"] == 100_000.0
        assert d["spread_pct"] == 0.10
        assert d["property_id"] == "PROP-VAL-01"

        reconstructed = ValuationResult.from_dict(d)
        assert reconstructed.property_id == res.property_id
        assert reconstructed.estimated_value == res.estimated_value
        assert reconstructed.spread == res.spread

    def test_spread_pct_zero_estimate(self) -> None:
        """Test spread_pct returns 0.0 when estimated_value is zero."""
        res = ValuationResult(
            property_id="P0",
            estimated_value=0.0,
            confidence_score=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            comps_used=[],
            methodology="fallback",
        )
        assert res.spread_pct == 0.0

    def test_valuation_result_comps_used_variants(self) -> None:
        """Test ValuationResult serialization and deserialization with various comp formats."""
        base_prop = Property(
            id="PROP-COMP-2",
            address="456 Test St",
            coordinates=Coordinates(lat=37.77, lon=-122.41),
            price=600_000.0,
        )
        cand = CompCandidate(
            property=base_prop,
            similarity_score=0.9,
            distance_km=0.5,
            time_diff_days=5,
            adjusted_price=605_000.0,
        )

        # Mix of CompCandidate (has to_dict), Property (has to_dict), dict, and string
        raw_dict_comp = {"raw_key": "raw_val"}
        vr = ValuationResult(
            property_id="P_VAL_1",
            estimated_value=600_000.0,
            confidence_score=0.95,
            lower_bound=580_000.0,
            upper_bound=620_000.0,
            comps_used=[cand, base_prop, raw_dict_comp, "STR_COMP_ID"],
        )

        d = vr.to_dict()
        assert len(d["comps_used"]) == 4
        assert isinstance(d["comps_used"][0], dict) and "property" in d["comps_used"][0]
        assert isinstance(d["comps_used"][1], dict) and "coordinates" in d["comps_used"][1]
        assert d["comps_used"][2] == raw_dict_comp
        assert d["comps_used"][3] == "STR_COMP_ID"

        # Deserialize back
        reconstructed = ValuationResult.from_dict(d)
        assert len(reconstructed.comps_used) == 4
        assert isinstance(reconstructed.comps_used[0], CompCandidate)
        assert isinstance(reconstructed.comps_used[1], Property)
        assert reconstructed.comps_used[2] == raw_dict_comp
        assert reconstructed.comps_used[3] == "STR_COMP_ID"


class TestModelMetrics:
    """Test suite for ModelMetrics dataclass."""

    def test_metrics_serialization_and_summary(self) -> None:
        """Test ModelMetrics serialization, deserialization, and summary formatting."""
        metrics = ModelMetrics(
            mape=4.25,
            mdape=3.10,
            rmse=25_400.50,
            mae=18_200.25,
            r2=0.9123,
            within_5_pct=75.5,
            within_10_pct=92.0,
            within_20_pct=98.5,
        )

        d = metrics.to_dict()
        assert d["mape"] == 4.25
        assert d["r2"] == 0.9123

        reconstructed = ModelMetrics.from_dict(d)
        assert reconstructed == metrics

        summary_text = metrics.summary()
        assert "Model Metrics Summary:" in summary_text
        assert "MAPE:          4.25%" in summary_text
        assert "MdAPE:         3.10%" in summary_text
        assert "MAE:           $18,200.25" in summary_text
        assert "RMSE:          $25,400.50" in summary_text
        assert "R²:            0.9123" in summary_text
        assert "Within 5%:     75.50%" in summary_text
        assert "Within 10%:    92.00%" in summary_text
        assert "Within 20%:    98.50%" in summary_text
