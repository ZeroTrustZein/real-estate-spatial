"""Pytest fixtures for real-estate-spatial tests."""

from __future__ import annotations

from typing import List

import pytest

from real_estate_spatial.types import Coordinates, Property


@pytest.fixture
def sample_coordinates() -> Coordinates:
    """Fixture returning reference coordinate in San Francisco."""
    return Coordinates(lat=37.7749, lon=-122.4194)


@pytest.fixture
def sample_property() -> Property:
    """Fixture returning a standard single family property."""
    return Property(
        id="PROP-TEST-001",
        address="100 Market St",
        coordinates=Coordinates(lat=37.7749, lon=-122.4194),
        price=1250000.0,
        sqft=1850.0,
        bedrooms=3,
        bathrooms=2.0,
        lot_size_sqft=4500.0,
        year_built=1995,
        property_type="single_family",
        sale_date="2025-06-15",
        features={"garage": True, "renovated": True},
    )


@pytest.fixture
def sample_properties() -> List[Property]:
    """Fixture returning small list of distinct properties for quick indexing & comp tests."""
    return [
        Property(
            id="PROP-01",
            address="100 Market St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            price=1200000.0,
            sqft=1800.0,
            bedrooms=3,
            bathrooms=2.0,
            year_built=1995,
            sale_date="2025-06-01",
        ),
        Property(
            id="PROP-02",
            address="150 Market St",
            coordinates=Coordinates(lat=37.7755, lon=-122.4185),
            price=1250000.0,
            sqft=1900.0,
            bedrooms=3,
            bathrooms=2.0,
            year_built=1998,
            sale_date="2025-07-10",
        ),
        Property(
            id="PROP-03",
            address="200 Mission St",
            coordinates=Coordinates(lat=37.7760, lon=-122.4170),
            price=1400000.0,
            sqft=2100.0,
            bedrooms=4,
            bathrooms=3.0,
            year_built=2005,
            sale_date="2025-08-15",
        ),
        Property(
            id="PROP-04",
            address="50 Pine St",
            coordinates=Coordinates(lat=37.7910, lon=-122.3990),
            price=950000.0,
            sqft=1300.0,
            bedrooms=2,
            bathrooms=1.5,
            year_built=1980,
            sale_date="2025-05-20",
        ),
        Property(
            id="PROP-05",
            address="300 Castro St",
            coordinates=Coordinates(lat=37.7620, lon=-122.4350),
            price=1600000.0,
            sqft=2400.0,
            bedrooms=4,
            bathrooms=3.0,
            year_built=2010,
            sale_date="2025-09-01",
        ),
    ]


@pytest.fixture
def sample_dataset() -> List[Property]:
    """Fixture returning 50 synthetic properties for training and valuation."""
    from real_estate_spatial.data.generator import SyntheticRealEstateGenerator

    gen = SyntheticRealEstateGenerator(
        center_lat=37.7749,
        center_lon=-122.4194,
        seed=123,
    )
    return gen.generate(n_samples=50)
