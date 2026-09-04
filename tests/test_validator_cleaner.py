"""Unit tests for DataValidator and DataCleaner."""

from __future__ import annotations

from typing import List

import pytest

from real_estate_spatial.data import (
    CleaningReport,
    DataCleaner,
    DataValidator,
    ValidationIssue,
    ValidationResult,
)
from real_estate_spatial.types import Coordinates, Property


@pytest.fixture
def clean_properties() -> List[Property]:
    """Return list of valid properties for testing."""
    return [
        Property(
            id=f"PROP-{i:03d}",
            address=f"{100 + i} Main St",
            coordinates=Coordinates(lat=37.77 + i * 0.001, lon=-122.42 + i * 0.001),
            price=500_000.0 + i * 25_000.0,
            sqft=1500.0 + i * 50.0,
            bedrooms=3,
            bathrooms=2.0,
            lot_size_sqft=5000.0,
            year_built=1990 + (i % 20),
            property_type="single_family",
        )
        for i in range(10)
    ]


class TestValidationClasses:
    """Test ValidationIssue and ValidationResult data structures."""

    def test_validation_issue_to_dict(self) -> None:
        issue = ValidationIssue(
            property_id="P1",
            field="sqft",
            rule="invalid_sqft",
            value=-100,
            message="Square footage negative",
            severity="error",
        )
        d = issue.to_dict()
        assert d["property_id"] == "P1"
        assert d["field"] == "sqft"
        assert d["rule"] == "invalid_sqft"
        assert d["severity"] == "error"

    def test_validation_result_to_dict(self) -> None:
        issue = ValidationIssue(
            property_id="P1",
            field="price",
            rule="missing_price",
            value=None,
            message="No price",
            severity="warning",
        )
        res = ValidationResult(
            is_valid=True,
            total_records=1,
            valid_records=1,
            invalid_records=0,
            issues=[issue],
            summary={"missing_price": 1, "warnings": 1},
        )
        d = res.to_dict()
        assert d["is_valid"] is True
        assert d["total_records"] == 1
        assert len(d["issues"]) == 1
        assert d["summary"]["missing_price"] == 1


class TestDataValidator:
    """Test DataValidator rule checking and outlier detection."""

    def test_validate_clean_property(self, clean_properties: List[Property]) -> None:
        validator = DataValidator()
        issues = validator.validate_property(clean_properties[0])
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_validate_property_id_errors(self) -> None:
        validator = DataValidator()
        p = Property(
            id="",
            address="123 Main St",
            coordinates=Coordinates(lat=37.77, lon=-122.42),
            price=500000.0,
            sqft=1500.0,
        )
        issues = validator.validate_property(p)
        assert any(i.rule == "missing_id" and i.severity == "error" for i in issues)

    def test_validate_coordinates_warnings_and_errors(self) -> None:
        validator = DataValidator()
        # Null island (0, 0)
        p_null_island = Property(
            id="P-NULL",
            address="123 Main St",
            coordinates=Coordinates(lat=0.0, lon=0.0),
            price=500000.0,
            sqft=1500.0,
        )
        issues = validator.validate_property(p_null_island)
        assert any(i.rule == "null_island_coordinates" and i.severity == "warning" for i in issues)

    def test_validate_sqft_and_price_ranges(self) -> None:
        validator = DataValidator(min_sqft=300.0, max_sqft=10_000.0, min_price=50_000.0)

        # Negative sqft handled by Property init, but test sqft out of range
        p_small = Property(
            id="P-SMALL",
            address="Tiny St",
            coordinates=Coordinates(lat=37.77, lon=-122.42),
            price=100_000.0,
            sqft=200.0,
        )
        issues_small = validator.validate_property(p_small)
        assert any(i.rule == "sqft_out_of_range" for i in issues_small)

        # Price below min_price
        p_cheap = Property(
            id="P-CHEAP",
            address="Cheap St",
            coordinates=Coordinates(lat=37.77, lon=-122.42),
            price=20_000.0,
            sqft=1000.0,
        )
        issues_cheap = validator.validate_property(p_cheap)
        assert any(i.rule == "price_out_of_range" for i in issues_cheap)

    def test_validate_property_type(self) -> None:
        validator = DataValidator(allowed_property_types=["single_family", "condo"])
        p = Property(
            id="P-COMMERCIAL",
            address="Store",
            coordinates=Coordinates(lat=37.77, lon=-122.42),
            price=1_000_000.0,
            sqft=2000.0,
            property_type="commercial",
        )
        issues = validator.validate_property(p)
        assert any(i.rule == "invalid_property_type" and i.severity == "error" for i in issues)

    def test_validate_year_built_and_lot_size(self) -> None:
        validator = DataValidator(min_year_built=1800, max_year_built=2025)
        p = Property(
            id="P-OLD",
            address="Historic",
            coordinates=Coordinates(lat=37.77, lon=-122.42),
            price=800_000.0,
            sqft=2000.0,
            year_built=1650,
            lot_size_sqft=1500.0,  # smaller than sqft for single family
        )
        issues = validator.validate_property(p)
        assert any(i.rule == "year_built_out_of_range" and i.severity == "error" for i in issues)
        assert any(
            i.rule == "lot_size_smaller_than_sqft" and i.severity == "warning" for i in issues
        )

    def test_validate_dataset_success_and_raise_on_error(
        self, clean_properties: List[Property]
    ) -> None:
        validator = DataValidator()
        result = validator.validate_dataset(clean_properties, raise_on_error=True)
        assert result.is_valid is True
        assert result.total_records == len(clean_properties)
        assert result.invalid_records == 0

        # Inject invalid property
        dirty = list(clean_properties)
        dirty.append(
            Property(
                id="P-BAD",
                address="Bad",
                coordinates=Coordinates(lat=37.77, lon=-122.42),
                price=10.0,  # below default min_price
                sqft=1500.0,
            )
        )
        result_dirty = validator.validate_dataset(dirty, raise_on_error=False)
        assert result_dirty.is_valid is False
        assert result_dirty.invalid_records == 1
        assert result_dirty.valid_records == len(clean_properties)

        with pytest.raises(ValueError, match="Dataset validation failed"):
            validator.validate_dataset(dirty, raise_on_error=True)

    def test_detect_spatial_outliers_zscore_and_iqr(self) -> None:
        validator = DataValidator()
        # Create cluster of normal properties (~$300/sqft) and 1 extreme outlier ($5000/sqft)
        cluster = [
            Property(
                id=f"P-{i}",
                address=f"{i} Cluster St",
                coordinates=Coordinates(lat=37.770 + i * 0.0002, lon=-122.420 + i * 0.0002),
                price=300_000.0 + (i % 3) * 5000.0,
                sqft=1000.0,
            )
            for i in range(15)
        ]
        outlier = Property(
            id="P-OUTLIER",
            address="Outlier Palace",
            coordinates=Coordinates(lat=37.771, lon=-122.421),
            price=5_000_000.0,  # $5000/sqft
            sqft=1000.0,
        )
        dataset = cluster + [outlier]

        # Z-score method
        outliers_z = validator.detect_spatial_outliers(
            dataset, k_neighbors=10, z_threshold=2.5, method="zscore"
        )
        assert len(outliers_z) >= 1
        assert any(p.id == "P-OUTLIER" for p, score, reason in outliers_z)

        # IQR method
        outliers_iqr = validator.detect_spatial_outliers(dataset, k_neighbors=10, method="iqr")
        assert len(outliers_iqr) >= 1
        assert any(p.id == "P-OUTLIER" for p, score, reason in outliers_iqr)

        # Invalid method
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            validator.detect_spatial_outliers(dataset, method="invalid_method")

    def test_detect_spatial_outliers_small_dataset(self) -> None:
        validator = DataValidator()
        small = [
            Property(
                id="P-1",
                address="1 St",
                coordinates=Coordinates(lat=37.77, lon=-122.42),
                price=300_000.0,
                sqft=1000.0,
            )
        ]
        assert validator.detect_spatial_outliers(small) == []


class TestDataCleaner:
    """Test DataCleaner deduplication, invalid filtering, and imputation."""

    def test_clean_report_to_dict(self) -> None:
        report = CleaningReport(
            initial_count=10,
            cleaned_count=8,
            dropped_count=2,
            imputed_fields={"year_built": 3},
            dropped_reasons={"duplicate_id": 1, "validation_error": 1},
        )
        d = report.to_dict()
        assert d["initial_count"] == 10
        assert d["cleaned_count"] == 8
        assert d["dropped_count"] == 2
        assert d["imputed_fields"]["year_built"] == 3
        assert d["dropped_reasons"]["duplicate_id"] == 1

    def test_clean_dataset_deduplication(self, clean_properties: List[Property]) -> None:
        cleaner = DataCleaner()
        # Duplicate first two properties
        dataset_with_dupes = clean_properties + [clean_properties[0], clean_properties[1]]
        cleaned, report = cleaner.clean_dataset(dataset_with_dupes)

        assert len(cleaned) == len(clean_properties)
        assert report.initial_count == len(dataset_with_dupes)
        assert report.cleaned_count == len(clean_properties)
        assert report.dropped_reasons.get("duplicate_id") == 2

    def test_clean_dataset_drops_invalid(self, clean_properties: List[Property]) -> None:
        cleaner = DataCleaner(drop_invalid=True)
        bad_prop = Property(
            id="P-INVALID",
            address="Bad",
            coordinates=Coordinates(lat=37.77, lon=-122.42),
            price=10.0,  # Below min price
            sqft=1500.0,
        )
        dataset = clean_properties + [bad_prop]
        cleaned, report = cleaner.clean_dataset(dataset)

        assert len(cleaned) == len(clean_properties)
        assert all(p.id != "P-INVALID" for p in cleaned)
        assert report.dropped_reasons.get("validation_error") == 1

    def test_clean_dataset_imputation(self) -> None:
        cleaner = DataCleaner(impute_missing=True, default_lot_multiplier=2.5)
        props = [
            Property(
                id="P-1",
                address="1 St",
                coordinates=Coordinates(lat=37.77, lon=-122.42),
                price=500_000.0,
                sqft=1000.0,
                year_built=2000,
                lot_size_sqft=None,  # Missing lot size
            ),
            Property(
                id="P-2",
                address="2 St",
                coordinates=Coordinates(lat=37.78, lon=-122.41),
                price=600_000.0,
                sqft=1200.0,
                year_built=None,  # Missing year built
                lot_size_sqft=3000.0,
            ),
            Property(
                id="P-3",
                address="3 St",
                coordinates=Coordinates(lat=37.79, lon=-122.40),
                price=700_000.0,
                sqft=1400.0,
                year_built=2010,
                lot_size_sqft=3500.0,
            ),
        ]

        cleaned, report = cleaner.clean_dataset(props)
        assert len(cleaned) == 3
        # Lot size imputed for P-1: 1000 * 2.5 = 2500.0
        assert cleaned[0].lot_size_sqft == 2500.0
        # Year built imputed for P-2: median of [2000, 2010] = 2005
        assert cleaned[1].year_built == 2005
        assert report.imputed_fields.get("lot_size_sqft") == 1
        assert report.imputed_fields.get("year_built") == 1

    def test_clean_dataset_drops_outliers(self) -> None:
        cleaner = DataCleaner(drop_outliers=True, outlier_z_threshold=2.5)
        cluster = [
            Property(
                id=f"P-{i}",
                address=f"{i} St",
                coordinates=Coordinates(lat=37.770 + i * 0.0002, lon=-122.420 + i * 0.0002),
                price=300_000.0 + (i % 3) * 5000.0,
                sqft=1000.0,
                year_built=2000,
                lot_size_sqft=2000.0,
            )
            for i in range(15)
        ]
        outlier = Property(
            id="P-OUTLIER",
            address="Outlier St",
            coordinates=Coordinates(lat=37.771, lon=-122.421),
            price=5_000_000.0,
            sqft=1000.0,
            year_built=2000,
            lot_size_sqft=2000.0,
        )
        dataset = cluster + [outlier]

        cleaned, report = cleaner.clean_dataset(dataset)
        assert len(cleaned) == 15
        assert all(p.id != "P-OUTLIER" for p in cleaned)
        assert report.dropped_reasons.get("spatial_outlier") == 1

    def test_clean_dataset_drop_invalid_false_and_impute_false(
        self, clean_properties: List[Property]
    ) -> None:
        cleaner = DataCleaner(drop_invalid=False, impute_missing=False)
        bad_prop = Property(
            id="P_BAD",
            address="Bad St",
            coordinates=Coordinates(37.77, -122.42),
            price=10.0,
            sqft=1500.0,
            year_built=None,
        )
        dataset = [clean_properties[0], bad_prop]
        cleaned, report = cleaner.clean_dataset(dataset)
        assert len(cleaned) == 2
        assert any(p.id == "P_BAD" for p in cleaned)
        assert report.cleaned_count == 2
        assert report.dropped_count == 0
        assert cleaned[1].year_built is None

    def test_clean_dataset_empty_input(self) -> None:
        cleaner = DataCleaner()
        cleaned, report = cleaner.clean_dataset([])
        assert len(cleaned) == 0
        assert report.initial_count == 0
        assert report.cleaned_count == 0


class TestValidatorSubsystemEdgeCases:
    """Detailed edge cases for DataValidator rule checking."""

    def test_validate_coordinates_out_of_range_and_non_coordinates(self) -> None:
        validator = DataValidator()

        class DummyCoordinates:
            pass

        p_invalid = Property(
            id="P_DUMMY",
            address="123 Dummy St",
            coordinates=Coordinates(37.77, -122.42),
            price=500000.0,
            sqft=1500.0,
        )
        object.__setattr__(p_invalid, "coordinates", DummyCoordinates())
        issues = validator.validate_property(p_invalid)
        assert any(i.rule == "invalid_coordinates" and i.severity == "error" for i in issues)

        p_oob = Property(
            id="P_OOB",
            address="123 OOB St",
            coordinates=Coordinates(0.0, 0.0),
            price=500000.0,
            sqft=1500.0,
        )
        coord_oob = Coordinates(0.0, 0.0)
        object.__setattr__(coord_oob, "lat", 95.0)
        object.__setattr__(p_oob, "coordinates", coord_oob)
        issues_oob = validator.validate_property(p_oob)
        assert any(
            i.rule == "coordinates_out_of_range" and i.severity == "error" for i in issues_oob
        )

    def test_validate_price_missing_and_negative(self) -> None:
        validator = DataValidator()
        p_no_price = Property(
            id="P_NOPRICE",
            address="St",
            coordinates=Coordinates(37.77, -122.42),
            price=None,
            sqft=1000.0,
        )
        issues = validator.validate_property(p_no_price)
        assert any(i.rule == "missing_price" and i.severity == "warning" for i in issues)

        p_neg_price = Property(
            id="P_NEGPRICE",
            address="St",
            coordinates=Coordinates(37.77, -122.42),
            price=100.0,
            sqft=1000.0,
        )
        object.__setattr__(p_neg_price, "price", -500.0)
        issues_neg = validator.validate_property(p_neg_price)
        assert any(i.rule == "invalid_price" and i.severity == "error" for i in issues_neg)

        p_neg_sqft = Property(
            id="P_NEGSQFT",
            address="St",
            coordinates=Coordinates(37.77, -122.42),
            price=500000.0,
            sqft=1000.0,
        )
        object.__setattr__(p_neg_sqft, "sqft", -100.0)
        issues_sqft = validator.validate_property(p_neg_sqft)
        assert any(i.rule == "invalid_sqft" and i.severity == "error" for i in issues_sqft)

    def test_validate_bedrooms_bathrooms_and_lot_size_ranges(self) -> None:
        validator = DataValidator()
        p = Property(
            id="P_ROOMS",
            address="St",
            coordinates=Coordinates(37.77, -122.42),
            price=500000.0,
            sqft=1000.0,
        )
        object.__setattr__(p, "bedrooms", -2)
        object.__setattr__(p, "bathrooms", -1.0)
        object.__setattr__(p, "lot_size_sqft", -500.0)
        issues = validator.validate_property(p)
        assert any(i.rule == "invalid_bedrooms" and i.severity == "error" for i in issues)
        assert any(i.rule == "invalid_bathrooms" and i.severity == "error" for i in issues)
        assert any(i.rule == "invalid_lot_size" and i.severity == "error" for i in issues)

        object.__setattr__(p, "bedrooms", 60)
        object.__setattr__(p, "bathrooms", 55.0)
        object.__setattr__(p, "lot_size_sqft", 5000.0)
        issues_excess = validator.validate_property(p)
        assert any(
            i.rule == "excessive_bedrooms" and i.severity == "warning" for i in issues_excess
        )
        assert any(
            i.rule == "excessive_bathrooms" and i.severity == "warning" for i in issues_excess
        )

    def test_detect_spatial_outliers_insufficient_local_neighbors(self) -> None:
        validator = DataValidator()
        p1 = Property(
            id="P1",
            address="A",
            coordinates=Coordinates(lat=0.0, lon=0.0),
            price=300000.0,
            sqft=1000.0,
        )
        p2 = Property(
            id="P2",
            address="B",
            coordinates=Coordinates(lat=40.0, lon=40.0),
            price=300000.0,
            sqft=1000.0,
        )
        p3 = Property(
            id="P3",
            address="C",
            coordinates=Coordinates(lat=-40.0, lon=-40.0),
            price=300000.0,
            sqft=1000.0,
        )
        outliers = validator.detect_spatial_outliers([p1, p2, p3], k_neighbors=2)
        assert outliers == []
