"""Data validation and spatial outlier detection for real estate property datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from real_estate_spatial.spatial.index import KDTreeIndex
from real_estate_spatial.types import Coordinates, Property, PropertyType


@dataclass
class ValidationIssue:
    """Individual data validation failure or warning record."""

    property_id: str
    field: str
    rule: str
    value: Any
    message: str
    severity: str = "error"  # "error" | "warning"

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation issue to dictionary."""
        return {
            "property_id": self.property_id,
            "field": self.field,
            "rule": self.rule,
            "value": self.value,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """Aggregated validation output containing issue details and summary metrics."""

    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": dict(self.summary),
        }


class DataValidator:
    """Property record validator with rule-based and spatial anomaly detection."""

    def __init__(
        self,
        min_sqft: float = 100.0,
        max_sqft: float = 50000.0,
        min_price: float = 1000.0,
        max_price: float = 100_000_000.0,
        min_price_per_sqft: float = 10.0,
        max_price_per_sqft: float = 20000.0,
        min_year_built: int = 1700,
        max_year_built: int = 2035,
        allowed_property_types: Optional[Sequence[Union[str, PropertyType]]] = None,
    ) -> None:
        self.min_sqft = float(min_sqft)
        self.max_sqft = float(max_sqft)
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.min_price_per_sqft = float(min_price_per_sqft)
        self.max_price_per_sqft = float(max_price_per_sqft)
        self.min_year_built = int(min_year_built)
        self.max_year_built = int(max_year_built)

        if allowed_property_types is None:
            self.allowed_property_types = {pt.value for pt in PropertyType}
        else:
            self.allowed_property_types = {
                pt.value if isinstance(pt, PropertyType) else str(pt)
                for pt in allowed_property_types
            }

    def validate_property(self, prop: Property) -> List[ValidationIssue]:
        """Validate an individual property against structural and value integrity rules."""
        issues: List[ValidationIssue] = []

        # Identifier check
        if not prop.id or not str(prop.id).strip():
            issues.append(
                ValidationIssue(
                    property_id=str(prop.id),
                    field="id",
                    rule="missing_id",
                    value=prop.id,
                    message="Property ID must not be empty",
                    severity="error",
                )
            )

        # Coordinates check
        if not isinstance(prop.coordinates, Coordinates):
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="coordinates",
                    rule="invalid_coordinates",
                    value=prop.coordinates,
                    message="Coordinates must be a valid Coordinates instance",
                    severity="error",
                )
            )
        else:
            lat, lon = prop.coordinates.lat, prop.coordinates.lon
            if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="coordinates",
                        rule="coordinates_out_of_range",
                        value=(lat, lon),
                        message=f"Coordinates out of bounds: ({lat}, {lon})",
                        severity="error",
                    )
                )
            elif lat == 0.0 and lon == 0.0:
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="coordinates",
                        rule="null_island_coordinates",
                        value=(0.0, 0.0),
                        message="Coordinates located at (0.0, 0.0)",
                        severity="warning",
                    )
                )

        # Address check
        if not prop.address or not str(prop.address).strip():
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="address",
                    rule="missing_address",
                    value=prop.address,
                    message="Address is empty or missing",
                    severity="warning",
                )
            )

        # Square footage checks
        if prop.sqft <= 0:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="sqft",
                    rule="invalid_sqft",
                    value=prop.sqft,
                    message=f"Square footage must be positive, got {prop.sqft}",
                    severity="error",
                )
            )
        elif prop.sqft < self.min_sqft or prop.sqft > self.max_sqft:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="sqft",
                    rule="sqft_out_of_range",
                    value=prop.sqft,
                    message=f"Square footage {prop.sqft} outside range [{self.min_sqft}, {self.max_sqft}]",
                    severity="error",
                )
            )

        # Price checks
        if prop.price is not None:
            if prop.price <= 0:
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="price",
                        rule="invalid_price",
                        value=prop.price,
                        message=f"Price must be positive, got {prop.price}",
                        severity="error",
                    )
                )
            elif prop.price < self.min_price or prop.price > self.max_price:
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="price",
                        rule="price_out_of_range",
                        value=prop.price,
                        message=f"Price {prop.price} outside range [{self.min_price}, {self.max_price}]",
                        severity="error",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="price",
                    rule="missing_price",
                    value=None,
                    message="Property price is not specified",
                    severity="warning",
                )
            )

        # Price per square foot check
        if prop.price is not None and prop.sqft > 0:
            pps = prop.price / prop.sqft
            if pps < self.min_price_per_sqft or pps > self.max_price_per_sqft:
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="price_per_sqft",
                        rule="price_per_sqft_out_of_range",
                        value=round(pps, 2),
                        message=f"Price per sqft ${pps:.2f} outside range [{self.min_price_per_sqft}, {self.max_price_per_sqft}]",
                        severity="error",
                    )
                )

        # Property type check
        prop_type_val = (
            prop.property_type.value
            if isinstance(prop.property_type, PropertyType)
            else str(prop.property_type)
        )
        if prop_type_val not in self.allowed_property_types:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="property_type",
                    rule="invalid_property_type",
                    value=prop.property_type,
                    message=f"Property type '{prop.property_type}' not in allowed types: {sorted(self.allowed_property_types)}",
                    severity="error",
                )
            )

        # Bedrooms & bathrooms checks
        if prop.bedrooms < 0:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="bedrooms",
                    rule="invalid_bedrooms",
                    value=prop.bedrooms,
                    message=f"Bedrooms cannot be negative, got {prop.bedrooms}",
                    severity="error",
                )
            )
        elif prop.bedrooms > 50:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="bedrooms",
                    rule="excessive_bedrooms",
                    value=prop.bedrooms,
                    message=f"High bedroom count: {prop.bedrooms}",
                    severity="warning",
                )
            )

        if prop.bathrooms < 0:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="bathrooms",
                    rule="invalid_bathrooms",
                    value=prop.bathrooms,
                    message=f"Bathrooms cannot be negative, got {prop.bathrooms}",
                    severity="error",
                )
            )
        elif prop.bathrooms > 50:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="bathrooms",
                    rule="excessive_bathrooms",
                    value=prop.bathrooms,
                    message=f"High bathroom count: {prop.bathrooms}",
                    severity="warning",
                )
            )

        # Year built checks
        if prop.year_built is not None:
            if prop.year_built < self.min_year_built or prop.year_built > self.max_year_built:
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="year_built",
                        rule="year_built_out_of_range",
                        value=prop.year_built,
                        message=f"Year built {prop.year_built} outside range [{self.min_year_built}, {self.max_year_built}]",
                        severity="error",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="year_built",
                    rule="missing_year_built",
                    value=None,
                    message="Year built is missing",
                    severity="warning",
                )
            )

        # Lot size checks
        if prop.lot_size_sqft is not None:
            if prop.lot_size_sqft < 0:
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="lot_size_sqft",
                        rule="invalid_lot_size",
                        value=prop.lot_size_sqft,
                        message=f"Lot size cannot be negative, got {prop.lot_size_sqft}",
                        severity="error",
                    )
                )
            elif prop.lot_size_sqft < prop.sqft and prop_type_val in ("single_family", "multi_family"):
                issues.append(
                    ValidationIssue(
                        property_id=prop.id,
                        field="lot_size_sqft",
                        rule="lot_size_smaller_than_sqft",
                        value=prop.lot_size_sqft,
                        message=f"Lot size {prop.lot_size_sqft} sqft is smaller than living area {prop.sqft} sqft",
                        severity="warning",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    property_id=prop.id,
                    field="lot_size_sqft",
                    rule="missing_lot_size",
                    value=None,
                    message="Lot size is missing",
                    severity="warning",
                )
            )

        return issues

    def validate_dataset(
        self, properties: Sequence[Property], raise_on_error: bool = False
    ) -> ValidationResult:
        """Validate a sequence of property records and aggregate findings."""
        all_issues: List[ValidationIssue] = []
        invalid_ids = set()

        for prop in properties:
            prop_issues = self.validate_property(prop)
            all_issues.extend(prop_issues)
            if any(issue.severity == "error" for issue in prop_issues):
                invalid_ids.add(prop.id)

        total_records = len(properties)
        invalid_records = len(invalid_ids)
        valid_records = total_records - invalid_records
        is_valid = invalid_records == 0

        summary: Dict[str, int] = {
            "total_errors": sum(1 for i in all_issues if i.severity == "error"),
            "total_warnings": sum(1 for i in all_issues if i.severity == "warning"),
        }
        for issue in all_issues:
            summary[issue.rule] = summary.get(issue.rule, 0) + 1

        if raise_on_error and not is_valid:
            error_details = [
                f"[{i.property_id}] {i.field}: {i.message}"
                for i in all_issues
                if i.severity == "error"
            ]
            raise ValueError(
                f"Dataset validation failed with {invalid_records} invalid records: "
                + "; ".join(error_details[:5])
                + (f" ... and {len(error_details) - 5} more errors" if len(error_details) > 5 else "")
            )

        return ValidationResult(
            is_valid=is_valid,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            issues=all_issues,
            summary=summary,
        )

    def detect_spatial_outliers(
        self,
        properties: Sequence[Property],
        k_neighbors: int = 15,
        z_threshold: float = 3.0,
        method: str = "zscore",
    ) -> List[Tuple[Property, float, str]]:
        """Detect local price per sqft outliers relative to spatial k-nearest neighbors."""
        priced_props = [p for p in properties if p.price is not None and p.sqft > 0]
        if len(priced_props) < 3:
            return []

        method_clean = method.lower().strip()
        if method_clean not in ("zscore", "iqr"):
            raise ValueError(f"Unknown outlier detection method: '{method}'. Supported methods: 'zscore', 'iqr'")

        index = KDTreeIndex(priced_props)
        outliers: List[Tuple[Property, float, str]] = []

        for prop in priced_props:
            target_pps = prop.price / prop.sqft  # type: ignore[operator]
            k_query = min(len(priced_props), k_neighbors + 1)
            knn_res = index.query_knn(prop.coordinates, k=k_query)

            # Exclude current property
            neighbor_pps = [
                n.price / n.sqft  # type: ignore[operator]
                for n, dist in knn_res
                if n.id != prop.id and n.price is not None and n.sqft > 0
            ]

            if len(neighbor_pps) < 3:
                continue

            if method_clean == "zscore":
                mean_val = float(np.mean(neighbor_pps))
                std_val = float(np.std(neighbor_pps))
                if std_val > 1e-6:
                    z = (target_pps - mean_val) / std_val
                    if abs(z) >= z_threshold:
                        reason = (
                            f"Price/sqft ${target_pps:.2f} deviates from local mean ${mean_val:.2f} "
                            f"(z-score: {z:.2f}, threshold: {z_threshold})"
                        )
                        outliers.append((prop, float(z), reason))

            elif method_clean == "iqr":
                q25 = float(np.percentile(neighbor_pps, 25))
                q75 = float(np.percentile(neighbor_pps, 75))
                iqr = q75 - q25
                if iqr > 1e-6:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    median_val = float(np.median(neighbor_pps))
                    if target_pps < lower_bound or target_pps > upper_bound:
                        score = float((target_pps - median_val) / iqr)
                        reason = (
                            f"Price/sqft ${target_pps:.2f} outside local IQR bounds "
                            f"[${lower_bound:.2f}, ${upper_bound:.2f}] (IQR score: {score:.2f})"
                        )
                        outliers.append((prop, score, reason))

        return outliers
