"""Data cleaning, deduplication, imputation, and anomaly filtering for property datasets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from real_estate_spatial.data.validator import DataValidator
from real_estate_spatial.types import Property


@dataclass
class CleaningReport:
    """Summary report detailing dataset transformations, dropped records, and imputed values."""

    initial_count: int
    cleaned_count: int
    dropped_count: int
    imputed_fields: Dict[str, int] = field(default_factory=dict)
    dropped_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert cleaning report to dictionary."""
        return {
            "initial_count": self.initial_count,
            "cleaned_count": self.cleaned_count,
            "dropped_count": self.dropped_count,
            "imputed_fields": dict(self.imputed_fields),
            "dropped_reasons": dict(self.dropped_reasons),
        }


class DataCleaner:
    """Dataset cleaner performing deduplication, validation filtering, outlier removal, and imputation."""

    def __init__(
        self,
        validator: Optional[DataValidator] = None,
        drop_invalid: bool = True,
        drop_outliers: bool = False,
        impute_missing: bool = True,
        default_lot_multiplier: float = 2.0,
        outlier_z_threshold: float = 3.0,
    ) -> None:
        self.validator = validator or DataValidator()
        self.drop_invalid = drop_invalid
        self.drop_outliers = drop_outliers
        self.impute_missing = impute_missing
        self.default_lot_multiplier = float(default_lot_multiplier)
        self.outlier_z_threshold = float(outlier_z_threshold)

    def clean_dataset(
        self, properties: Sequence[Property]
    ) -> Tuple[List[Property], CleaningReport]:
        """Clean dataset by deduplicating, filtering errors/outliers, and imputing missing fields."""
        initial_count = len(properties)
        dropped_reasons: Dict[str, int] = {}
        imputed_fields: Dict[str, int] = {}

        # 1. Deduplication by ID (keep first occurrence)
        seen_ids: Set[str] = set()
        deduped_props: List[Property] = []
        dup_count = 0

        for prop in properties:
            if prop.id in seen_ids:
                dup_count += 1
            else:
                seen_ids.add(prop.id)
                deduped_props.append(prop)

        if dup_count > 0:
            dropped_reasons["duplicate_id"] = dup_count

        # 2. Validation & invalid record dropping
        if self.drop_invalid:
            valid_props: List[Property] = []
            invalid_count = 0
            for prop in deduped_props:
                issues = self.validator.validate_property(prop)
                has_error = any(issue.severity == "error" for issue in issues)
                if has_error:
                    invalid_count += 1
                else:
                    valid_props.append(prop)

            if invalid_count > 0:
                dropped_reasons["validation_error"] = invalid_count
        else:
            valid_props = deduped_props

        # 3. Spatial outlier detection and removal
        if self.drop_outliers and len(valid_props) > 2:
            outliers = self.validator.detect_spatial_outliers(
                valid_props,
                z_threshold=self.outlier_z_threshold,
                method="zscore",
            )
            outlier_ids = {outlier_prop.id for outlier_prop, _, _ in outliers}
            if outlier_ids:
                dropped_reasons["spatial_outlier"] = len(outlier_ids)
                filtered_props = [p for p in valid_props if p.id not in outlier_ids]
            else:
                filtered_props = valid_props
        else:
            filtered_props = valid_props

        # 4. Imputation of missing values
        cleaned_props: List[Property] = []
        if self.impute_missing and filtered_props:
            # Determine median year_built
            known_years = [
                p.year_built for p in filtered_props if p.year_built is not None
            ]
            median_year = (
                int(round(float(np.median(known_years)))) if known_years else 1990
            )

            for p in filtered_props:
                lot_imputed = False
                year_imputed = False

                new_lot_size = p.lot_size_sqft
                if new_lot_size is None or new_lot_size <= 0:
                    new_lot_size = round(p.sqft * self.default_lot_multiplier, 2)
                    lot_imputed = True
                    imputed_fields["lot_size_sqft"] = (
                        imputed_fields.get("lot_size_sqft", 0) + 1
                    )

                new_year_built = p.year_built
                if new_year_built is None:
                    new_year_built = median_year
                    year_imputed = True
                    imputed_fields["year_built"] = (
                        imputed_fields.get("year_built", 0) + 1
                    )

                if lot_imputed or year_imputed:
                    cleaned_props.append(
                        replace(
                            p,
                            lot_size_sqft=new_lot_size,
                            year_built=new_year_built,
                        )
                    )
                else:
                    cleaned_props.append(p)
        else:
            cleaned_props = filtered_props

        cleaned_count = len(cleaned_props)
        dropped_count = initial_count - cleaned_count

        report = CleaningReport(
            initial_count=initial_count,
            cleaned_count=cleaned_count,
            dropped_count=dropped_count,
            imputed_fields=imputed_fields,
            dropped_reasons=dropped_reasons,
        )

        return cleaned_props, report
