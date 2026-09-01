"""Property similarity scoring for comparable selection."""

from __future__ import annotations

import math
from typing import Optional

from real_estate_spatial.spatial.coordinates import haversine_distance
from real_estate_spatial.types import Property


class SimilarityScorer:
    """Calculates weighted similarity score [0.0, 1.0] between a subject and comparable property."""

    def __init__(
        self,
        weight_distance: float = 0.35,
        weight_sqft: float = 0.25,
        weight_beds: float = 0.15,
        weight_baths: float = 0.10,
        weight_age: float = 0.15,
        max_distance_km: float = 10.0,
    ) -> None:
        total_weight = (
            weight_distance + weight_sqft + weight_beds + weight_baths + weight_age
        )
        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive")

        # Normalize weights
        self.w_distance = weight_distance / total_weight
        self.w_sqft = weight_sqft / total_weight
        self.w_beds = weight_beds / total_weight
        self.w_baths = weight_baths / total_weight
        self.w_age = weight_age / total_weight
        self.max_distance_km = max(0.1, max_distance_km)

    def score(
        self,
        subject: Property,
        comp: Property,
        distance_km: Optional[float] = None,
    ) -> float:
        """Calculate normalized similarity score between subject and comp in [0.0, 1.0]."""
        if distance_km is None:
            distance_km = haversine_distance(
                subject.coordinates.lat,
                subject.coordinates.lon,
                comp.coordinates.lat,
                comp.coordinates.lon,
            )

        # Distance score: exponential decay
        dist_sim = math.exp(-2.0 * (distance_km / self.max_distance_km))
        dist_sim = max(0.0, min(1.0, dist_sim))

        # Sqft score: percentage difference
        ref_sqft = max(100.0, subject.sqft)
        sqft_pct_diff = abs(subject.sqft - comp.sqft) / ref_sqft
        sqft_sim = max(0.0, 1.0 - sqft_pct_diff)

        # Bedroom score
        bed_diff = abs(subject.bedrooms - comp.bedrooms)
        bed_sim = max(0.0, 1.0 - (bed_diff / 3.0))

        # Bathroom score
        bath_diff = abs(subject.bathrooms - comp.bathrooms)
        bath_sim = max(0.0, 1.0 - (bath_diff / 2.0))

        # Year built / age score
        if subject.year_built is not None and comp.year_built is not None:
            year_diff = abs(subject.year_built - comp.year_built)
            age_sim = max(0.0, 1.0 - (year_diff / 40.0))
        else:
            age_sim = 0.8  # neutral default

        # Property type bonus/penalty
        type_multiplier = (
            1.0 if subject.property_type == comp.property_type else 0.8
        )

        composite_score = (
            self.w_distance * dist_sim
            + self.w_sqft * sqft_sim
            + self.w_beds * bed_sim
            + self.w_baths * bath_sim
            + self.w_age * age_sim
        ) * type_multiplier

        return max(0.0, min(1.0, composite_score))
