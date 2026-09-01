"""Automated Property Comparables (Comps) Valuation Engine."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import List, Optional, Sequence

import numpy as np

from real_estate_spatial.comps.adjustments import CompAdjustmentCalculator
from real_estate_spatial.comps.scorer import SimilarityScorer
from real_estate_spatial.spatial.coordinates import haversine_distance
from real_estate_spatial.spatial.index import KDTreeIndex, SpatialIndex
from real_estate_spatial.types import CompCandidate, Property, ValuationResult


class CompsEngine:
    """Finds top comparable properties within radius, calculates adjustments, and estimates property value."""

    def __init__(
        self,
        index: Optional[SpatialIndex] = None,
        scorer: Optional[SimilarityScorer] = None,
        adjuster: Optional[CompAdjustmentCalculator] = None,
    ) -> None:
        self.index: SpatialIndex = index or KDTreeIndex()
        self.scorer: SimilarityScorer = scorer or SimilarityScorer()
        self.adjuster: CompAdjustmentCalculator = adjuster or CompAdjustmentCalculator()
        self._properties: List[Property] = []

    def fit(self, properties: Sequence[Property]) -> CompsEngine:
        """Fit the comparables engine with property sale records."""
        # Only use properties with known price for comps
        self._properties = [p for p in properties if p.price is not None and p.price > 0]
        self.index.build(self._properties)
        return self

    def _parse_date(self, d_str: Optional[str]) -> Optional[date]:
        if not d_str:
            return None
        try:
            iso_str = d_str.replace("Z", "+00:00") if d_str.endswith("Z") else d_str
            return datetime.fromisoformat(iso_str).date()
        except (ValueError, TypeError):
            pass
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(d_str, fmt).date()
            except ValueError:
                continue
        return None

    def find_comps(
        self,
        subject: Property,
        max_comps: int = 5,
        max_radius_km: float = 10.0,
        max_sale_age_days: Optional[int] = 365 * 2,
        min_similarity: float = 0.3,
    ) -> List[CompCandidate]:
        """Find and rank comparable properties for the subject property."""
        if len(self._properties) == 0:
            return []

        # Find nearby candidates using spatial index
        nearby = self.index.query_radius(subject.coordinates, max_radius_km)
        if len(nearby) < max_comps:
            # Fallback to kNN if radius returned too few candidates
            knn_res = self.index.query_knn(subject.coordinates, k=max(max_comps * 2, 20))
            nearby = [p for p, _ in knn_res]

        today = date.today()
        candidates: List[CompCandidate] = []

        for comp in nearby:
            # Skip self
            if comp.id == subject.id and subject.id:
                continue

            # Calculate distance
            dist_km = haversine_distance(
                subject.coordinates.lat,
                subject.coordinates.lon,
                comp.coordinates.lat,
                comp.coordinates.lon,
            )

            # Check sale date age if configured
            time_diff_days: Optional[int] = None
            if comp.sale_date:
                sale_d = self._parse_date(comp.sale_date)
                if sale_d:
                    time_diff_days = abs((today - sale_d).days)
                    if max_sale_age_days is not None and time_diff_days > max_sale_age_days:
                        continue

            sim_score = self.scorer.score(subject, comp, distance_km=dist_km)
            if sim_score < min_similarity:
                continue

            comp_price = comp.price or 0.0
            adjusted_price, adjustments = self.adjuster.calculate_adjustments(
                subject, comp, comp_price=comp_price
            )

            candidates.append(
                CompCandidate(
                    property=comp,
                    similarity_score=sim_score,
                    distance_km=dist_km,
                    time_diff_days=time_diff_days,
                    adjusted_price=adjusted_price,
                    adjustments=adjustments,
                )
            )

        # Sort by similarity score descending
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return candidates[:max_comps]

    def value(
        self,
        subject: Property,
        max_comps: int = 5,
        max_radius_km: float = 10.0,
    ) -> ValuationResult:
        """Estimate property value using weighted comparable sales."""
        comps = self.find_comps(
            subject, max_comps=max_comps, max_radius_km=max_radius_km, min_similarity=0.1
        )

        if not comps:
            # Fallback when no comps available
            default_val = max(50000.0, subject.sqft * self.adjuster.default_price_per_sqft)
            return ValuationResult(
                property_id=subject.id,
                estimated_value=default_val,
                confidence_score=0.1,
                lower_bound=default_val * 0.8,
                upper_bound=default_val * 1.2,
                comps_used=[],
                methodology="fallback_sqft_heuristic",
                feature_contributions={},
            )

        total_weight = sum(c.similarity_score for c in comps)
        if total_weight <= 0:
            total_weight = float(len(comps))
            weights = [1.0 / total_weight] * len(comps)
        else:
            weights = [c.similarity_score / total_weight for c in comps]

        adjusted_prices = np.array([c.adjusted_price for c in comps], dtype=np.float64)
        weights_arr = np.array(weights, dtype=np.float64)

        estimated_val = float(np.sum(adjusted_prices * weights_arr))

        # Variance and confidence estimation
        weighted_var = float(
            np.sum(weights_arr * ((adjusted_prices - estimated_val) ** 2))
        )
        std_dev = math.sqrt(weighted_var) if weighted_var > 0 else (estimated_val * 0.05)

        avg_similarity = float(np.mean([c.similarity_score for c in comps]))
        comp_count_factor = min(1.0, len(comps) / max(1.0, float(max_comps)))
        dispersion_factor = max(0.2, 1.0 - (std_dev / max(1.0, estimated_val)))
        confidence_score = float(
            np.clip(
                0.5 * avg_similarity + 0.3 * comp_count_factor + 0.2 * dispersion_factor,
                0.1,
                0.99,
            )
        )

        # Margin of error based on confidence
        margin_pct = max(0.04, (1.0 - confidence_score) * 0.25)
        lower_bound = max(1000.0, estimated_val * (1.0 - margin_pct))
        upper_bound = estimated_val * (1.0 + margin_pct)

        # Feature contributions
        avg_adjustments: dict[str, float] = {}
        for c in comps:
            for k, v in c.adjustments.items():
                avg_adjustments[k] = avg_adjustments.get(k, 0.0) + v / len(comps)

        return ValuationResult(
            property_id=subject.id,
            estimated_value=estimated_val,
            confidence_score=confidence_score,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            comps_used=comps,
            methodology="spatial_comparables_weighted_average",
            feature_contributions=avg_adjustments,
        )
