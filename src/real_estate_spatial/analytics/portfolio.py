"""Portfolio valuation and spatial risk assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.spatial.coordinates import haversine_distance
from real_estate_spatial.types import Coordinates, Property, ValuationResult


@dataclass
class PortfolioRiskSummary:
    """Aggregated portfolio valuation metrics and risk indicators."""

    total_properties: int
    total_estimated_value: float
    mean_estimated_value: float
    median_estimated_value: float
    min_estimated_value: float
    max_estimated_value: float
    average_confidence: float
    high_risk_count: int
    high_risk_property_ids: List[str] = field(default_factory=list)
    total_lower_bound: float = 0.0
    total_upper_bound: float = 0.0
    portfolio_spread_pct: float = 0.0
    property_type_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_properties": self.total_properties,
            "total_estimated_value": round(self.total_estimated_value, 2),
            "mean_estimated_value": round(self.mean_estimated_value, 2),
            "median_estimated_value": round(self.median_estimated_value, 2),
            "min_estimated_value": round(self.min_estimated_value, 2),
            "max_estimated_value": round(self.max_estimated_value, 2),
            "average_confidence": round(self.average_confidence, 4),
            "high_risk_count": self.high_risk_count,
            "high_risk_ratio": round(self.high_risk_count / max(1, self.total_properties), 4),
            "high_risk_property_ids": self.high_risk_property_ids,
            "total_lower_bound": round(self.total_lower_bound, 2),
            "total_upper_bound": round(self.total_upper_bound, 2),
            "portfolio_spread_pct": round(self.portfolio_spread_pct, 2),
            "property_type_breakdown": self.property_type_breakdown,
        }


class PortfolioAssessor:
    """Evaluates real estate portfolios, identifying valuation dispersion and spatial risk."""

    def __init__(self, orchestrator: Optional[ValuationOrchestrator] = None) -> None:
        self.orchestrator = orchestrator or ValuationOrchestrator()

    def assess_portfolio(
        self,
        properties: Sequence[Property],
        valuation_results: Optional[Sequence[ValuationResult]] = None,
        risk_confidence_threshold: float = 0.6,
        risk_spread_threshold_pct: float = 25.0,
    ) -> Tuple[List[ValuationResult], PortfolioRiskSummary]:
        """Assess portfolio valuation, aggregated risk, and asset distributions."""
        if not properties:
            empty_summary = PortfolioRiskSummary(
                total_properties=0,
                total_estimated_value=0.0,
                mean_estimated_value=0.0,
                median_estimated_value=0.0,
                min_estimated_value=0.0,
                max_estimated_value=0.0,
                average_confidence=0.0,
                high_risk_count=0,
            )
            return [], empty_summary

        if valuation_results is None:
            results = self.orchestrator.batch_value(properties)
        else:
            results = list(valuation_results)

        values = np.array([r.estimated_value for r in results], dtype=np.float64)
        confidences = np.array([r.confidence_score for r in results], dtype=np.float64)
        lower_bounds = np.array([r.lower_bound for r in results], dtype=np.float64)
        upper_bounds = np.array([r.upper_bound for r in results], dtype=np.float64)

        total_val = float(np.sum(values))
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        avg_conf = float(np.mean(confidences))
        total_lower = float(np.sum(lower_bounds))
        total_upper = float(np.sum(upper_bounds))
        spread_pct = float(((total_upper - total_lower) / max(1.0, total_val)) * 100.0)

        threshold_ratio = (
            risk_spread_threshold_pct / 100.0
            if risk_spread_threshold_pct > 1.0
            else risk_spread_threshold_pct
        )
        high_risk_ids: List[str] = []
        for r in results:
            if r.confidence_score < risk_confidence_threshold or r.spread_pct > threshold_ratio:
                high_risk_ids.append(r.property_id)

        # Property type breakdown
        prop_map = {p.id: p for p in properties}
        type_groups: Dict[str, List[Tuple[float, float]]] = {}
        for r in results:
            p = prop_map.get(r.property_id)
            ptype = p.property_type if p else "unknown"
            if isinstance(ptype, object) and hasattr(ptype, "value"):
                ptype_str = str(ptype.value)
            else:
                ptype_str = str(ptype)
            sqft = p.sqft if p else 0.0
            if ptype_str not in type_groups:
                type_groups[ptype_str] = []
            type_groups[ptype_str].append((r.estimated_value, sqft))

        type_breakdown: Dict[str, Dict[str, float]] = {}
        for ptype, items in type_groups.items():
            count = len(items)
            tot_v = sum(v for v, _ in items)
            tot_s = sum(s for _, s in items)
            type_breakdown[ptype] = {
                "count": count,
                "total_value": round(tot_v, 2),
                "mean_value": round(tot_v / max(1, count), 2),
                "mean_sqft": round(tot_s / max(1, count), 2),
            }

        summary = PortfolioRiskSummary(
            total_properties=len(properties),
            total_estimated_value=total_val,
            mean_estimated_value=mean_val,
            median_estimated_value=median_val,
            min_estimated_value=min_val,
            max_estimated_value=max_val,
            average_confidence=avg_conf,
            high_risk_count=len(high_risk_ids),
            high_risk_property_ids=high_risk_ids,
            total_lower_bound=total_lower,
            total_upper_bound=total_upper,
            portfolio_spread_pct=spread_pct,
            property_type_breakdown=type_breakdown,
        )

        return results, summary

    def stress_test_portfolio(
        self,
        properties: Sequence[Property],
        valuation_results: Optional[Sequence[ValuationResult]] = None,
        shock_pct: float = -10.0,
        spatial_hotspots: Optional[List[Tuple[Coordinates, float, float]]] = None,
    ) -> Dict[str, Any]:
        """Perform macroeconomic and spatial stress testing on portfolio."""
        results, summary = self.assess_portfolio(properties, valuation_results=valuation_results)
        baseline_total = summary.total_estimated_value

        prop_map = {p.id: p for p in properties}
        stressed_values: List[float] = []

        for r in results:
            p = prop_map.get(r.property_id)
            multiplier = 1.0 + (shock_pct / 100.0)

            # Apply localized spatial shock if within hotspot radius
            if p and spatial_hotspots:
                for center, radius_km, local_shock_pct in spatial_hotspots:
                    dist = haversine_distance(
                        p.coordinates.lat, p.coordinates.lon, center.lat, center.lon
                    )
                    if dist <= radius_km:
                        multiplier *= 1.0 + (local_shock_pct / 100.0)

            stressed_val = max(0.0, r.estimated_value * multiplier)
            stressed_values.append(stressed_val)

        stressed_total = float(sum(stressed_values))
        value_delta = stressed_total - baseline_total
        pct_delta = float((value_delta / max(1.0, baseline_total)) * 100.0)

        return {
            "baseline_total_value": round(baseline_total, 2),
            "stressed_total_value": round(stressed_total, 2),
            "value_delta": round(value_delta, 2),
            "percentage_change": round(pct_delta, 2),
            "general_shock_pct": shock_pct,
            "spatial_hotspots_count": len(spatial_hotspots) if spatial_hotspots else 0,
        }
