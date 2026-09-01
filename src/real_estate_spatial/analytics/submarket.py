"""Submarket clustering, spatial aggregation, and market reporting."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from real_estate_spatial.spatial.coordinates import haversine_distance
from real_estate_spatial.types import Coordinates, Property


@dataclass
class SubmarketMetrics:
    """Aggregated market statistics for a spatial cluster or submarket."""

    submarket_id: str
    property_count: int
    center_lat: float
    center_lon: float
    median_price: float
    mean_price: float
    median_price_per_sqft: float
    mean_price_per_sqft: float
    min_price: float
    max_price: float
    price_std_dev: float
    price_volatility_pct: float
    median_sqft: float
    property_types: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submarket_id": self.submarket_id,
            "property_count": self.property_count,
            "center_lat": round(self.center_lat, 6),
            "center_lon": round(self.center_lon, 6),
            "median_price": round(self.median_price, 2),
            "mean_price": round(self.mean_price, 2),
            "median_price_per_sqft": round(self.median_price_per_sqft, 2),
            "mean_price_per_sqft": round(self.mean_price_per_sqft, 2),
            "min_price": round(self.min_price, 2),
            "max_price": round(self.max_price, 2),
            "price_std_dev": round(self.price_std_dev, 2),
            "price_volatility_pct": round(self.price_volatility_pct, 2),
            "median_sqft": round(self.median_sqft, 2),
            "property_types": self.property_types,
        }


class SubmarketAnalyzer:
    """Groups properties into spatial micro-markets and extracts pricing dynamics."""

    def __init__(
        self,
        grid_cell_size_deg: float = 0.05,
        min_cluster_size: int = 2,
    ) -> None:
        self.grid_cell_size_deg = grid_cell_size_deg
        self.min_cluster_size = min_cluster_size

    def _get_cell_key(self, lat: float, lon: float) -> Tuple[int, int]:
        return (
            int(math.floor(lat / self.grid_cell_size_deg)),
            int(math.floor(lon / self.grid_cell_size_deg)),
        )

    def identify_submarkets(
        self,
        properties: Sequence[Property],
        method: str = "grid",
    ) -> List[SubmarketMetrics]:
        """Identify spatial clusters and calculate submarket metrics."""
        valid_props = [p for p in properties if p.price is not None and p.price > 0]
        if not valid_props:
            return []

        clusters: Dict[str, List[Property]] = defaultdict(list)

        if method == "grid":
            for p in valid_props:
                r, c = self._get_cell_key(p.coordinates.lat, p.coordinates.lon)
                cell_id = f"SUBM-{r}_{c}"
                clusters[cell_id].append(p)
        else:
            # Simple radial spatial clustering
            assigned = set()
            cluster_idx = 1
            for i, seed in enumerate(valid_props):
                if seed.id in assigned:
                    continue
                cid = f"SUBM-CLUST-{cluster_idx}"
                clusters[cid].append(seed)
                assigned.add(seed.id)

                for other in valid_props[i + 1 :]:
                    if other.id in assigned:
                        continue
                    d = haversine_distance(
                        seed.coordinates.lat,
                        seed.coordinates.lon,
                        other.coordinates.lat,
                        other.coordinates.lon,
                    )
                    if d <= (self.grid_cell_size_deg * 111.0):  # approx km
                        clusters[cid].append(other)
                        assigned.add(other.id)
                cluster_idx += 1

        submarkets: List[SubmarketMetrics] = []
        for cid, props in clusters.items():
            if len(props) < self.min_cluster_size:
                continue

            prices = np.array([p.price for p in props], dtype=np.float64)
            pps_vals = np.array([p.price_per_sqft for p in props], dtype=np.float64)
            sqfts = np.array([p.sqft for p in props], dtype=np.float64)
            lats = [p.coordinates.lat for p in props]
            lons = [p.coordinates.lon for p in props]

            mean_p = float(np.mean(prices))
            std_p = float(np.std(prices))
            volatility = float((std_p / max(1.0, mean_p)) * 100.0)

            types_count: Dict[str, int] = defaultdict(int)
            for p in props:
                pt = p.property_type
                pt_str = pt.value if hasattr(pt, "value") else str(pt)
                types_count[pt_str] += 1

            sm = SubmarketMetrics(
                submarket_id=cid,
                property_count=len(props),
                center_lat=float(np.mean(lats)),
                center_lon=float(np.mean(lons)),
                median_price=float(np.median(prices)),
                mean_price=mean_p,
                median_price_per_sqft=float(np.median(pps_vals)),
                mean_price_per_sqft=float(np.mean(pps_vals)),
                min_price=float(np.min(prices)),
                max_price=float(np.max(prices)),
                price_std_dev=std_p,
                price_volatility_pct=volatility,
                median_sqft=float(np.median(sqfts)),
                property_types=dict(types_count),
            )
            submarkets.append(sm)

        submarkets.sort(key=lambda s: s.median_price, reverse=True)
        return submarkets

    def generate_market_report(self, properties: Sequence[Property]) -> Dict[str, Any]:
        """Generate comprehensive market overview and submarket comparisons."""
        submarkets = self.identify_submarkets(properties)
        valid_props = [p for p in properties if p.price is not None and p.price > 0]

        if not valid_props:
            return {
                "total_properties": len(properties),
                "priced_properties": 0,
                "submarket_count": 0,
                "overall_median_price": 0.0,
                "overall_median_price_per_sqft": 0.0,
                "submarkets": [],
            }

        prices = [p.price for p in valid_props]
        pps_list = [p.price_per_sqft for p in valid_props]

        # Top rankings
        by_price = sorted(submarkets, key=lambda s: s.median_price, reverse=True)
        by_volatility = sorted(submarkets, key=lambda s: s.price_volatility_pct, reverse=True)

        return {
            "total_properties": len(properties),
            "priced_properties": len(valid_props),
            "submarket_count": len(submarkets),
            "overall_median_price": round(float(np.median(prices)), 2),
            "overall_mean_price": round(float(np.mean(prices)), 2),
            "overall_median_price_per_sqft": round(float(np.median(pps_list)), 2),
            "overall_mean_price_per_sqft": round(float(np.mean(pps_list)), 2),
            "submarkets": [s.to_dict() for s in submarkets],
            "top_expensive_submarkets": [s.to_dict() for s in by_price[:3]],
            "top_affordable_submarkets": [s.to_dict() for s in by_price[-3:]],
            "most_volatile_submarkets": [s.to_dict() for s in by_volatility[:3]],
        }
