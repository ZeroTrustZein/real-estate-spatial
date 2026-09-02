"""Spatial feature extraction for real estate valuation models."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from real_estate_spatial.spatial.coordinates import haversine_distance
from real_estate_spatial.types import Coordinates, Property


class SpatialFeatureExtractor:
    """Extract spatial lag, local density, and landmark distance features."""

    def __init__(self, landmarks: Optional[Dict[str, Coordinates]] = None) -> None:
        self.landmarks = landmarks or {}

    def add_landmark(self, name: str, coord: Coordinates) -> None:
        """Add a landmark coordinate."""
        self.landmarks[name] = coord

    def spatial_lag(
        self,
        target_coord: Coordinates,
        properties: Sequence[Property],
        value_attr: str = "price",
        k: int = 5,
        decay_power: float = 1.0,
        exclude_id: Optional[str] = None,
    ) -> float:
        """Calculate inverse-distance weighted spatial lag of a property attribute."""
        valid_props = [
            p
            for p in properties
            if (exclude_id is None or p.id != exclude_id)
            and getattr(p, value_attr, None) is not None
        ]

        if not valid_props:
            return 0.0

        distances_and_vals: List[tuple[float, float]] = []
        for p in valid_props:
            val = float(getattr(p, value_attr))
            d = haversine_distance(
                target_coord.lat, target_coord.lon, p.coordinates.lat, p.coordinates.lon
            )
            distances_and_vals.append((d, val))

        distances_and_vals.sort(key=lambda x: x[0])
        neighbors = distances_and_vals[:k]

        weights_sum = 0.0
        weighted_val_sum = 0.0

        for d, val in neighbors:
            # Add epsilon to distance to prevent division by zero
            weight = 1.0 / ((max(0.01, d)) ** decay_power)
            weighted_val_sum += weight * val
            weights_sum += weight

        if weights_sum == 0.0:
            return 0.0

        return weighted_val_sum / weights_sum

    def local_density(
        self,
        target_coord: Coordinates,
        properties: Sequence[Property],
        radius_km: float = 2.0,
        exclude_id: Optional[str] = None,
    ) -> int:
        """Calculate count of properties within radius_km."""
        count = 0
        for p in properties:
            if exclude_id and p.id == exclude_id:
                continue
            d = haversine_distance(
                target_coord.lat, target_coord.lon, p.coordinates.lat, p.coordinates.lon
            )
            if d <= radius_km:
                count += 1
        return count

    def distance_to_landmarks(
        self,
        target_coord: Coordinates,
        landmarks: Optional[Dict[str, Coordinates]] = None,
    ) -> Dict[str, float]:
        """Compute distances in km to configured landmarks."""
        active_landmarks = landmarks if landmarks is not None else self.landmarks
        result: Dict[str, float] = {}
        for name, coord in active_landmarks.items():
            dist = haversine_distance(target_coord.lat, target_coord.lon, coord.lat, coord.lon)
            result[f"dist_{name}_km"] = round(dist, 4)
        return result

    def extract_all(
        self,
        prop: Property,
        background_properties: Sequence[Property],
        landmarks: Optional[Dict[str, Coordinates]] = None,
    ) -> Dict[str, float]:
        """Extract full suite of spatial features for a property."""
        features: Dict[str, float] = {}
        coord = prop.coordinates

        # Coordinates
        features["lat"] = coord.lat
        features["lon"] = coord.lon

        # Spatial lag of price and price_per_sqft
        features["spatial_lag_price_k5"] = self.spatial_lag(
            coord, background_properties, value_attr="price", k=5, exclude_id=prop.id
        )
        features["spatial_lag_price_k10"] = self.spatial_lag(
            coord, background_properties, value_attr="price", k=10, exclude_id=prop.id
        )

        # Density at 1km, 3km, 5km
        features["density_1km"] = float(
            self.local_density(coord, background_properties, radius_km=1.0, exclude_id=prop.id)
        )
        features["density_3km"] = float(
            self.local_density(coord, background_properties, radius_km=3.0, exclude_id=prop.id)
        )
        features["density_5km"] = float(
            self.local_density(coord, background_properties, radius_km=5.0, exclude_id=prop.id)
        )

        # Landmark distances
        lm_dists = self.distance_to_landmarks(coord, landmarks)
        features.update(lm_dists)

        return features
