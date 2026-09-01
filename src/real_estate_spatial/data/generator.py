"""Synthetic spatial real estate dataset generator."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import List, Optional

import numpy as np

from real_estate_spatial.spatial.coordinates import haversine_distance
from real_estate_spatial.types import Coordinates, Property


class SyntheticRealEstateGenerator:
    """Generates realistic geospatial property datasets with spatial price gradients and clusters."""

    def __init__(
        self,
        center_lat: float = 37.7749,
        center_lon: float = -122.4194,
        base_price_sqft: float = 650.0,
        cbd_gradient: float = -20.0,  # $/sqft decrease per km from CBD
        seed: Optional[int] = 42,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.base_price_sqft = base_price_sqft
        self.cbd_gradient = cbd_gradient
        self.seed = seed

    def generate(self, n_samples: int = 500, seed: Optional[int] = None) -> List[Property]:
        """Generate n realistic synthetic property records with spatial pricing effects."""
        active_seed = seed if seed is not None else self.seed
        if active_seed is not None:
            np.random.seed(active_seed)
            random.seed(active_seed)

        # Neighborhood clusters with premium factors
        clusters = [
            {"lat": self.center_lat, "lon": self.center_lon, "premium": 1.3, "weight": 0.3},
            {"lat": self.center_lat + 0.04, "lon": self.center_lon - 0.05, "premium": 1.15, "weight": 0.25},
            {"lat": self.center_lat - 0.05, "lon": self.center_lon + 0.03, "premium": 0.85, "weight": 0.25},
            {"lat": self.center_lat + 0.06, "lon": self.center_lon + 0.06, "premium": 0.95, "weight": 0.20},
        ]
        cluster_weights = [c["weight"] for c in clusters]

        properties: List[Property] = []
        today = date.today()

        street_names = [
            "Market St", "Mission St", "Pine St", "Bush St", "California St",
            "Valencia St", "Folsom St", "Howard St", "Harrison St", "Geary Blvd",
            "Divisadero St", "Haight St", "Castro St", "Fillmore St", "Columbus Ave"
        ]

        for i in range(n_samples):
            # Pick a cluster
            cluster = np.random.choice(clusters, p=cluster_weights)
            lat = cluster["lat"] + np.random.normal(0.0, 0.018)
            lon = cluster["lon"] + np.random.normal(0.0, 0.018)

            # Clamp lat/lon
            lat = max(-89.9, min(89.9, lat))
            lon = max(-179.9, min(179.9, lon))

            # Property attributes
            bedrooms = int(np.clip(np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05]), 1, 6))
            bathrooms = float(bedrooms * 0.5 + np.random.choice([0.5, 1.0, 1.5]))
            bathrooms = round(np.clip(bathrooms, 1.0, 5.0) * 2) / 2.0

            base_sqft = bedrooms * 550.0 + bathrooms * 200.0
            sqft = round(float(np.clip(np.random.normal(base_sqft, 250.0), 500.0, 6000.0)))

            year_built = int(np.clip(np.random.normal(1985, 25), 1900, 2024))
            lot_size_sqft = round(sqft * np.random.uniform(1.5, 4.0))

            prop_type = np.random.choice(
                ["single_family", "condo", "townhouse"], p=[0.55, 0.35, 0.10]
            )

            # Spatial distance to CBD (center)
            dist_cbd_km = haversine_distance(lat, lon, self.center_lat, self.center_lon)

            # Compute realistic market price
            ppsqft = (
                self.base_price_sqft
                + self.cbd_gradient * dist_cbd_km
            ) * cluster["premium"]
            ppsqft = max(200.0, ppsqft)

            # Adjust for age and bathrooms
            age = max(0, 2026 - year_built)
            age_factor = 1.0 - (age * 0.002)
            noise = np.random.normal(1.0, 0.06)

            price = round(
                sqft * ppsqft * age_factor * (1.0 + (bedrooms - 2) * 0.05) * noise,
                -2,
            )
            price = max(75000.0, price)

            # Sale date within last 2 years
            days_ago = int(np.random.exponential(180))
            sale_date = (today - timedelta(days=min(730, days_ago))).isoformat()

            address_num = random.randint(100, 4999)
            street = random.choice(street_names)
            address = f"{address_num} {street}"

            prop = Property(
                id=f"PROP-{i+1:05d}",
                address=address,
                coordinates=Coordinates(lat=round(lat, 6), lon=round(lon, 6)),
                price=float(price),
                sqft=float(sqft),
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                lot_size_sqft=float(lot_size_sqft),
                year_built=year_built,
                property_type=prop_type,
                sale_date=sale_date,
                features={
                    "dist_cbd_km": round(dist_cbd_km, 3),
                    "cluster_premium": cluster["premium"],
                },
            )
            properties.append(prop)

        return properties
