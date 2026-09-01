"""Type definitions and data classes for real-estate-spatial."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Coordinates:
    """Geographic coordinates with validation and distance calculation."""

    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.lat}")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.lon}")

    def distance_to(self, other: Coordinates) -> float:
        """Calculate great-circle distance in kilometers using the Haversine formula."""
        r = 6371.0  # Earth radius in km
        d_lat = math.radians(other.lat - self.lat)
        d_lon = math.radians(other.lon - self.lon)
        lat1 = math.radians(self.lat)
        lat2 = math.radians(other.lat)

        a = (
            math.sin(d_lat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def to_tuple(self) -> Tuple[float, float]:
        """Return (latitude, longitude) tuple."""
        return (self.lat, self.lon)

    def to_dict(self) -> Dict[str, float]:
        """Convert coordinates to dictionary."""
        return {"lat": self.lat, "lon": self.lon}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Coordinates:
        """Create Coordinates instance from dictionary."""
        return cls(lat=float(data["lat"]), lon=float(data["lon"]))


@dataclass
class Property:
    """Real estate property record with spatial and structural attributes."""

    id: str
    address: str
    coordinates: Coordinates
    price: Optional[float] = None
    sqft: float = 0.0
    bedrooms: int = 0
    bathrooms: float = 0.0
    lot_size_sqft: Optional[float] = None
    year_built: Optional[int] = None
    property_type: str = "single_family"
    sale_date: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert property instance to serializable dictionary."""
        return {
            "id": self.id,
            "address": self.address,
            "coordinates": self.coordinates.to_dict(),
            "price": self.price,
            "sqft": self.sqft,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "lot_size_sqft": self.lot_size_sqft,
            "year_built": self.year_built,
            "property_type": self.property_type,
            "sale_date": self.sale_date,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Property:
        """Create Property instance from dictionary."""
        coords_raw = data.get("coordinates")
        if isinstance(coords_raw, Coordinates):
            coords = coords_raw
        elif isinstance(coords_raw, dict):
            coords = Coordinates.from_dict(coords_raw)
        elif isinstance(coords_raw, (list, tuple)) and len(coords_raw) == 2:
            coords = Coordinates(lat=float(coords_raw[0]), lon=float(coords_raw[1]))
        else:
            lat = float(data.get("lat", data.get("latitude", 0.0)))
            lon = float(data.get("lon", data.get("longitude", 0.0)))
            coords = Coordinates(lat=lat, lon=lon)

        return cls(
            id=str(data.get("id", "")),
            address=str(data.get("address", "")),
            coordinates=coords,
            price=float(data["price"]) if data.get("price") is not None else None,
            sqft=float(data.get("sqft", 0.0)),
            bedrooms=int(data.get("bedrooms", 0)),
            bathrooms=float(data.get("bathrooms", 0.0)),
            lot_size_sqft=(
                float(data["lot_size_sqft"]) if data.get("lot_size_sqft") is not None else None
            ),
            year_built=int(data["year_built"]) if data.get("year_built") is not None else None,
            property_type=str(data.get("property_type", "single_family")),
            sale_date=str(data["sale_date"]) if data.get("sale_date") is not None else None,
            features=dict(data.get("features", {})),
        )


@dataclass
class CompCandidate:
    """Comparable property candidate with similarity and value adjustments."""

    property: Property
    similarity_score: float
    distance_km: float
    time_diff_days: Optional[int]
    adjusted_price: float
    adjustments: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert comp candidate to dictionary."""
        return {
            "property": self.property.to_dict(),
            "similarity_score": round(self.similarity_score, 4),
            "distance_km": round(self.distance_km, 3),
            "time_diff_days": self.time_diff_days,
            "adjusted_price": round(self.adjusted_price, 2),
            "adjustments": {k: round(v, 2) for k, v in self.adjustments.items()},
        }


@dataclass
class ValuationResult:
    """Automated valuation result with confidence bounds and methodology metadata."""

    property_id: str
    estimated_value: float
    confidence_score: float
    lower_bound: float
    upper_bound: float
    comps_used: List[Any]
    methodology: str
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert valuation result to dictionary."""
        comps = []
        for c in self.comps_used:
            if hasattr(c, "to_dict"):
                comps.append(c.to_dict())
            elif isinstance(c, dict):
                comps.append(c)
            else:
                comps.append(str(c))

        return {
            "property_id": self.property_id,
            "estimated_value": round(self.estimated_value, 2),
            "confidence_score": round(self.confidence_score, 4),
            "lower_bound": round(self.lower_bound, 2),
            "upper_bound": round(self.upper_bound, 2),
            "comps_used": comps,
            "methodology": self.methodology,
            "feature_contributions": {
                k: round(v, 4) for k, v in self.feature_contributions.items()
            },
            "timestamp": self.timestamp,
        }


@dataclass
class ModelMetrics:
    """Evaluation metrics for spatial valuation models."""

    mape: float
    mdape: float
    rmse: float
    mae: float
    r2: float
    within_5_pct: float
    within_10_pct: float
    within_20_pct: float

    def to_dict(self) -> Dict[str, float]:
        """Convert model metrics to dictionary."""
        return {
            "mape": round(self.mape, 4),
            "mdape": round(self.mdape, 4),
            "rmse": round(self.rmse, 2),
            "mae": round(self.mae, 2),
            "r2": round(self.r2, 4),
            "within_5_pct": round(self.within_5_pct, 4),
            "within_10_pct": round(self.within_10_pct, 4),
            "within_20_pct": round(self.within_20_pct, 4),
        }
