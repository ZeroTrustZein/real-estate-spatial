"""Type definitions and data classes for real-estate-spatial."""

from __future__ import annotations

import builtins
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class PropertyType(str, Enum):
    """Supported property classification types."""

    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi_family"
    COMMERCIAL = "commercial"
    LAND = "land"


class ValuationMethodology(str, Enum):
    """Valuation model and strategy methodology identifiers."""

    COMPS = "comps"
    HEDONIC = "hedonic"
    GBM = "gbm"
    ENSEMBLE = "ensemble"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class Coordinates:
    """Geographic coordinates with validation and distance calculation."""

    lat: float
    lon: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.lat, bool)
            or isinstance(self.lon, bool)
            or not isinstance(self.lat, (int, float))
            or not isinstance(self.lon, (int, float))
        ):
            raise TypeError(
                f"Coordinates lat and lon must be numeric, got lat={type(self.lat)}, lon={type(self.lon)}"
            )
        if math.isnan(self.lat) or math.isnan(self.lon):
            raise ValueError("Coordinates cannot be NaN")
        if math.isinf(self.lat) or math.isinf(self.lon):
            raise ValueError("Coordinates cannot be Infinite")
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.lat}")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.lon}")

        object.__setattr__(self, "lat", float(self.lat))
        object.__setattr__(self, "lon", float(self.lon))

    def distance_to(self, other: Coordinates, radius_km: float = 6371.0) -> float:
        """Calculate great-circle distance in kilometers using the Haversine formula."""
        if not isinstance(other, Coordinates):
            raise TypeError(f"Expected Coordinates instance, got {type(other)}")
        d_lat = math.radians(other.lat - self.lat)
        d_lon = math.radians(other.lon - self.lon)
        lat1 = math.radians(self.lat)
        lat2 = math.radians(other.lat)

        a = (
            math.sin(d_lat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
        )
        a = min(1.0, max(0.0, a))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return float(radius_km * c)

    def to_tuple(self) -> Tuple[float, float]:
        """Return (latitude, longitude) tuple."""
        return (self.lat, self.lon)

    def to_dict(self) -> Dict[str, float]:
        """Convert coordinates to dictionary."""
        return {"lat": self.lat, "lon": self.lon}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Coordinates:
        """Create Coordinates instance from dictionary."""
        lat = data.get("lat", data.get("latitude"))
        lon = data.get("lon", data.get("longitude", data.get("lng")))
        if lat is None or lon is None:
            raise ValueError(f"Missing lat or lon in dictionary: {data}")
        return cls(lat=float(lat), lon=float(lon))

    @classmethod
    def from_tuple(cls, data: Sequence[float]) -> Coordinates:
        """Create Coordinates instance from a (lat, lon) sequence."""
        if len(data) != 2:
            raise ValueError(f"Expected sequence of 2 items (lat, lon), got {len(data)}")
        return cls(lat=float(data[0]), lon=float(data[1]))


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box defined by min/max latitudes and longitudes."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def __post_init__(self) -> None:
        coords = [self.min_lat, self.min_lon, self.max_lat, self.max_lon]
        for val in coords:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"BoundingBox coordinates must be numeric, got {type(val)}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError("BoundingBox coordinates cannot be NaN or Infinite")

        min_lat_f = float(self.min_lat)
        min_lon_f = float(self.min_lon)
        max_lat_f = float(self.max_lat)
        max_lon_f = float(self.max_lon)

        if not (-90.0 <= min_lat_f <= 90.0) or not (-90.0 <= max_lat_f <= 90.0):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180.0 <= min_lon_f <= 180.0) or not (-180.0 <= max_lon_f <= 180.0):
            raise ValueError("Longitude must be between -180 and 180")
        if min_lat_f > max_lat_f:
            raise ValueError(f"min_lat ({min_lat_f}) cannot be greater than max_lat ({max_lat_f})")
        if min_lon_f > max_lon_f:
            raise ValueError(f"min_lon ({min_lon_f}) cannot be greater than max_lon ({max_lon_f})")

        object.__setattr__(self, "min_lat", min_lat_f)
        object.__setattr__(self, "min_lon", min_lon_f)
        object.__setattr__(self, "max_lat", max_lat_f)
        object.__setattr__(self, "max_lon", max_lon_f)

    def contains(self, point: Coordinates | Sequence[float]) -> bool:
        """Check if a coordinate point lies inside this bounding box."""
        if isinstance(point, Coordinates):
            lat, lon = point.lat, point.lon
        elif (
            isinstance(point, (tuple, list, Sequence))
            and not isinstance(point, (str, bytes))
            and len(point) == 2
        ):
            lat, lon = float(point[0]), float(point[1])
        else:
            raise TypeError(f"Expected Coordinates or 2-element sequence, got {type(point)}")
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return (min_lat, min_lon, max_lat, max_lon) tuple."""
        return (self.min_lat, self.min_lon, self.max_lat, self.max_lon)

    def to_dict(self) -> Dict[str, float]:
        """Convert bounding box to dictionary."""
        return {
            "min_lat": self.min_lat,
            "min_lon": self.min_lon,
            "max_lat": self.max_lat,
            "max_lon": self.max_lon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox:
        """Create BoundingBox instance from dictionary."""
        return cls(
            min_lat=float(data["min_lat"]),
            min_lon=float(data["min_lon"]),
            max_lat=float(data["max_lat"]),
            max_lon=float(data["max_lon"]),
        )

    @classmethod
    def from_tuple(cls, bbox: Sequence[float]) -> BoundingBox:
        """Create BoundingBox instance from sequence of 4 floats."""
        if len(bbox) != 4:
            raise ValueError(
                f"Expected sequence of 4 items (min_lat, min_lon, max_lat, max_lon), got {len(bbox)}"
            )
        return cls(
            min_lat=float(bbox[0]),
            min_lon=float(bbox[1]),
            max_lat=float(bbox[2]),
            max_lon=float(bbox[3]),
        )


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

    def __post_init__(self) -> None:
        if isinstance(self.property_type, Enum):
            self.property_type = self.property_type.value
        if not isinstance(self.coordinates, Coordinates):
            if isinstance(self.coordinates, (list, tuple)):
                self.coordinates = Coordinates.from_tuple(self.coordinates)
            elif isinstance(self.coordinates, dict):
                self.coordinates = Coordinates.from_dict(self.coordinates)
            else:
                raise TypeError(f"Invalid coordinates type: {type(self.coordinates)}")

        if self.sqft < 0:
            raise ValueError(f"sqft cannot be negative, got {self.sqft}")
        if self.bedrooms < 0:
            raise ValueError(f"bedrooms cannot be negative, got {self.bedrooms}")
        if self.bathrooms < 0:
            raise ValueError(f"bathrooms cannot be negative, got {self.bathrooms}")
        if self.price is not None and self.price < 0:
            raise ValueError(f"price cannot be negative, got {self.price}")
        if self.lot_size_sqft is not None and self.lot_size_sqft < 0:
            raise ValueError(f"lot_size_sqft cannot be negative, got {self.lot_size_sqft}")

    @property
    def price_per_sqft(self) -> Optional[float]:
        """Compute price per square foot if both price and sqft are valid."""
        if self.price is not None and self.sqft > 0:
            return self.price / self.sqft
        return None

    def age(self, ref_year: int = 2026) -> float:
        """Calculate property age relative to reference year (default 2026).

        Falls back to 25.0 if year_built is not set or invalid.
        """
        if self.year_built is not None:
            try:
                yb = int(self.year_built)
                return float(max(0, int(ref_year) - yb))
            except (ValueError, TypeError):
                return 25.0
        return 25.0

    def effective_lot_size(self) -> float:
        """Return lot size in sqft if present and positive, otherwise estimate as 3x sqft."""
        lot = getattr(self, "lot_size_sqft", None) or getattr(self, "lot_sqft", None)
        if lot is not None and lot > 0:
            return float(lot)
        return float(self.sqft * 3.0)

    @property
    def lot_sqft(self) -> Optional[float]:
        """Convenience alias for lot_size_sqft."""
        return self.lot_size_sqft

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
            "property_type": str(self.property_type),
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
            coords = Coordinates.from_tuple(coords_raw)
        elif "lat" in data or "latitude" in data:
            lat = float(data.get("lat", data.get("latitude", 0.0)))
            lon = float(data.get("lon", data.get("longitude", data.get("lng", 0.0))))
            coords = Coordinates(lat=lat, lon=lon)
        else:
            lat = float(data.get("lat", 0.0))
            lon = float(data.get("lon", 0.0))
            coords = Coordinates(lat=lat, lon=lon)

        price_raw = data.get("price")
        price = float(price_raw) if price_raw is not None and price_raw != "" else None

        lot_size_raw = data.get("lot_size_sqft")
        lot_size = float(lot_size_raw) if lot_size_raw is not None and lot_size_raw != "" else None

        year_built_raw = data.get("year_built")
        year_built = (
            int(float(year_built_raw))
            if year_built_raw is not None and year_built_raw != ""
            else None
        )

        sale_date_raw = data.get("sale_date")
        sale_date = (
            str(sale_date_raw)
            if sale_date_raw is not None and str(sale_date_raw).strip() != ""
            else None
        )

        return cls(
            id=str(data.get("id", "")),
            address=str(data.get("address", "")),
            coordinates=coords,
            price=price,
            sqft=float(data.get("sqft", 0.0)),
            bedrooms=int(float(data.get("bedrooms", 0))),
            bathrooms=float(data.get("bathrooms", 0.0)),
            lot_size_sqft=lot_size,
            year_built=year_built,
            property_type=str(data.get("property_type", "single_family")),
            sale_date=sale_date,
            features=dict(data.get("features", {})),
        )

    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert property into a GeoJSON Feature dictionary."""
        p_dict = self.to_dict()
        del p_dict["coordinates"]
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {
                "type": "Point",
                "coordinates": [self.coordinates.lon, self.coordinates.lat],
            },
            "properties": p_dict,
        }


@dataclass
class CompCandidate:
    """Comparable property candidate with similarity and value adjustments."""

    property: Property
    similarity_score: float
    distance_km: float
    time_diff_days: Optional[int]
    adjusted_price: float
    adjustments: Dict[str, float] = field(default_factory=dict)

    @builtins.property
    def net_adjustment(self) -> float:
        """Sum of all positive and negative adjustment amounts."""
        return sum(self.adjustments.values())

    @builtins.property
    def gross_adjustment(self) -> float:
        """Sum of absolute adjustment amounts."""
        return sum(abs(v) for v in self.adjustments.values())

    @builtins.property
    def net_adjustment_pct(self) -> float:
        """Percentage of net adjustment relative to original property price."""
        if self.property.price is not None and self.property.price > 0:
            return self.net_adjustment / self.property.price
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert comp candidate to dictionary."""
        return {
            "property": self.property.to_dict(),
            "similarity_score": round(self.similarity_score, 4),
            "distance_km": round(self.distance_km, 3),
            "time_diff_days": self.time_diff_days,
            "adjusted_price": round(self.adjusted_price, 2),
            "adjustments": {k: round(v, 2) for k, v in self.adjustments.items()},
            "net_adjustment": round(self.net_adjustment, 2),
            "gross_adjustment": round(self.gross_adjustment, 2),
            "net_adjustment_pct": round(self.net_adjustment_pct, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompCandidate:
        """Create CompCandidate instance from dictionary."""
        prop_raw = data.get("property")
        if isinstance(prop_raw, Property):
            prop = prop_raw
        elif isinstance(prop_raw, dict):
            prop = Property.from_dict(prop_raw)
        else:
            raise ValueError("CompCandidate data missing valid property")

        return cls(
            property=prop,
            similarity_score=float(data.get("similarity_score", 0.0)),
            distance_km=float(data.get("distance_km", 0.0)),
            time_diff_days=(
                int(data["time_diff_days"]) if data.get("time_diff_days") is not None else None
            ),
            adjusted_price=float(data.get("adjusted_price", 0.0)),
            adjustments={k: float(v) for k, v in data.get("adjustments", {}).items()},
        )


@dataclass
class ValuationResult:
    """Automated valuation result with confidence bounds and methodology metadata."""

    property_id: str
    estimated_value: float
    confidence_score: float
    lower_bound: float
    upper_bound: float
    comps_used: List[Any] = field(default_factory=list)
    methodology: str = "valuation"
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def spread(self) -> float:
        """Absolute spread between upper and lower valuation bounds."""
        return self.upper_bound - self.lower_bound

    @property
    def spread_pct(self) -> float:
        """Relative spread percentage to estimated value."""
        if self.estimated_value > 0:
            return (self.upper_bound - self.lower_bound) / self.estimated_value
        return 0.0

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
            "spread": round(self.spread, 2),
            "spread_pct": round(self.spread_pct, 4),
            "comps_used": comps,
            "methodology": self.methodology,
            "feature_contributions": {
                k: round(v, 4) for k, v in self.feature_contributions.items()
            },
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValuationResult:
        """Create ValuationResult instance from dictionary."""
        raw_comps = data.get("comps_used", [])
        comps: List[Any] = []
        for c in raw_comps:
            if isinstance(c, dict):
                if "property" in c:
                    comps.append(CompCandidate.from_dict(c))
                elif "id" in c and "coordinates" in c:
                    comps.append(Property.from_dict(c))
                else:
                    comps.append(c)
            else:
                comps.append(c)

        return cls(
            property_id=str(data.get("property_id", "")),
            estimated_value=float(data.get("estimated_value", 0.0)),
            confidence_score=float(data.get("confidence_score", 0.0)),
            lower_bound=float(data.get("lower_bound", 0.0)),
            upper_bound=float(data.get("upper_bound", 0.0)),
            comps_used=comps,
            methodology=str(data.get("methodology", "")),
            feature_contributions={
                k: float(v) for k, v in data.get("feature_contributions", {}).items()
            },
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetrics:
        """Create ModelMetrics instance from dictionary."""
        return cls(
            mape=float(data["mape"]),
            mdape=float(data["mdape"]),
            rmse=float(data["rmse"]),
            mae=float(data["mae"]),
            r2=float(data["r2"]),
            within_5_pct=float(data["within_5_pct"]),
            within_10_pct=float(data["within_10_pct"]),
            within_20_pct=float(data["within_20_pct"]),
        )

    def summary(self) -> str:
        """Return formatted string summary of model evaluation metrics."""
        return (
            f"Model Metrics Summary:\n"
            f"  MAPE:          {self.mape:.2f}%\n"
            f"  MdAPE:         {self.mdape:.2f}%\n"
            f"  MAE:           ${self.mae:,.2f}\n"
            f"  RMSE:          ${self.rmse:,.2f}\n"
            f"  R²:            {self.r2:.4f}\n"
            f"  Within 5%:     {self.within_5_pct:.2f}%\n"
            f"  Within 10%:    {self.within_10_pct:.2f}%\n"
            f"  Within 20%:    {self.within_20_pct:.2f}%"
        )
