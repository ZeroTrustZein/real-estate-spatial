"""Property dataset loader and exporter for CSV, JSON, and GeoJSON formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from real_estate_spatial.types import Coordinates, Property


class PropertyDataLoader:
    """I/O loader and serialiser for property datasets across CSV, JSON, and GeoJSON formats."""

    @staticmethod
    def load_json(filepath: Union[str, Path]) -> List[Property]:
        """Load list of properties from a JSON file."""
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "properties" in data:
            data = data["properties"]

        if not isinstance(data, list):
            raise ValueError("Expected JSON file containing a list of properties")

        return [Property.from_dict(item) for item in data]

    @staticmethod
    def save_json(
        properties: Sequence[Property],
        filepath: Union[str, Path],
        indent: int = 2,
    ) -> None:
        """Save list of properties to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in properties]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def _clean_str(val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None

    @staticmethod
    def load_csv(filepath: Union[str, Path]) -> List[Property]:
        """Load properties from a CSV file."""
        path = Path(filepath)
        properties: List[Property] = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat_str = PropertyDataLoader._clean_str(row.get("lat")) or PropertyDataLoader._clean_str(row.get("latitude"))
                lat = float(lat_str) if lat_str is not None else 0.0

                lon_str = PropertyDataLoader._clean_str(row.get("lon")) or PropertyDataLoader._clean_str(row.get("longitude"))
                lon = float(lon_str) if lon_str is not None else 0.0

                price_str = PropertyDataLoader._clean_str(row.get("price"))
                price = float(price_str) if price_str is not None else None

                sqft_str = PropertyDataLoader._clean_str(row.get("sqft"))
                sqft = float(sqft_str) if sqft_str is not None else 0.0

                beds_str = PropertyDataLoader._clean_str(row.get("bedrooms"))
                beds = int(float(beds_str)) if beds_str is not None else 0

                baths_str = PropertyDataLoader._clean_str(row.get("bathrooms"))
                baths = float(baths_str) if baths_str is not None else 0.0

                lot_str = PropertyDataLoader._clean_str(row.get("lot_size_sqft"))
                lot_size = float(lot_str) if lot_str is not None else None

                year_str = PropertyDataLoader._clean_str(row.get("year_built"))
                year_built = int(float(year_str)) if year_str is not None else None

                prop_type = PropertyDataLoader._clean_str(row.get("property_type")) or "single_family"
                sale_date = PropertyDataLoader._clean_str(row.get("sale_date"))
                prop_id = PropertyDataLoader._clean_str(row.get("id")) or f"PROP-{len(properties)+1}"
                address = PropertyDataLoader._clean_str(row.get("address")) or ""

                properties.append(
                    Property(
                        id=prop_id,
                        address=address,
                        coordinates=Coordinates(lat=lat, lon=lon),
                        price=price,
                        sqft=sqft,
                        bedrooms=beds,
                        bathrooms=baths,
                        lot_size_sqft=lot_size,
                        year_built=year_built,
                        property_type=prop_type,
                        sale_date=sale_date,
                    )
                )

        return properties

    @staticmethod
    def save_csv(
        properties: Sequence[Property], filepath: Union[str, Path]
    ) -> None:
        """Save properties to a CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "id",
            "address",
            "lat",
            "lon",
            "price",
            "sqft",
            "bedrooms",
            "bathrooms",
            "lot_size_sqft",
            "year_built",
            "property_type",
            "sale_date",
        ]

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in properties:
                writer.writerow(
                    {
                        "id": p.id,
                        "address": p.address,
                        "lat": p.coordinates.lat,
                        "lon": p.coordinates.lon,
                        "price": p.price if p.price is not None else "",
                        "sqft": p.sqft,
                        "bedrooms": p.bedrooms,
                        "bathrooms": p.bathrooms,
                        "lot_size_sqft": p.lot_size_sqft if p.lot_size_sqft is not None else "",
                        "year_built": p.year_built if p.year_built is not None else "",
                        "property_type": p.property_type,
                        "sale_date": p.sale_date or "",
                    }
                )

    @staticmethod
    def load_geojson(filepath: Union[str, Path]) -> List[Property]:
        """Load properties from GeoJSON FeatureCollection format."""
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        properties: List[Property] = []

        for feat in features:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            coords = geom.get("coordinates", [0.0, 0.0])  # GeoJSON is [lon, lat]
            lon = float(coords[0])
            lat = float(coords[1])

            prop_data = dict(props)
            prop_data["lat"] = lat
            prop_data["lon"] = lon
            if "id" not in prop_data and "id" in feat:
                prop_data["id"] = feat["id"]

            properties.append(Property.from_dict(prop_data))

        return properties

    @staticmethod
    def save_geojson(
        properties: Sequence[Property],
        filepath: Union[str, Path],
        indent: int = 2,
    ) -> None:
        """Export properties as a GeoJSON FeatureCollection."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        features: List[Dict[str, Any]] = []
        for p in properties:
            p_dict = p.to_dict()
            del p_dict["coordinates"]
            feature = {
                "type": "Feature",
                "id": p.id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [p.coordinates.lon, p.coordinates.lat],  # GeoJSON lon, lat
                },
                "properties": p_dict,
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=indent)
