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
    def load_csv(filepath: Union[str, Path]) -> List[Property]:
        """Load properties from a CSV file."""
        path = Path(filepath)
        properties: List[Property] = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = float(row.get("lat") or row.get("latitude", 0.0))
                lon = float(row.get("lon") or row.get("longitude", 0.0))
                price = float(row["price"]) if row.get("price") else None
                sqft = float(row.get("sqft", 0.0))
                beds = int(float(row.get("bedrooms", 0)))
                baths = float(row.get("bathrooms", 0.0))
                lot_size = (
                    float(row["lot_size_sqft"])
                    if row.get("lot_size_sqft")
                    else None
                )
                year_built = (
                    int(float(row["year_built"]))
                    if row.get("year_built")
                    else None
                )
                prop_type = row.get("property_type", "single_family")
                sale_date = row.get("sale_date") or None

                properties.append(
                    Property(
                        id=row.get("id", f"PROP-{len(properties)+1}"),
                        address=row.get("address", ""),
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
