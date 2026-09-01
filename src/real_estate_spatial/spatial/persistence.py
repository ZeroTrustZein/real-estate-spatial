"""Serialization, persistence, and caching utilities for spatial indices and queries."""

from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from real_estate_spatial.spatial.index import GridIndex, KDTreeIndex, SpatialIndex
from real_estate_spatial.types import Coordinates, Property


class SpatialIndexSerializer:
    """Serializer and deserializer for spatial indices and property datasets."""

    @staticmethod
    def save_index(
        index: SpatialIndex,
        file_path: Union[str, Path],
        format: str = "json",
    ) -> str:
        """Save a SpatialIndex (KDTreeIndex or GridIndex) to disk in JSON or binary format.

        Returns:
            The string path to the saved file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format.lower()

        index_type = "grid" if isinstance(index, GridIndex) else "kdtree"
        params: Dict[str, Any] = {}
        if isinstance(index, GridIndex):
            params["cell_size_deg"] = index.cell_size_deg

        properties = getattr(index, "_properties", [])
        tmp_path = path.with_name(f".{path.name}.tmp")

        if fmt in ("binary", "pickle", "pkl", "bin"):
            payload = {
                "version": "0.1.0",
                "index_type": index_type,
                "property_count": len(properties),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "params": params,
                "properties": [p.to_dict() for p in properties],
                "index": index,
            }
            with tmp_path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt == "json":
            payload = {
                "version": "0.1.0",
                "index_type": index_type,
                "property_count": len(properties),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "params": params,
                "properties": [p.to_dict() for p in properties],
            }
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats are 'json' and 'binary'")

        tmp_path.replace(path)
        return str(path)

    @staticmethod
    def load_index(
        file_path: Union[str, Path],
        format: Optional[str] = None,
    ) -> SpatialIndex:
        """Load and reconstruct a SpatialIndex from disk.

        Infers format from file extension if format is not explicitly passed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        fmt = format.lower() if format is not None else None
        if fmt is None:
            ext = path.suffix.lower()
            if ext in (".pkl", ".pickle", ".bin", ".dat"):
                fmt = "binary"
            elif ext in (".json", ".geojson"):
                fmt = "json"
            else:
                # Attempt JSON first, fallback to binary
                try:
                    with path.open("r", encoding="utf-8") as f:
                        json.load(f)
                    fmt = "json"
                except Exception:
                    fmt = "binary"

        if fmt in ("binary", "pickle", "pkl", "bin"):
            with path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, SpatialIndex):
                return obj
            if isinstance(obj, dict):
                if "index" in obj and isinstance(obj["index"], SpatialIndex):
                    return obj["index"]
                index_type = obj.get("index_type", "kdtree").lower()
                params = obj.get("params", {})
                props_raw = obj.get("properties", [])
                props = [
                    p if isinstance(p, Property) else Property.from_dict(p)
                    for p in props_raw
                ]
                if index_type == "grid":
                    cell_size = float(params.get("cell_size_deg", 0.05))
                    return GridIndex(props, cell_size_deg=cell_size)
                return KDTreeIndex(props)
            raise ValueError(f"Invalid serialized index binary payload: {type(obj)}")

        elif fmt == "json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object for spatial index, got {type(data)}")

            index_type = data.get("index_type", "kdtree").lower()
            params = data.get("params", {})
            props_raw = data.get("properties", [])
            props = [Property.from_dict(p) for p in props_raw]

            if index_type == "grid":
                cell_size = float(params.get("cell_size_deg", 0.05))
                return GridIndex(props, cell_size_deg=cell_size)
            return KDTreeIndex(props)

        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def save_properties(
        properties: Sequence[Property],
        file_path: Union[str, Path],
        format: str = "json",
    ) -> str:
        """Save a collection of properties to disk in JSON or binary format.

        Returns:
            The string path to the saved file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format.lower()
        tmp_path = path.with_name(f".{path.name}.tmp")

        if fmt in ("binary", "pickle", "pkl", "bin"):
            with tmp_path.open("wb") as f:
                pickle.dump([p.to_dict() for p in properties], f, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt == "json":
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump([p.to_dict() for p in properties], f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        tmp_path.replace(path)
        return str(path)

    @staticmethod
    def load_properties(
        file_path: Union[str, Path],
        format: Optional[str] = None,
    ) -> List[Property]:
        """Load a list of properties from disk in JSON or binary format."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Properties file not found: {path}")

        fmt = format.lower() if format is not None else None
        if fmt is None:
            ext = path.suffix.lower()
            if ext in (".pkl", ".pickle", ".bin", ".dat"):
                fmt = "binary"
            elif ext in (".json", ".geojson"):
                fmt = "json"
            else:
                try:
                    with path.open("r", encoding="utf-8") as f:
                        json.load(f)
                    fmt = "json"
                except Exception:
                    fmt = "binary"

        if fmt in ("binary", "pickle", "pkl", "bin"):
            with path.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                return [
                    p if isinstance(p, Property) else Property.from_dict(p)
                    for p in data
                ]
            if isinstance(data, dict) and "properties" in data:
                return [
                    p if isinstance(p, Property) else Property.from_dict(p)
                    for p in data["properties"]
                ]
            raise ValueError(f"Unexpected binary structure for properties: {type(data)}")

        elif fmt == "json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "properties" in data:
                data = data["properties"]
            if not isinstance(data, list):
                raise ValueError(f"Expected list of properties in JSON, got {type(data)}")
            return [Property.from_dict(p) for p in data]

        else:
            raise ValueError(f"Unsupported format: {format}")


class SpatialQueryCache:
    """In-memory LRU cache for spatial kNN, radius queries, and spatial lag calculations."""

    def __init__(self, max_size: int = 5000, precision: int = 5) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = max_size
        self.precision = precision
        self._cache: OrderedDict[Tuple[Any, ...], Any] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def _round_coord(self, coord: Coordinates) -> Tuple[float, float]:
        return (round(coord.lat, self.precision), round(coord.lon, self.precision))

    @property
    def hits(self) -> int:
        """Total cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate percentage [0.0, 1.0]."""
        total = self._hits + self._misses
        return float(self._hits / total) if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Current number of items in cache."""
        return len(self._cache)

    def __len__(self) -> int:
        """Return current number of items in cache."""
        return self.size

    @property
    def capacity(self) -> int:
        """Maximum cache capacity."""
        return self._max_size

    def clear(self) -> None:
        """Clear cache entries and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_knn(
        self, coord: Coordinates, k: int
    ) -> Optional[List[Tuple[Property, float]]]:
        """Retrieve cached k-nearest neighbor query results."""
        key = ("knn", self._round_coord(coord), k)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return list(self._cache[key])
        self._misses += 1
        return None

    def set_knn(
        self,
        coord: Coordinates,
        k: int,
        results: List[Tuple[Property, float]],
    ) -> None:
        """Store k-nearest neighbor query results into cache."""
        key = ("knn", self._round_coord(coord), k)
        self._cache[key] = list(results)
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def get_radius(
        self, coord: Coordinates, radius_km: float
    ) -> Optional[List[Property]]:
        """Retrieve cached radius query results."""
        key = ("radius", self._round_coord(coord), round(radius_km, self.precision))
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return list(self._cache[key])
        self._misses += 1
        return None

    def set_radius(
        self,
        coord: Coordinates,
        radius_km: float,
        results: List[Property],
    ) -> None:
        """Store radius query results into cache."""
        key = ("radius", self._round_coord(coord), round(radius_km, self.precision))
        self._cache[key] = list(results)
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def get_lag(
        self, coord: Coordinates, attr: str, k: int
    ) -> Optional[float]:
        """Retrieve cached spatial lag value."""
        key = ("lag", self._round_coord(coord), attr, k)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return float(self._cache[key])
        self._misses += 1
        return None

    def set_lag(
        self, coord: Coordinates, attr: str, k: int, val: float
    ) -> None:
        """Store spatial lag calculation into cache."""
        key = ("lag", self._round_coord(coord), attr, k)
        self._cache[key] = float(val)
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
