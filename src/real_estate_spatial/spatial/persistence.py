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


def _resolve_format(path: Path, format: Optional[str] = None, allow_inference: bool = True) -> str:
    """Resolve file serialization format ('json' or 'binary')."""
    if format is not None:
        fmt = str(format).lower().strip()
        if fmt in ("binary", "pickle", "pkl", "bin", "dat"):
            return "binary"
        elif fmt in ("json", "geojson"):
            return "json"
        else:
            raise ValueError(
                f"Unsupported format: {format}. Supported formats are 'json' and 'binary'"
            )

    if allow_inference:
        ext = path.suffix.lower()
        if ext in (".pkl", ".pickle", ".bin", ".dat"):
            return "binary"
        elif ext in (".json", ".geojson"):
            return "json"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    json.load(f)
                return "json"
            except Exception:
                return "binary"

    return "json"


def _atomic_dump(path: Path, payload: Any, fmt: str) -> str:
    """Atomically dump payload to disk in JSON or binary format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    if fmt == "binary":
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "json":
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    tmp_path.replace(path)
    return str(path)


def _load_payload(path: Path, fmt: str) -> Any:
    """Load and return deserialized payload from disk."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if fmt == "binary":
        with path.open("rb") as f:
            return pickle.load(f)
    elif fmt == "json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


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
        fmt = _resolve_format(path, format, allow_inference=False)
        index_type = "grid" if isinstance(index, GridIndex) else "kdtree"
        params: Dict[str, Any] = {}
        if isinstance(index, GridIndex):
            params["cell_size_deg"] = index.cell_size_deg

        properties = getattr(index, "_properties", [])
        payload: Dict[str, Any] = {
            "version": "0.1.0",
            "index_type": index_type,
            "property_count": len(properties),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "properties": [p.to_dict() for p in properties],
        }
        if fmt == "binary":
            payload["index"] = index

        return _atomic_dump(path, payload, fmt)

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

        fmt = _resolve_format(path, format, allow_inference=True)
        data = _load_payload(path, fmt)

        if isinstance(data, SpatialIndex):
            return data
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object for spatial index, got {type(data)}")

        if "index" in data and isinstance(data["index"], SpatialIndex):
            return data["index"]

        index_type = data.get("index_type", "kdtree").lower()
        params = data.get("params", {})
        props_raw = data.get("properties", [])
        props = [p if isinstance(p, Property) else Property.from_dict(p) for p in props_raw]
        if index_type == "grid":
            cell_size = float(params.get("cell_size_deg", 0.05))
            return GridIndex(props, cell_size_deg=cell_size)
        return KDTreeIndex(props)

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
        fmt = _resolve_format(path, format, allow_inference=False)
        payload = [p.to_dict() for p in properties]
        return _atomic_dump(path, payload, fmt)

    @staticmethod
    def load_properties(
        file_path: Union[str, Path],
        format: Optional[str] = None,
    ) -> List[Property]:
        """Load a list of properties from disk in JSON or binary format."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Properties file not found: {path}")

        fmt = _resolve_format(path, format, allow_inference=True)
        data = _load_payload(path, fmt)

        if isinstance(data, dict) and "properties" in data:
            data = data["properties"]
        if not isinstance(data, list):
            raise ValueError(f"Expected list of properties, got {type(data)}")
        return [p if isinstance(p, Property) else Property.from_dict(p) for p in data]


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

    def get_knn(self, coord: Coordinates, k: int) -> Optional[List[Tuple[Property, float]]]:
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

    def get_radius(self, coord: Coordinates, radius_km: float) -> Optional[List[Property]]:
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

    def get_lag(self, coord: Coordinates, attr: str, k: int) -> Optional[float]:
        """Retrieve cached spatial lag value."""
        key = ("lag", self._round_coord(coord), attr, k)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return float(self._cache[key])
        self._misses += 1
        return None

    def set_lag(self, coord: Coordinates, attr: str, k: int, val: float) -> None:
        """Store spatial lag calculation into cache."""
        key = ("lag", self._round_coord(coord), attr, k)
        self._cache[key] = float(val)
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
