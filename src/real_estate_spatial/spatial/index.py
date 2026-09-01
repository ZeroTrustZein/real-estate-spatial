"""Spatial indexing implementations for fast nearest neighbors and radius queries."""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

from real_estate_spatial.spatial.coordinates import (
    EARTH_RADIUS_KM,
    bounding_box,
    haversine_distance,
    haversine_distance_vectorized,
)
from real_estate_spatial.types import Coordinates, Property


class SpatialIndex(ABC):
    """Abstract base class for property spatial indices."""

    @abstractmethod
    def build(self, properties: Sequence[Property]) -> SpatialIndex:
        """Build index from collection of properties."""
        ...

    @abstractmethod
    def query_radius(
        self, coord: Coordinates, radius_km: float
    ) -> List[Property]:
        """Find all properties within radius_km of coordinate."""
        ...

    @abstractmethod
    def query_knn(
        self, coord: Coordinates, k: int
    ) -> List[Tuple[Property, float]]:
        """Find k nearest properties and their distances in km."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Return number of indexed properties."""
        ...

    def __len__(self) -> int:
        return self.size()


class _KDTreeNode:
    __slots__ = ("point", "prop", "left", "right", "axis")

    def __init__(
        self,
        point: np.ndarray,
        prop: Property,
        axis: int,
        left: Optional[_KDTreeNode] = None,
        right: Optional[_KDTreeNode] = None,
    ) -> None:
        self.point = point
        self.prop = prop
        self.axis = axis
        self.left = left
        self.right = right


def _lat_lon_to_cartesian(lat: float, lon: float) -> np.ndarray:
    """Convert (lat, lon) in degrees to 3D unit sphere Cartesian coordinates."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    x = math.cos(phi) * math.cos(lam)
    y = math.cos(phi) * math.sin(lam)
    z = math.sin(phi)
    return np.array([x, y, z], dtype=np.float64)


def _chord_distance_to_great_circle_km(chord_dist: float) -> float:
    """Convert 3D Euclidean chord distance on unit sphere to great-circle distance in km."""
    half_chord = min(1.0, max(0.0, chord_dist / 2.0))
    angle_rad = 2.0 * math.asin(half_chord)
    return EARTH_RADIUS_KM * angle_rad


def _great_circle_km_to_chord_distance(distance_km: float) -> float:
    """Convert great-circle distance in km to 3D Euclidean chord distance on unit sphere."""
    angle_rad = min(math.pi, distance_km / EARTH_RADIUS_KM)
    return 2.0 * math.sin(angle_rad / 2.0)


class KDTreeIndex(SpatialIndex):
    """3D Cartesian KD-Tree spatial index for spherical coordinates with O(log N) lookup."""

    def __init__(self, properties: Optional[Sequence[Property]] = None) -> None:
        self._properties: List[Property] = []
        self._root: Optional[_KDTreeNode] = None
        self._lats: np.ndarray = np.array([], dtype=np.float64)
        self._lons: np.ndarray = np.array([], dtype=np.float64)
        if properties:
            self.build(properties)

    def build(self, properties: Sequence[Property]) -> KDTreeIndex:
        """Build 3D KD-Tree from properties."""
        self._properties = list(properties)
        n = len(self._properties)
        if n == 0:
            self._root = None
            self._lats = np.array([], dtype=np.float64)
            self._lons = np.array([], dtype=np.float64)
            return self

        self._lats = np.array([p.coordinates.lat for p in self._properties], dtype=np.float64)
        self._lons = np.array([p.coordinates.lon for p in self._properties], dtype=np.float64)

        points = np.zeros((n, 3), dtype=np.float64)
        for i, p in enumerate(self._properties):
            points[i] = _lat_lon_to_cartesian(p.coordinates.lat, p.coordinates.lon)

        items = list(zip(points, self._properties))
        self._root = self._build_tree(items, depth=0)
        return self

    def _build_tree(
        self, items: List[Tuple[np.ndarray, Property]], depth: int
    ) -> Optional[_KDTreeNode]:
        if not items:
            return None

        axis = depth % 3
        items.sort(key=lambda item: item[0][axis])
        median_idx = len(items) // 2

        point, prop = items[median_idx]
        node = _KDTreeNode(
            point=point,
            prop=prop,
            axis=axis,
            left=self._build_tree(items[:median_idx], depth + 1),
            right=self._build_tree(items[median_idx + 1 :], depth + 1),
        )
        return node

    def size(self) -> int:
        return len(self._properties)

    def query_radius(
        self, coord: Coordinates, radius_km: float
    ) -> List[Property]:
        """Find properties within radius_km."""
        if self._root is None:
            return []

        target_pt = _lat_lon_to_cartesian(coord.lat, coord.lon)
        chord_threshold = _great_circle_km_to_chord_distance(radius_km)
        results: List[Property] = []

        def _search(node: Optional[_KDTreeNode]) -> None:
            if node is None:
                return

            dist = np.linalg.norm(node.point - target_pt)
            if dist <= chord_threshold:
                # Double-check with precise haversine
                real_dist = haversine_distance(
                    coord.lat, coord.lon, node.prop.coordinates.lat, node.prop.coordinates.lon
                )
                if real_dist <= radius_km:
                    results.append(node.prop)

            axis = node.axis
            plane_diff = target_pt[axis] - node.point[axis]

            first = node.left if plane_diff <= 0 else node.right
            second = node.right if plane_diff <= 0 else node.left

            _search(first)
            if abs(plane_diff) <= chord_threshold:
                _search(second)

        _search(self._root)
        return results

    def query_knn(
        self, coord: Coordinates, k: int
    ) -> List[Tuple[Property, float]]:
        """Find k nearest properties and distance in km."""
        if self._root is None or k <= 0:
            return []

        if len(self._properties) <= k:
            distances = haversine_distance_vectorized(
                coord.lat, coord.lon, self._lats, self._lons
            )
            paired = [(self._properties[i], float(distances[i])) for i in range(len(self._properties))]
            paired.sort(key=lambda x: x[1])
            return paired

        target_pt = _lat_lon_to_cartesian(coord.lat, coord.lon)
        # Max-heap storing (-chord_dist, counter, Property)
        heap: List[Tuple[float, int, Property]] = []
        counter = 0

        def _search(node: Optional[_KDTreeNode]) -> None:
            nonlocal counter
            if node is None:
                return

            dist = float(np.linalg.norm(node.point - target_pt))
            if len(heap) < k:
                heapq.heappush(heap, (-dist, counter, node.prop))
                counter += 1
            else:
                if dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, counter, node.prop))
                    counter += 1

            axis = node.axis
            plane_diff = target_pt[axis] - node.point[axis]

            first = node.left if plane_diff <= 0 else node.right
            second = node.right if plane_diff <= 0 else node.left

            _search(first)
            max_current_chord = -heap[0][0] if len(heap) == k else float("inf")
            if abs(plane_diff) < max_current_chord:
                _search(second)

        _search(self._root)

        # Convert chord distances to true great circle km and sort ascending
        results = []
        for neg_dist, _, prop in heap:
            real_dist = haversine_distance(
                coord.lat, coord.lon, prop.coordinates.lat, prop.coordinates.lon
            )
            results.append((prop, real_dist))

        results.sort(key=lambda x: x[1])
        return results[:k]


class GridIndex(SpatialIndex):
    """Spatial hash grid index with cell buckets for fast local spatial range queries."""

    def __init__(
        self,
        properties: Optional[Sequence[Property]] = None,
        cell_size_deg: float = 0.05,
    ) -> None:
        self.cell_size_deg = cell_size_deg
        self._grid: defaultdict[Tuple[int, int], List[Property]] = defaultdict(list)
        self._properties: List[Property] = []
        if properties:
            self.build(properties)

    def _get_cell_key(self, lat: float, lon: float) -> Tuple[int, int]:
        return (
            int(math.floor(lat / self.cell_size_deg)),
            int(math.floor(lon / self.cell_size_deg)),
        )

    def build(self, properties: Sequence[Property]) -> GridIndex:
        """Populate grid index with properties."""
        self._grid.clear()
        self._properties = list(properties)
        for p in self._properties:
            key = self._get_cell_key(p.coordinates.lat, p.coordinates.lon)
            self._grid[key].append(p)
        return self

    def size(self) -> int:
        return len(self._properties)

    def query_radius(
        self, coord: Coordinates, radius_km: float
    ) -> List[Property]:
        """Find properties within radius_km using bounding box grid search."""
        min_lat, min_lon, max_lat, max_lon = bounding_box(coord.lat, coord.lon, radius_km)
        min_r, min_c = self._get_cell_key(min_lat, min_lon)
        max_r, max_c = self._get_cell_key(max_lat, max_lon)

        candidates: List[Property] = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                cell_props = self._grid.get((r, c))
                if cell_props:
                    candidates.extend(cell_props)

        results: List[Property] = []
        for prop in candidates:
            dist = haversine_distance(
                coord.lat, coord.lon, prop.coordinates.lat, prop.coordinates.lon
            )
            if dist <= radius_km:
                results.append(prop)

        return results

    def query_knn(
        self, coord: Coordinates, k: int
    ) -> List[Tuple[Property, float]]:
        """Find k nearest properties and their distances in km."""
        if not self._properties or k <= 0:
            return []

        # Start with small search radius and expand if not enough candidates found
        search_radius_km = 2.0
        max_limit_km = 500.0

        while search_radius_km <= max_limit_km:
            neighbors = self.query_radius(coord, search_radius_km)
            if len(neighbors) >= k or search_radius_km >= 100.0:
                break
            search_radius_km *= 2.5

        if len(neighbors) < k:
            neighbors = self._properties

        scored = [
            (
                p,
                haversine_distance(
                    coord.lat, coord.lon, p.coordinates.lat, p.coordinates.lon
                ),
            )
            for p in neighbors
        ]
        scored.sort(key=lambda x: x[1])
        return scored[:k]
