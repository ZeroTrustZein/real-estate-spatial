"""Hedonic Spatial Regression model using ridge/OLS regression on spatial and structural features."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from real_estate_spatial.models.base import (
    BaseValuationModel,
    extract_structural_features,
)
from real_estate_spatial.spatial.coordinates import haversine_distance
from real_estate_spatial.types import Property, ValuationResult


class HedonicSpatialRegression(BaseValuationModel):
    """Hedonic spatial regression model for property valuation."""

    FEATURE_NAMES = [
        "intercept",
        "sqft",
        "sqft_squared",
        "bedrooms",
        "bathrooms",
        "age",
        "lot_size_sqft",
        "lat",
        "lon",
        "lat_sq",
        "lon_sq",
        "lat_lon_interaction",
        "dist_to_center_km",
    ]

    def __init__(
        self,
        alpha: float = 1.0,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
    ) -> None:
        super().__init__(name="HedonicSpatialRegression")
        self.alpha = alpha
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.coefficients: np.ndarray = np.array([])
        self.feature_means: np.ndarray = np.array([])
        self.feature_stds: np.ndarray = np.array([])

    def _extract_feature_row(self, prop: Property) -> np.ndarray:
        struct_feats = extract_structural_features(prop)
        c_lat = self.center_lat if self.center_lat is not None else prop.coordinates.lat
        c_lon = self.center_lon if self.center_lon is not None else prop.coordinates.lon
        dist_center = haversine_distance(
            prop.coordinates.lat, prop.coordinates.lon, c_lat, c_lon
        )

        lat = prop.coordinates.lat
        lon = prop.coordinates.lon

        # Raw features without intercept (added during standardization)
        return np.array(
            [
                struct_feats["sqft"],
                struct_feats["sqft_squared"],
                struct_feats["bedrooms"],
                struct_feats["bathrooms"],
                struct_feats["age"],
                struct_feats["lot_size_sqft"],
                lat,
                lon,
                lat**2,
                lon**2,
                lat * lon,
                dist_center,
            ],
            dtype=np.float64,
        )

    def _build_design_matrix(
        self, properties: Sequence[Property]
    ) -> np.ndarray:
        rows = [self._extract_feature_row(p) for p in properties]
        return np.vstack(rows) if rows else np.empty((0, 12))

    def fit(self, properties: Sequence[Property]) -> HedonicSpatialRegression:
        """Fit hedonic regression coefficients with L2 (ridge) regularization."""
        valid_props = [p for p in properties if p.price is not None and p.price > 0]
        if len(valid_props) < 3:
            raise ValueError("Need at least 3 valid property sales to fit hedonic model")

        self._training_properties = list(valid_props)

        # Infer center if not provided
        if self.center_lat is None or self.center_lon is None:
            self.center_lat = float(
                np.mean([p.coordinates.lat for p in valid_props])
            )
            self.center_lon = float(
                np.mean([p.coordinates.lon for p in valid_props])
            )

        X_raw = self._build_design_matrix(valid_props)
        y = np.array([p.price for p in valid_props], dtype=np.float64)

        # Standardize features
        self.feature_means = np.mean(X_raw, axis=0)
        self.feature_stds = np.std(X_raw, axis=0)
        self.feature_stds[self.feature_stds == 0.0] = 1.0

        X_scaled = (X_raw - self.feature_means) / self.feature_stds

        # Add intercept column of 1s
        n = len(valid_props)
        X = np.hstack([np.ones((n, 1), dtype=np.float64), X_scaled])

        # Ridge regularizer (do not penalize intercept)
        reg_matrix = self.alpha * np.eye(X.shape[1], dtype=np.float64)
        reg_matrix[0, 0] = 0.0

        # Normal equation: (X^T X + alpha I) w = X^T y
        A = X.T @ X + reg_matrix
        b = X.T @ y

        try:
            self.coefficients = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        self.is_fitted = True
        return self

    def predict(self, properties: Sequence[Property]) -> np.ndarray:
        """Predict market values for a batch of properties."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if not properties:
            return np.array([], dtype=np.float64)

        X_raw = self._build_design_matrix(properties)
        X_scaled = (X_raw - self.feature_means) / self.feature_stds
        n = len(properties)
        X = np.hstack([np.ones((n, 1), dtype=np.float64), X_scaled])

        preds = X @ self.coefficients
        return np.maximum(10000.0, preds)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return normalized absolute feature coefficients (excluding intercept)."""
        if not self.is_fitted or len(self.coefficients) == 0:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        coefs = np.abs(self.coefficients[1:])
        total = float(np.sum(coefs))
        if total > 0:
            return coefs / total
        return np.ones(len(coefs), dtype=np.float64) / max(1, len(coefs))

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dictionary mapping feature names (excluding intercept) to relative importance."""
        importances = self.feature_importances_
        return {name: float(imp) for name, imp in zip(self.FEATURE_NAMES[1:], importances)}

    def get_feature_contributions(self, prop: Property) -> Dict[str, float]:
        """Calculate additive feature contributions to property value."""
        if not self.is_fitted:
            return {}

        x_raw = self._extract_feature_row(prop)
        x_scaled = (x_raw - self.feature_means) / self.feature_stds
        x_full = np.hstack([[1.0], x_scaled])

        contributions: Dict[str, float] = {}
        for name, coef, val in zip(self.FEATURE_NAMES, self.coefficients, x_full):
            contributions[name] = float(coef * val)
        return contributions

    def predict_one(self, prop: Property) -> ValuationResult:
        """Predict valuation for one property with feature contributions."""
        preds = self.predict([prop])
        val = float(preds[0])
        contributions = self.get_feature_contributions(prop)

        return ValuationResult(
            property_id=prop.id,
            estimated_value=val,
            confidence_score=0.82,
            lower_bound=val * 0.92,
            upper_bound=val * 1.08,
            comps_used=[],
            methodology="hedonic_spatial_ridge_regression",
            feature_contributions=contributions,
        )
