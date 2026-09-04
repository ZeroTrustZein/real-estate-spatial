"""Spatial Gradient Boosting Valuation Model with XGBoost and NumPy fallback."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from real_estate_spatial.models.base import (
    BaseValuationModel,
    extract_structural_features,
)
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.types import Property, ValuationResult

# Try importing xgboost
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    HAS_XGBOOST = False


class _SimpleDecisionTreeLeaf:
    def __init__(self, value: float) -> None:
        self.value = value


class _SimpleDecisionTreeNode:
    def __init__(
        self,
        feature_idx: int,
        threshold: float,
        left: Any,
        right: Any,
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right


class _SimpleDecisionTreeRegressor:
    """Lightweight pure NumPy regression tree for fallback gradient boosting."""

    def __init__(self, max_depth: int = 4, min_samples_split: int = 5) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SimpleDecisionTreeRegressor:
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Any:
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split or np.all(y == y[0]):
            return _SimpleDecisionTreeLeaf(float(np.mean(y)))

        best_feat = -1
        best_thresh = 0.0
        best_loss = float("inf")

        current_loss = np.sum((y - np.mean(y)) ** 2)

        for feat in range(n_features):
            values = np.unique(X[:, feat])
            if len(values) <= 1:
                continue

            # Check candidate thresholds (percentiles)
            if len(values) > 10:
                thresholds = np.percentile(values, np.linspace(10, 90, 9))
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0

            for thresh in thresholds:
                mask_left = X[:, feat] <= thresh
                mask_right = ~mask_left

                if np.sum(mask_left) < 2 or np.sum(mask_right) < 2:
                    continue

                y_left = y[mask_left]
                y_right = y[mask_right]

                loss = np.sum((y_left - np.mean(y_left)) ** 2) + np.sum(
                    (y_right - np.mean(y_right)) ** 2
                )
                if loss < best_loss:
                    best_loss = loss
                    best_feat = feat
                    best_thresh = thresh

        if best_feat == -1 or best_loss >= current_loss:
            return _SimpleDecisionTreeLeaf(float(np.mean(y)))

        mask_l = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[mask_l], y[mask_l], depth + 1)
        right_child = self._build_tree(X[~mask_l], y[~mask_l], depth + 1)

        return _SimpleDecisionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
        )

    def _predict_row(self, node: Any, row: np.ndarray) -> float:
        if isinstance(node, _SimpleDecisionTreeLeaf):
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(self.root, row) for row in X], dtype=np.float64)


class _NumpyGradientBoostingRegressor:
    """NumPy gradient boosting ensemble fallback."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 4,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: List[_SimpleDecisionTreeRegressor] = []
        self.init_val: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _NumpyGradientBoostingRegressor:
        self.init_val = float(np.mean(y))
        y_pred = np.full(len(y), self.init_val, dtype=np.float64)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = _SimpleDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            step_pred = tree.predict(X)
            y_pred += self.learning_rate * step_pred
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.full(len(X), self.init_val, dtype=np.float64)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds

    def get_feature_importances(self, n_features: int) -> np.ndarray:
        counts = np.zeros(n_features, dtype=np.float64)

        def _traverse(node: Any) -> None:
            if isinstance(node, _SimpleDecisionTreeNode):
                counts[node.feature_idx] += 1.0
                _traverse(node.left)
                _traverse(node.right)

        for tree in self.trees:
            _traverse(tree.root)
        total = float(np.sum(counts))
        if total > 0:
            return counts / total
        return np.ones(n_features, dtype=np.float64) / max(1, n_features)


class SpatialGradientBoostingModel(BaseValuationModel):
    """Gradient boosting regressor augmented with spatial lag and proximity features."""

    FEATURE_NAMES = [
        "sqft",
        "bedrooms",
        "bathrooms",
        "age",
        "lot_size_sqft",
        "lat",
        "lon",
        "spatial_lag_price_k5",
        "spatial_lag_price_k10",
        "density_1km",
        "density_3km",
    ]

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.08,
        max_depth: int = 5,
        use_xgboost: bool = True,
    ) -> None:
        super().__init__(name="SpatialGradientBoostingModel")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self._extractor = SpatialFeatureExtractor()
        self._model: Any = None

    def _extract_features(self, prop: Property, background: Sequence[Property]) -> np.ndarray:
        struct_feats = extract_structural_features(prop)
        spatial_feats = self._extractor.extract_all(prop, background)

        return np.array(
            [
                struct_feats["sqft"],
                struct_feats["bedrooms"],
                struct_feats["bathrooms"],
                struct_feats["age"],
                struct_feats["lot_size_sqft"],
                prop.coordinates.lat,
                prop.coordinates.lon,
                spatial_feats.get("spatial_lag_price_k5", 0.0),
                spatial_feats.get("spatial_lag_price_k10", 0.0),
                spatial_feats.get("density_1km", 0.0),
                spatial_feats.get("density_3km", 0.0),
            ],
            dtype=np.float64,
        )

    def fit(self, properties: Sequence[Property]) -> SpatialGradientBoostingModel:
        """Fit spatial gradient boosting model."""
        valid_props = [p for p in properties if p.price is not None and p.price > 0]
        if len(valid_props) < 5:
            raise ValueError("Need at least 5 valid property sales to fit spatial GBM")

        self._training_properties = list(valid_props)

        X_list = [self._extract_features(p, valid_props) for p in valid_props]
        X = np.vstack(X_list)
        y = np.array([p.price for p in valid_props], dtype=np.float64)

        if self.use_xgboost and HAS_XGBOOST:
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
            )
            self._model.fit(X, y)
        else:
            self._model = _NumpyGradientBoostingRegressor(
                n_estimators=min(self.n_estimators, 50),
                learning_rate=self.learning_rate,
                max_depth=min(self.max_depth, 4),
            )
            self._model.fit(X, y)

        self.is_fitted = True
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importance array."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if self.use_xgboost and HAS_XGBOOST and hasattr(self._model, "feature_importances_"):
            return np.asarray(self._model.feature_importances_, dtype=np.float64)
        if hasattr(self._model, "get_feature_importances"):
            return self._model.get_feature_importances(len(self.FEATURE_NAMES))
        return np.ones(len(self.FEATURE_NAMES), dtype=np.float64) / len(self.FEATURE_NAMES)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dictionary mapping feature names to their relative importance."""
        importances = self.feature_importances_
        return {name: float(imp) for name, imp in zip(self.FEATURE_NAMES, importances)}

    def predict(self, properties: Sequence[Property]) -> np.ndarray:
        """Predict prices using trained spatial gradient booster."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if not properties:
            return np.array([], dtype=np.float64)

        background = self._training_properties if self._training_properties else properties
        X_list = [self._extract_features(p, background) for p in properties]
        X = np.vstack(X_list)

        preds = self._model.predict(X)
        return np.maximum(10000.0, np.asarray(preds, dtype=np.float64))

    def predict_one(self, prop: Property) -> ValuationResult:
        """Predict valuation for a single property with spatial context."""
        preds = self.predict([prop])
        estimated_val = float(preds[0])

        return ValuationResult(
            property_id=prop.id,
            estimated_value=estimated_val,
            confidence_score=0.90,
            lower_bound=estimated_val * 0.94,
            upper_bound=estimated_val * 1.06,
            comps_used=[],
            methodology="spatial_gradient_boosting",
            feature_contributions={},
        )
