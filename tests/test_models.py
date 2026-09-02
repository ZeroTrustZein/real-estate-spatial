"""Comprehensive tests for Hedonic Regression, Spatial Gradient Boosting, and Valuation Metrics."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from real_estate_spatial.models.base import (
    BaseValuationModel,
    extract_structural_features,
)
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.types import Coordinates, ModelMetrics, Property, ValuationResult


class TestModelMetrics:
    """Test valuation model evaluation metrics and error calculations."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions yield 0 error metrics, R2=1.0, and 100% hit rates."""
        y_true = np.array([500000.0, 750000.0, 1000000.0])
        y_pred = np.array([500000.0, 750000.0, 1000000.0])

        metrics = BaseValuationModel.compute_metrics(y_true, y_pred)
        assert metrics.mape == 0.0
        assert metrics.mdape == 0.0
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.r2 == pytest.approx(1.0)
        assert metrics.within_5_pct == 100.0
        assert metrics.within_10_pct == 100.0
        assert metrics.within_20_pct == 100.0

    def test_known_error_metrics(self) -> None:
        """Verify metric mathematical calculations on known error distribution."""
        # Errors: +10% (1.1x), -5% (0.95x), +20% (1.2x), 0% (1.0x)
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 360.0, 400.0])

        metrics = BaseValuationModel.compute_metrics(y_true, y_pred)
        # pct_errors = [0.10, 0.05, 0.20, 0.0]
        # mape = mean(0.10, 0.05, 0.20, 0.0) * 100 = 8.75%
        assert metrics.mape == pytest.approx(8.75)
        # mdape = median([0.0, 0.05, 0.10, 0.20]) = 0.075 * 100 = 7.5%
        assert metrics.mdape == pytest.approx(7.5)
        # mae = (10 + 10 + 60 + 0) / 4 = 20.0
        assert metrics.mae == pytest.approx(20.0)
        # rmse = sqrt((100 + 100 + 3600 + 0) / 4) = sqrt(950) ≈ 30.822
        assert metrics.rmse == pytest.approx(np.sqrt(950.0))

        # Hit rates:
        # <= 5%: 0.05 and 0.0 (2 out of 4) = 50%
        assert metrics.within_5_pct == pytest.approx(50.0)
        # <= 10%: 0.0, 0.05, 0.10 (3 out of 4) = 75%
        assert metrics.within_10_pct == pytest.approx(75.0)
        # <= 20%: all 4 = 100%
        assert metrics.within_20_pct == pytest.approx(100.0)

    def test_empty_or_zero_values(self) -> None:
        """Compute metrics handles empty arrays safely."""
        metrics = BaseValuationModel.compute_metrics(np.array([]), np.array([]))
        assert metrics.mape == 0.0
        assert metrics.r2 == 0.0

    def test_extract_structural_features_helper(self) -> None:
        """extract_structural_features extracts standardized dictionary features."""
        p = Property(
            id="P1",
            address="123 Test St",
            coordinates=Coordinates(lat=37.77, lon=-122.42),
            sqft=2000.0,
            bedrooms=4,
            bathrooms=2.5,
            lot_size_sqft=6000.0,
            year_built=2006,
        )
        feats = extract_structural_features(p, ref_year=2026)
        assert feats["sqft"] == 2000.0
        assert feats["sqft_squared"] == (2000.0 / 1000.0) ** 2
        assert feats["bedrooms"] == 4.0
        assert feats["bathrooms"] == 2.5
        assert feats["age"] == 20.0
        assert feats["lot_size_sqft"] == 6000.0


class TestHedonicSpatialRegression:
    """Test Hedonic Spatial Regression training, inference, and regularization."""

    def test_fit_and_predict(self, sample_dataset: List[Property]) -> None:
        """Test standard fit, predict, and evaluate lifecycle."""
        train = sample_dataset[:35]
        test = sample_dataset[35:]

        model = HedonicSpatialRegression(alpha=1.0)
        model.fit(train)

        assert model.is_fitted is True
        assert len(model.coefficients) == len(HedonicSpatialRegression.FEATURE_NAMES)

        preds = model.predict(test)
        assert len(preds) == len(test)
        assert np.all(preds > 0.0)

        # Single prediction
        single_res = model.predict_one(test[0])
        assert isinstance(single_res, ValuationResult)
        assert single_res.estimated_value > 0.0
        assert single_res.lower_bound <= single_res.estimated_value <= single_res.upper_bound
        assert len(single_res.feature_contributions) == len(HedonicSpatialRegression.FEATURE_NAMES)

        # Evaluate on test set
        metrics = model.evaluate(test)
        assert isinstance(metrics, ModelMetrics)
        assert metrics.mape > 0.0

    def test_feature_importances(self, sample_dataset: List[Property]) -> None:
        """Test feature importance normalization and naming."""
        model = HedonicSpatialRegression().fit(sample_dataset[:40])

        importances = model.feature_importances_
        assert len(importances) == len(HedonicSpatialRegression.FEATURE_NAMES) - 1
        assert np.sum(importances) == pytest.approx(1.0, rel=1e-5)
        assert np.all(importances >= 0.0)

        imp_dict = model.get_feature_importances()
        assert isinstance(imp_dict, dict)
        assert "sqft" in imp_dict
        assert "bedrooms" in imp_dict
        assert sum(imp_dict.values()) == pytest.approx(1.0, rel=1e-5)

    def test_ridge_regularization_shrinkage(self, sample_dataset: List[Property]) -> None:
        """Higher alpha values should shrink regression coefficient norms."""
        train = sample_dataset[:40]

        model_small_alpha = HedonicSpatialRegression(alpha=0.01).fit(train)
        model_large_alpha = HedonicSpatialRegression(alpha=1000.0).fit(train)

        # Non-intercept coefficient norm
        norm_small = np.linalg.norm(model_small_alpha.coefficients[1:])
        norm_large = np.linalg.norm(model_large_alpha.coefficients[1:])

        assert norm_small > norm_large

    def test_errors_on_unfitted_or_insufficient_data(
        self, sample_properties: List[Property]
    ) -> None:
        """Predicting before fit raises RuntimeError, fitting on <3 samples raises ValueError."""
        model = HedonicSpatialRegression()
        with pytest.raises(RuntimeError):
            model.predict(sample_properties)

        with pytest.raises(RuntimeError):
            _ = model.feature_importances_

        # Fit with only 2 properties
        with pytest.raises(ValueError):
            model.fit(sample_properties[:2])


class TestSpatialGradientBoostingModel:
    """Test Spatial Gradient Boosting Model with fallback tree ensemble and XGBoost."""

    def test_numpy_fallback_fit_and_predict(self, sample_dataset: List[Property]) -> None:
        """Test pure NumPy gradient boosting fallback."""
        train = sample_dataset[:35]
        test = sample_dataset[35:]

        model = SpatialGradientBoostingModel(
            n_estimators=15,
            learning_rate=0.1,
            max_depth=3,
            use_xgboost=False,  # force fallback
        )
        model.fit(train)

        assert model.is_fitted is True
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert np.all(preds > 0.0)

        # Feature importances
        imp_arr = model.feature_importances_
        assert len(imp_arr) == len(SpatialGradientBoostingModel.FEATURE_NAMES)
        assert np.sum(imp_arr) == pytest.approx(1.0, rel=1e-5)

        imp_dict = model.get_feature_importances()
        assert len(imp_dict) == len(SpatialGradientBoostingModel.FEATURE_NAMES)
        assert "spatial_lag_price_k5" in imp_dict

        # ValuationResult
        single_res = model.predict_one(test[0])
        assert isinstance(single_res, ValuationResult)
        assert single_res.estimated_value > 0.0

    def test_predict_empty_list(self, sample_dataset: List[Property]) -> None:
        """Predicting an empty list returns an empty numpy array."""
        model = SpatialGradientBoostingModel(n_estimators=5, use_xgboost=False).fit(
            sample_dataset[:20]
        )
        empty_preds = model.predict([])
        assert len(empty_preds) == 0

    def test_errors_on_unfitted_or_insufficient_samples(
        self, sample_properties: List[Property]
    ) -> None:
        """Predicting before fit raises RuntimeError; fitting on <5 samples raises ValueError."""
        model = SpatialGradientBoostingModel()
        with pytest.raises(RuntimeError):
            model.predict(sample_properties)

        with pytest.raises(RuntimeError):
            _ = model.feature_importances_

        # Fit with only 4 samples
        with pytest.raises(ValueError):
            model.fit(sample_properties[:4])
