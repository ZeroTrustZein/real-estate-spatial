"""Edge case and stress tests for Hedonic Regression, Spatial GBM, Ensemble, and Comps Engine."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.comps.scorer import SimilarityScorer
from real_estate_spatial.models.base import BaseValuationModel
from real_estate_spatial.models.ensemble import EnsembleValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.types import Coordinates, Property, ValuationResult


class TestModelMetricsEdgeCases:
    """Test mathematical robustness of compute_metrics under edge scenarios."""

    def test_single_element_metrics(self) -> None:
        """Single observation metrics computation."""
        y_true = np.array([500000.0])
        y_pred = np.array([550000.0])

        metrics = BaseValuationModel.compute_metrics(y_true, y_pred)
        assert metrics.mape == pytest.approx(10.0)
        assert metrics.mdape == pytest.approx(10.0)
        assert metrics.mae == pytest.approx(50000.0)
        assert metrics.rmse == pytest.approx(50000.0)
        assert metrics.within_10_pct == 100.0
        assert metrics.within_5_pct == 0.0

    def test_large_divergence_metrics(self) -> None:
        """Handling large prediction divergence without overflow."""
        y_true = np.array([100000.0, 200000.0])
        y_pred = np.array([1000000.0, 2000000.0])  # 10x error

        metrics = BaseValuationModel.compute_metrics(y_true, y_pred)
        assert metrics.mape == pytest.approx(900.0)
        assert metrics.within_20_pct == 0.0
        assert metrics.r2 < 0.0  # R2 is negative when predictions are worse than mean


class TestHedonicRegressionEdgeCases:
    """Edge cases for Hedonic Spatial Regression."""

    def test_high_alpha_regularization(self, sample_dataset: List[Property]) -> None:
        """Higher alpha shrinks regression coefficients towards zero."""
        model_low = HedonicSpatialRegression(alpha=0.01).fit(sample_dataset)
        model_high = HedonicSpatialRegression(alpha=10000.0).fit(sample_dataset)

        norm_low = np.linalg.norm(model_low.coefficients)
        norm_high = np.linalg.norm(model_high.coefficients)
        assert norm_high < norm_low

    def test_predict_with_missing_optional_attributes(self, sample_dataset: List[Property]) -> None:
        """Prediction gracefully handles test property with None lot_size and year_built."""
        model = HedonicSpatialRegression().fit(sample_dataset)
        sparse_prop = Property(
            id="SPARSE-01",
            address="Unknown St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            sqft=1500.0,
            bedrooms=2,
            bathrooms=1.5,
            lot_size_sqft=None,
            year_built=None,
        )

        res = model.predict_one(sparse_prop)
        assert isinstance(res, ValuationResult)
        assert res.estimated_value > 0.0
        assert res.lower_bound <= res.estimated_value <= res.upper_bound


class TestSpatialGBMEdgeCases:
    """Edge cases for Spatial Gradient Boosting Model."""

    def test_hyperparameter_tuning_depth_and_estimators(
        self, sample_dataset: List[Property]
    ) -> None:
        """Fitting with varying tree depths and estimator counts."""
        train = sample_dataset[:35]
        test = sample_dataset[35:]

        gbm_shallow = SpatialGradientBoostingModel(n_estimators=10, max_depth=2, learning_rate=0.1)
        gbm_shallow.fit(train)
        preds_shallow = gbm_shallow.predict(test)
        assert len(preds_shallow) == len(test)
        assert np.all(preds_shallow > 0.0)

        importances = gbm_shallow.feature_importances_
        assert len(importances) > 0
        assert pytest.approx(float(np.sum(importances)), rel=1e-3) == 1.0

    def test_unfitted_raises_runtime_error(self, sample_property: Property) -> None:
        """Calling predict before fit raises RuntimeError."""
        model = SpatialGradientBoostingModel()
        with pytest.raises(RuntimeError, match="Model is not fitted"):
            model.predict([sample_property])


class TestEnsembleValuationModelEdgeCases:
    """Edge cases for EnsembleValuationModel combination strategies."""

    def test_custom_weights_normalization(self, sample_dataset: List[Property]) -> None:
        """Custom weights that do not sum to 1.0 are automatically normalized."""
        hedonic = HedonicSpatialRegression().fit(sample_dataset)
        gbm = SpatialGradientBoostingModel().fit(sample_dataset)
        model = EnsembleValuationModel(
            models=[hedonic, gbm],
            strategy="custom",
            weights=[0.4, 0.6],
        )
        model.fit(sample_dataset)

        test_prop = sample_dataset[0]
        res = model.predict_one(test_prop)
        assert res.estimated_value > 0.0
        assert res.confidence_score >= 0.0
        assert res.methodology == "ensemble_valuation"

    def test_custom_weight_strategies(self, sample_dataset: List[Property]) -> None:
        """Ensemble weighting strategy modes: performance, equal, custom."""
        hedonic = HedonicSpatialRegression().fit(sample_dataset)
        gbm = SpatialGradientBoostingModel().fit(sample_dataset)
        for strat in ("performance", "equal"):
            ens = EnsembleValuationModel(models=[hedonic, gbm], strategy=strat)
            ens.fit(sample_dataset)
            pred = ens.predict_one(sample_dataset[0])
            assert pred.estimated_value > 0.0
            assert pred.methodology == "ensemble_valuation"


class TestCompsEngineEdgeCases:
    """Edge cases for CompsEngine filtering, scoring, and adjustment calculation."""

    def test_strict_filters_trigger_fallback(self, sample_dataset: List[Property]) -> None:
        """When strict filters yield 0 comps, engine falls back gracefully."""
        engine = CompsEngine()
        engine.fit(sample_dataset)

        subject = Property(
            id="SUBJ-01",
            address="Remote Island",
            coordinates=Coordinates(lat=0.0, lon=0.0),  # Very far from SF
            sqft=2000.0,
            bedrooms=3,
            bathrooms=2.0,
        )

        # Min similarity 0.99 on distant property returns 0 comps
        comps = engine.find_comps(subject, min_similarity=0.99, max_radius_km=0.1)
        assert len(comps) == 0

        # Valuation should still produce fallback estimate
        val_result = engine.value(subject, max_radius_km=0.1)
        assert isinstance(val_result, ValuationResult)
        assert val_result.estimated_value > 0.0

    def test_similarity_scorer_weights_validation(self, sample_property: Property) -> None:
        """Similarity scorer validates weights and computes distance decay."""
        scorer = SimilarityScorer(max_distance_km=5.0)
        p1 = sample_property
        p2 = Property(
            id="PROP-NEAR",
            address="102 Market St",
            coordinates=Coordinates(lat=37.7750, lon=-122.4193),
            price=1260000.0,
            sqft=1860.0,
            bedrooms=3,
            bathrooms=2.0,
        )
        score = scorer.score(p1, p2, distance_km=0.05)
        assert 0.8 <= score <= 1.0
