"""Unit and integration tests for EnsembleValuationModel."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from real_estate_spatial.models.ensemble import EnsembleValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.types import ModelMetrics, Property, ValuationResult


class TestEnsembleValuationModel:
    """Test suite for EnsembleValuationModel."""

    def test_init_defaults_and_validation(self) -> None:
        """Verify default sub-models, strategy validation, and empty models validation."""
        ensemble = EnsembleValuationModel()
        assert len(ensemble.models) == 2
        assert isinstance(ensemble.models[0], HedonicSpatialRegression)
        assert isinstance(ensemble.models[1], SpatialGradientBoostingModel)
        assert ensemble.strategy == "performance"
        assert ensemble.is_fitted is False

        # Invalid strategy
        with pytest.raises(ValueError, match="Invalid strategy"):
            EnsembleValuationModel(strategy="invalid_strat")

        # Empty models list
        with pytest.raises(ValueError, match="at least one sub-model"):
            EnsembleValuationModel(models=[])

    def test_fit_and_predict_performance_strategy(self, sample_dataset: List[Property]) -> None:
        """Verify fitting with performance strategy and predictions."""
        train = sample_dataset[:40]
        test = sample_dataset[40:]

        ensemble = EnsembleValuationModel(strategy="performance")
        ensemble.fit(train, validation_split=0.25)

        assert ensemble.is_fitted is True
        weights = ensemble.get_model_weights()
        assert len(weights) == 2
        assert sum(weights.values()) == pytest.approx(1.0)
        for w in weights.values():
            assert 0.0 <= w <= 1.0

        metrics = ensemble.get_submodel_metrics()
        assert len(metrics) == 2
        for m_name, m_val in metrics.items():
            assert isinstance(m_val, ModelMetrics)

        preds = ensemble.predict(test)
        assert len(preds) == len(test)
        assert np.all(preds > 0.0)

        # Predict one
        one_res = ensemble.predict_one(test[0])
        assert isinstance(one_res, ValuationResult)
        assert one_res.property_id == test[0].id
        assert one_res.estimated_value > 0.0
        assert one_res.lower_bound <= one_res.estimated_value <= one_res.upper_bound
        assert 0.0 < one_res.confidence_score <= 1.0
        assert one_res.methodology == "ensemble_valuation"
        assert "submodel_HedonicSpatialRegression" in one_res.feature_contributions
        assert "submodel_SpatialGradientBoostingModel" in one_res.feature_contributions

    def test_fit_equal_strategy(self, sample_dataset: List[Property]) -> None:
        """Verify equal strategy sets equal weights to all submodels."""
        ensemble = EnsembleValuationModel(strategy="equal")
        ensemble.fit(sample_dataset[:30])

        weights = ensemble.get_model_weights()
        assert weights["HedonicSpatialRegression"] == pytest.approx(0.5)
        assert weights["SpatialGradientBoostingModel"] == pytest.approx(0.5)

    def test_fit_custom_strategy_dict_and_list(self, sample_dataset: List[Property]) -> None:
        """Verify custom weights passed as dict or list."""
        # Dict weights
        ens_dict = EnsembleValuationModel(
            weights={"HedonicSpatialRegression": 0.8, "SpatialGradientBoostingModel": 0.2},
            strategy="custom",
        )
        ens_dict.fit(sample_dataset[:30])
        w_dict = ens_dict.get_model_weights()
        assert w_dict["HedonicSpatialRegression"] == pytest.approx(0.8)
        assert w_dict["SpatialGradientBoostingModel"] == pytest.approx(0.2)

        # List weights
        ens_list = EnsembleValuationModel(
            weights=[0.3, 0.7],
            strategy="custom",
        )
        ens_list.fit(sample_dataset[:30])
        w_list = ens_list.get_model_weights()
        assert w_list["HedonicSpatialRegression"] == pytest.approx(0.3)
        assert w_list["SpatialGradientBoostingModel"] == pytest.approx(0.7)

        # Mismatched sequence length
        ens_bad = EnsembleValuationModel(weights=[0.5], strategy="custom")
        with pytest.raises(ValueError, match="Custom weights count"):
            ens_bad.fit(sample_dataset[:30])

    def test_unfitted_errors_and_edge_cases(self, sample_property: Property) -> None:
        """Verify errors on unfitted model and small sample size."""
        ensemble = EnsembleValuationModel()

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            ensemble.predict([sample_property])

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            ensemble.predict_one(sample_property)

        # Empty prediction list
        ensemble.is_fitted = True
        empty_preds = ensemble.predict([])
        assert len(empty_preds) == 0

        # Insufficient training samples (< 5)
        ensemble.is_fitted = False
        with pytest.raises(ValueError, match="Need at least 5 valid property sales"):
            ensemble.fit([sample_property, sample_property])

    def test_evaluate_metrics(self, sample_dataset: List[Property]) -> None:
        """Verify evaluate on ensemble returns valid ModelMetrics."""
        ensemble = EnsembleValuationModel().fit(sample_dataset[:35])
        metrics = ensemble.evaluate(sample_dataset[35:])
        assert isinstance(metrics, ModelMetrics)
        assert metrics.mape >= 0.0
        assert metrics.rmse >= 0.0
