"""Unit and integration tests for ValuationOrchestrator."""

from __future__ import annotations

from typing import List

import pytest

from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.types import Property, ValuationResult


class TestValuationOrchestrator:
    """Test suite for end-to-end ValuationOrchestrator pipeline."""

    def test_pipeline_fit_and_cleaning(self, sample_dataset: List[Property]) -> None:
        """Verify pipeline initialization, data cleaning, and model fitting."""
        orchestrator = ValuationOrchestrator()
        assert orchestrator.is_fitted is False

        # Add duplicate and invalid property to test auto_clean
        dirty_data = list(sample_dataset)
        dirty_data.append(sample_dataset[0])  # duplicate

        orchestrator.fit_pipeline(dirty_data, auto_clean=True)
        assert orchestrator.is_fitted is True
        assert orchestrator.cleaning_report is not None
        assert orchestrator.cleaning_report.initial_count == len(dirty_data)
        assert orchestrator.cleaning_report.cleaned_count == len(sample_dataset)
        assert orchestrator.comps_engine is not None
        assert orchestrator.ensemble_model.is_fitted is True

    def test_value_property_auto_strategy(self, sample_dataset: List[Property]) -> None:
        """Verify auto valuation strategy selecting blended comps/ensemble or ensemble."""
        orchestrator = ValuationOrchestrator()
        orchestrator.fit_pipeline(sample_dataset[:40])

        target = sample_dataset[45]
        res = orchestrator.value_property(target, strategy="auto")

        assert isinstance(res, ValuationResult)
        assert res.property_id == target.id
        assert res.estimated_value > 0.0
        assert res.lower_bound <= res.estimated_value <= res.upper_bound
        assert 0.0 < res.confidence_score <= 1.0
        assert res.methodology in (
            "auto_blended_comps_ensemble",
            "ensemble_valuation",
            "spatial_comparables_weighted_average",
        )

    def test_value_property_explicit_strategies(self, sample_dataset: List[Property]) -> None:
        """Verify explicit valuation strategies (ensemble, comps, hedonic, gbm, heuristic)."""
        orchestrator = ValuationOrchestrator().fit_pipeline(sample_dataset[:40])
        target = sample_dataset[42]

        # Ensemble strategy
        res_ens = orchestrator.value_property(target, strategy="ensemble")
        assert res_ens.methodology == "ensemble_valuation"
        assert res_ens.estimated_value > 0.0

        # Comps strategy
        res_comps = orchestrator.value_property(target, strategy="comps")
        assert (
            "comps" in res_comps.methodology
            or "comparables" in res_comps.methodology
            or "heuristic" in res_comps.methodology
        )
        assert res_comps.estimated_value > 0.0

        # Hedonic strategy
        res_hedonic = orchestrator.value_property(target, strategy="hedonic")
        assert res_hedonic.estimated_value > 0.0

        # GBM strategy
        res_gbm = orchestrator.value_property(target, strategy="gbm")
        assert res_gbm.estimated_value > 0.0

        # Heuristic strategy
        res_heur = orchestrator.value_property(target, strategy="heuristic")
        assert "heuristic" in res_heur.methodology
        assert res_heur.estimated_value > 0.0

        # Invalid strategy
        with pytest.raises(ValueError, match="Unknown valuation strategy"):
            orchestrator.value_property(target, strategy="non_existent")

    def test_unfitted_orchestrator_fallback(self, sample_property: Property) -> None:
        """Verify graceful fallback when orchestrator is not fitted."""
        orchestrator = ValuationOrchestrator()
        res = orchestrator.value_property(sample_property, strategy="auto")
        assert isinstance(res, ValuationResult)
        assert res.estimated_value > 0.0
        assert "heuristic" in res.methodology or "fallback" in res.methodology

    def test_batch_value(self, sample_dataset: List[Property]) -> None:
        """Verify batch valuation processing."""
        orchestrator = ValuationOrchestrator().fit_pipeline(sample_dataset[:35])
        targets = sample_dataset[35:45]

        results = orchestrator.batch_value(targets, strategy="auto")
        assert len(results) == len(targets)
        for r, t in zip(results, targets):
            assert isinstance(r, ValuationResult)
            assert r.property_id == t.id
            assert r.estimated_value > 0.0

    def test_explain_valuation(self, sample_dataset: List[Property]) -> None:
        """Verify structured explainability dictionary contents."""
        orchestrator = ValuationOrchestrator().fit_pipeline(sample_dataset[:40])
        target = sample_dataset[40]

        res = orchestrator.value_property(target, strategy="auto")
        explanation = orchestrator.explain_valuation(res)

        assert explanation["property_id"] == target.id
        assert explanation["estimated_value"] == res.estimated_value
        assert explanation["confidence_score"] == res.confidence_score
        assert explanation["methodology"] == res.methodology

        # Value range
        vr = explanation["value_range"]
        assert vr["lower_bound"] == res.lower_bound
        assert vr["upper_bound"] == res.upper_bound
        assert vr["spread_pct"] == pytest.approx(res.spread_pct)

        # Comps summary
        cs = explanation["comps_summary"]
        assert "number_of_comps" in cs
        assert "top_comp_addresses" in cs
        assert "distances" in cs
        assert "avg_distance_km" in cs

        # Drivers & narrative
        assert "key_drivers" in explanation
        assert isinstance(explanation["narrative_summary"], str)
        assert len(explanation["narrative_summary"]) > 10
        assert f"${res.estimated_value:,.2f}" in explanation["narrative_summary"]
