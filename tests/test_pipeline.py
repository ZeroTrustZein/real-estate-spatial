"""Unit and integration tests for ValuationOrchestrator."""

from __future__ import annotations

from typing import List

import pytest

from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.types import Coordinates, Property, ValuationResult


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

    def test_orchestrator_small_training_dataset(self) -> None:
        """When training dataset < 5 samples, CompsEngine is fitted but Ensemble is not."""
        props = [
            Property(
                id=f"P_SMALL_{i}",
                address=f"{i} Small St",
                coordinates=Coordinates(37.77 + i * 0.0001, -122.42),
                price=500000.0 + i * 1000.0,
                sqft=1500.0,
                bedrooms=3,
                bathrooms=2.0,
                year_built=2000,
            )
            for i in range(4)
        ]
        orchestrator = ValuationOrchestrator()
        orchestrator.fit_pipeline(props, auto_clean=False)
        assert orchestrator.is_fitted is True
        assert len(orchestrator.comps_engine._properties) == 4
        assert orchestrator.ensemble_model.is_fitted is False

        # In auto strategy, len(comps) == 3 and avg_sim >= 0.65, but ensemble unfitted -> returns comps_res directly
        target = props[0]
        res = orchestrator.value_property(target, strategy="auto")
        assert isinstance(res, ValuationResult)
        assert res.estimated_value > 0.0
        assert "comps" in res.methodology or "comparables" in res.methodology

        # When strategy="ensemble" but ensemble is unfitted, falls back to comps
        res_ens = orchestrator.value_property(target, strategy="ensemble")
        assert isinstance(res_ens, ValuationResult)
        assert (
            "comps" in res_ens.methodology
            or "comparables" in res_ens.methodology
            or "heuristic" in res_ens.methodology
        )

        # When strategy="hedonic" but hedonic is unfitted, falls back to comps
        res_hed = orchestrator.value_property(target, strategy="hedonic")
        assert isinstance(res_hed, ValuationResult)

        # When strategy="gbm" but gbm is unfitted, falls back to comps
        res_gbm = orchestrator.value_property(target, strategy="gbm")
        assert isinstance(res_gbm, ValuationResult)

        # With only 2 properties fitted (< 3 comps), auto strategy falls back to comps_engine.value
        orch_tiny = ValuationOrchestrator()
        orch_tiny.fit_pipeline(props[:2], auto_clean=False)
        res_tiny = orch_tiny.value_property(props[0], strategy="auto")
        assert isinstance(res_tiny, ValuationResult)
        assert res_tiny.estimated_value > 0.0

    def test_orchestrator_auto_strategy_ensemble_when_no_comps(
        self, sample_dataset: List[Property]
    ) -> None:
        """Auto strategy falls back to ensemble when comps similarity is low or comps empty."""
        orchestrator = ValuationOrchestrator().fit_pipeline(sample_dataset[:40])
        remote_prop = Property(
            id="P_REMOTE",
            address="Remote Island",
            coordinates=Coordinates(lat=85.0, lon=0.0),
            sqft=1500.0,
            bedrooms=3,
            bathrooms=2.0,
            year_built=2000,
        )
        res = orchestrator.value_property(remote_prop, strategy="auto", max_radius_km=1.0)
        assert isinstance(res, ValuationResult)
        assert res.methodology == "ensemble_valuation"

        # When comps_engine has 0 properties but ensemble is fitted
        orchestrator.comps_engine._properties = []
        res_no_comps = orchestrator.value_property(remote_prop, strategy="auto")
        assert res_no_comps.methodology == "ensemble_valuation"
        assert res_no_comps.comps_used == []

    def test_orchestrator_explain_valuation_dict_and_property_comps(self) -> None:
        """explain_valuation parses comps provided as dicts or Property objects."""
        orchestrator = ValuationOrchestrator()
        comp_prop = Property(
            id="C1",
            address="100 First St",
            coordinates=Coordinates(37.77, -122.42),
            price=600000.0,
            sqft=1500.0,
        )
        dict_comp = {
            "property": {"address": "200 Second St"},
            "distance_km": 0.85,
        }
        res = ValuationResult(
            property_id="SUBJ",
            estimated_value=610000.0,
            confidence_score=0.90,
            lower_bound=580000.0,
            upper_bound=640000.0,
            comps_used=[dict_comp, comp_prop],
            methodology="test_methodology",
            feature_contributions={"sqft": 50000.0, "location": 20000.0},
        )
        explanation = orchestrator.explain_valuation(res)
        summary = explanation["comps_summary"]
        assert summary["comp_count"] == 2
        assert "200 Second St" in summary["top_comp_addresses"]
        assert "100 First St" in summary["top_comp_addresses"]
        assert 0.85 in summary["distances"]

    def test_orchestrator_spatial_cache_and_fit_reset(self, sample_dataset: List[Property]) -> None:
        """Verifies spatial cache integration and cache reset on re-fit."""
        orchestrator = ValuationOrchestrator()
        orchestrator.spatial_cache.set_radius(
            sample_dataset[0].coordinates, 10.0, sample_dataset[:5]
        )
        assert (
            orchestrator.spatial_cache.get_radius(sample_dataset[0].coordinates, 10.0) is not None
        )

        orchestrator.fit_pipeline(sample_dataset[:20])
        assert orchestrator.spatial_cache.get_radius(sample_dataset[0].coordinates, 10.0) is None
