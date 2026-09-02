"""Tests for portfolio assessment and submarket spatial analytics."""

import math

import pytest

from real_estate_spatial.analytics.portfolio import (
    PortfolioAssessor,
)
from real_estate_spatial.analytics.submarket import (
    SubmarketAnalyzer,
    SubmarketMetrics,
)
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.types import Coordinates, Property, ValuationResult


@pytest.fixture
def sample_portfolio() -> list[Property]:
    gen = SyntheticRealEstateGenerator(center_lat=37.7749, center_lon=-122.4194, seed=42)
    return gen.generate(n_samples=30)


class TestPortfolioAssessor:
    def test_assess_empty_portfolio(self):
        assessor = PortfolioAssessor()
        results, summary = assessor.assess_portfolio([])
        assert len(results) == 0
        assert summary.total_properties == 0
        assert summary.total_estimated_value == 0.0

    def test_assess_portfolio_with_precomputed_results(self, sample_portfolio):
        assessor = PortfolioAssessor()
        fake_results = [
            ValuationResult(
                property_id=p.id,
                estimated_value=500000.0 + i * 10000.0,
                confidence_score=0.85 if i % 5 != 0 else 0.4,
                lower_bound=450000.0,
                upper_bound=550000.0,
                comps_used=[],
                methodology="mock",
            )
            for i, p in enumerate(sample_portfolio)
        ]

        results, summary = assessor.assess_portfolio(
            sample_portfolio,
            valuation_results=fake_results,
            risk_confidence_threshold=0.6,
        )

        assert len(results) == len(sample_portfolio)
        assert summary.total_properties == len(sample_portfolio)
        assert summary.total_estimated_value > 0
        assert summary.high_risk_count > 0
        assert len(summary.high_risk_property_ids) == summary.high_risk_count
        assert summary.portfolio_spread_pct > 0
        assert (
            "single_family" in summary.property_type_breakdown
            or len(summary.property_type_breakdown) > 0
        )

        summary_dict = summary.to_dict()
        assert "total_estimated_value" in summary_dict
        assert "high_risk_ratio" in summary_dict

    def test_assess_portfolio_with_live_orchestrator(self, sample_portfolio):
        train_data = sample_portfolio[:20]
        test_data = sample_portfolio[20:]

        orchestrator = ValuationOrchestrator()
        orchestrator.fit_pipeline(train_data)

        assessor = PortfolioAssessor(orchestrator=orchestrator)
        results, summary = assessor.assess_portfolio(test_data)

        assert len(results) == len(test_data)
        assert summary.total_properties == len(test_data)
        assert summary.total_estimated_value > 0
        assert summary.average_confidence > 0

    def test_stress_test_portfolio(self, sample_portfolio):
        assessor = PortfolioAssessor()
        fake_results = [
            ValuationResult(
                property_id=p.id,
                estimated_value=1_000_000.0,
                confidence_score=0.9,
                lower_bound=950000.0,
                upper_bound=1050000.0,
                comps_used=[],
                methodology="mock",
            )
            for p in sample_portfolio
        ]

        stress_res = assessor.stress_test_portfolio(
            sample_portfolio,
            valuation_results=fake_results,
            shock_pct=-15.0,
            spatial_hotspots=[(Coordinates(37.7749, -122.4194), 5.0, -10.0)],
        )

        assert stress_res["general_shock_pct"] == -15.0
        assert stress_res["stressed_total_value"] < stress_res["baseline_total_value"]
        assert stress_res["percentage_change"] < -15.0  # further lowered by spatial hotspot


class TestSubmarketAnalyzer:
    def test_empty_submarkets(self):
        analyzer = SubmarketAnalyzer()
        assert analyzer.identify_submarkets([]) == []
        rep = analyzer.generate_market_report([])
        assert rep["total_properties"] == 0
        assert rep["submarket_count"] == 0

    def test_identify_submarkets_grid(self, sample_portfolio):
        analyzer = SubmarketAnalyzer(grid_cell_size_deg=0.03, min_cluster_size=2)
        submarkets = analyzer.identify_submarkets(sample_portfolio, method="grid")

        assert len(submarkets) > 0
        sm = submarkets[0]
        assert isinstance(sm, SubmarketMetrics)
        assert sm.property_count >= 2
        assert sm.median_price > 0
        assert sm.median_price_per_sqft > 0
        assert sm.price_volatility_pct >= 0

        sm_dict = sm.to_dict()
        assert "submarket_id" in sm_dict
        assert "median_price" in sm_dict

    def test_identify_submarkets_spatial_clustering(self, sample_portfolio):
        analyzer = SubmarketAnalyzer(grid_cell_size_deg=0.05, min_cluster_size=2)
        submarkets = analyzer.identify_submarkets(sample_portfolio, method="spatial_clustering")
        assert len(submarkets) > 0

    def test_generate_market_report(self, sample_portfolio):
        analyzer = SubmarketAnalyzer(grid_cell_size_deg=0.05, min_cluster_size=2)
        report = analyzer.generate_market_report(sample_portfolio)

        assert report["total_properties"] == len(sample_portfolio)
        assert report["priced_properties"] == len(sample_portfolio)
        assert report["overall_median_price"] > 0
        assert report["overall_median_price_per_sqft"] > 0
        assert "submarkets" in report
        assert "top_expensive_submarkets" in report
        assert "most_volatile_submarkets" in report

    def test_zero_sqft_submarket_metrics(self):
        """Zero-sqft properties should not cause TypeError or NaN in price_per_sqft metrics."""
        props = [
            Property(
                id="P1",
                address="1 Main St",
                coordinates=Coordinates(lat=37.77, lon=-122.41),
                price=500000.0,
                sqft=0.0,
            ),
            Property(
                id="P2",
                address="2 Main St",
                coordinates=Coordinates(lat=37.771, lon=-122.411),
                price=600000.0,
                sqft=1000.0,
            ),
            Property(
                id="P3",
                address="3 Main St",
                coordinates=Coordinates(lat=37.772, lon=-122.412),
                price=700000.0,
                sqft=0.0,
            ),
        ]
        analyzer = SubmarketAnalyzer(grid_cell_size_deg=0.05, min_cluster_size=2)
        submarkets = analyzer.identify_submarkets(props, method="grid")
        assert len(submarkets) == 1
        sm = submarkets[0]
        assert sm.median_price_per_sqft == 600.0
        assert sm.mean_price_per_sqft == 600.0
        assert not math.isnan(sm.median_price_per_sqft)

        report = analyzer.generate_market_report(props)
        assert report["overall_median_price_per_sqft"] == 600.0
        assert report["overall_mean_price_per_sqft"] == 600.0
