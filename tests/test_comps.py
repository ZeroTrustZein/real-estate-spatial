"""Comprehensive tests for Comparable Properties (Comps) engine, scorer, and adjustments."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pytest

from real_estate_spatial.comps.adjustments import (
    CompAdjustmentCalculator,
    parse_date_string,
)
from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.comps.scorer import SimilarityScorer
from real_estate_spatial.types import CompCandidate, Coordinates, Property, ValuationResult


class TestCompAdjustmentCalculator:
    """Test line-item adjustments and adjusted comp pricing."""

    def test_sqft_adjustment(self) -> None:
        """Subject with greater sqft should produce positive adjustment on comp."""
        calc = CompAdjustmentCalculator(default_price_per_sqft=300.0, sqft_marginal_factor=0.5)
        subject = Property(
            id="SUBJ", address="A", coordinates=Coordinates(lat=0, lon=0), sqft=2000.0
        )
        comp = Property(
            id="COMP",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1600.0,
            price=600000.0,
        )

        # comp ppsqft = 600000 / 1600 = 375.0, marginal = 187.5
        # sqft_diff = +400 -> adjustment = +75000.0
        adjusted_price, adjs = calc.calculate_adjustments(subject, comp)
        assert adjs["sqft"] == pytest.approx(400.0 * (600000.0 / 1600.0) * 0.5)
        assert adjusted_price > comp.price

    def test_bedroom_and_bathroom_adjustments(self) -> None:
        """Test bed and bath value adjustments."""
        calc = CompAdjustmentCalculator(bedroom_value=20000.0, bathroom_value=12000.0)
        subject = Property(
            id="SUBJ",
            address="A",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            bedrooms=4,
            bathrooms=3.0,
        )
        comp = Property(
            id="COMP",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            bedrooms=3,
            bathrooms=2.0,
            price=500000.0,
        )

        _, adjs = calc.calculate_adjustments(subject, comp)
        assert adjs["bedrooms"] == 20000.0
        assert adjs["bathrooms"] == 12000.0

    def test_lot_size_and_year_built_adjustments(self) -> None:
        """Test lot size marginal adjustment and year built depreciation."""
        calc = CompAdjustmentCalculator(lot_sqft_marginal_factor=4.0)
        subject = Property(
            id="SUBJ",
            address="A",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            lot_size_sqft=6000.0,
            year_built=2015,
        )
        comp = Property(
            id="COMP",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            lot_size_sqft=5000.0,
            year_built=2005,
            price=500000.0,
        )

        _, adjs = calc.calculate_adjustments(subject, comp)
        assert adjs["lot_size"] == pytest.approx(1000.0 * 4.0)
        # 10 years newer @ $1,000/year = +$10,000
        assert adjs["year_built"] == pytest.approx(10000.0)

    def test_missing_lot_or_year_built(self) -> None:
        """Missing lot size or year built yields zero adjustment."""
        calc = CompAdjustmentCalculator()
        subject = Property(
            id="SUBJ",
            address="A",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            lot_size_sqft=None,
            year_built=None,
        )
        comp = Property(
            id="COMP",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            lot_size_sqft=5000.0,
            year_built=2000,
            price=500000.0,
        )

        _, adjs = calc.calculate_adjustments(subject, comp)
        assert adjs["lot_size"] == 0.0
        assert adjs["year_built"] == 0.0

    def test_time_trend_appreciation(self) -> None:
        """Test market appreciation over time and support for varied date formats."""
        calc = CompAdjustmentCalculator(annual_appreciation_rate=0.05)
        subject = Property(
            id="SUBJ", address="A", coordinates=Coordinates(lat=0, lon=0), sqft=1500.0
        )

        # Comp sold 2 years ago (730 days approx)
        comp_date = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")
        comp = Property(
            id="COMP",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            price=400000.0,
            sale_date=comp_date,
        )

        _, adjs = calc.calculate_adjustments(subject, comp)
        assert adjs["market_time"] > 0.0
        # Approximately 2 years of 5% compounded on 400k ≈ 41,000
        assert 35000.0 < adjs["market_time"] < 45000.0

        # Alternate date format: YYYY/MM/DD
        comp_slash = Property(
            id="COMP-S",
            address="C",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            price=400000.0,
            sale_date=(date.today() - timedelta(days=365)).strftime("%Y/%m/%d"),
        )
        _, adjs_slash = calc.calculate_adjustments(subject, comp_slash)
        assert adjs_slash["market_time"] > 0.0

    def test_iso_timestamp_comp_dates(self) -> None:
        """Test ISO 8601 timestamp formats (e.g. 2024-05-01T12:00:00Z)."""
        calc = CompAdjustmentCalculator(annual_appreciation_rate=0.05)
        subject = Property(
            id="SUBJ", address="A", coordinates=Coordinates(lat=0, lon=0), sqft=1500.0
        )

        comp_iso_z = Property(
            id="COMP-ISO-Z",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            price=400000.0,
            sale_date="2020-01-01T12:00:00Z",
        )
        _, adjs = calc.calculate_adjustments(
            subject, comp_iso_z, reference_date="2022-01-01T00:00:00Z"
        )
        assert adjs["market_time"] > 0.0
        # 2 years of 5% on 400k
        assert 35000.0 < adjs["market_time"] < 45000.0

        # ISO with offset
        comp_iso_offset = Property(
            id="COMP-ISO-OFF",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=1500.0,
            price=400000.0,
            sale_date="2020-01-01T08:00:00+00:00",
        )
        _, adjs_off = calc.calculate_adjustments(
            subject, comp_iso_offset, reference_date="2022-01-01"
        )
        assert adjs_off["market_time"] > 0.0


class TestSimilarityScorer:
    """Test multi-factor comparable similarity scoring."""

    def test_identical_property_similarity(self, sample_property: Property) -> None:
        """Similarity between identical properties at distance 0 should be 1.0."""
        scorer = SimilarityScorer()
        score = scorer.score(sample_property, sample_property, distance_km=0.0)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_distance_decay(self, sample_property: Property) -> None:
        """Score decreases monotonically with increasing distance."""
        scorer = SimilarityScorer(max_distance_km=10.0)

        score_0km = scorer.score(sample_property, sample_property, distance_km=0.0)
        score_2km = scorer.score(sample_property, sample_property, distance_km=2.0)
        score_5km = scorer.score(sample_property, sample_property, distance_km=5.0)
        score_15km = scorer.score(sample_property, sample_property, distance_km=15.0)

        assert 1.0 >= score_0km > score_2km > score_5km > score_15km >= 0.0

    def test_feature_penalties(self) -> None:
        """Test penalties for sqft, bedrooms, bathrooms, and age differences."""
        scorer = SimilarityScorer()
        subject = Property(
            id="SUBJ",
            address="A",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=2000.0,
            bedrooms=3,
            bathrooms=2.0,
            year_built=2000,
            property_type="single_family",
        )

        close_comp = Property(
            id="COMP-1",
            address="B",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=2050.0,
            bedrooms=3,
            bathrooms=2.0,
            year_built=2002,
            property_type="single_family",
        )

        far_comp = Property(
            id="COMP-2",
            address="C",
            coordinates=Coordinates(lat=0, lon=0),
            sqft=3500.0,
            bedrooms=5,
            bathrooms=4.0,
            year_built=1950,
            property_type="condo",  # type mismatch penalty
        )

        score_close = scorer.score(subject, close_comp, distance_km=0.1)
        score_far = scorer.score(subject, far_comp, distance_km=0.1)

        assert score_close > score_far
        assert score_close > 0.85
        assert score_far < 0.60

    def test_invalid_weights_raise_error(self) -> None:
        """Non-positive weight sums raise ValueError."""
        with pytest.raises(ValueError):
            SimilarityScorer(
                weight_distance=0.0,
                weight_sqft=0.0,
                weight_beds=0.0,
                weight_baths=0.0,
                weight_age=0.0,
            )


class TestCompsEngine:
    """Test full CompsEngine pipeline."""

    def test_fit_and_filter_invalid_price(self) -> None:
        """Fitting only keeps properties with positive prices."""
        p_valid = Property(
            id="P1", address="A", coordinates=Coordinates(lat=0, lon=0), sqft=1000.0, price=500000.0
        )
        p_zero = Property(
            id="P2", address="B", coordinates=Coordinates(lat=0, lon=0), sqft=1000.0, price=0.0
        )
        p_none = Property(
            id="P3", address="C", coordinates=Coordinates(lat=0, lon=0), sqft=1000.0, price=None
        )

        engine = CompsEngine().fit([p_valid, p_zero, p_none])
        assert len(engine._properties) == 1
        assert engine._properties[0].id == "P1"

    def test_find_comps_and_valuation(self, sample_properties: List[Property]) -> None:
        """Test comp ranking and confidence-weighted valuation."""
        engine = CompsEngine().fit(sample_properties)
        subject = Property(
            id="SUBJ",
            address="120 Market St",
            coordinates=Coordinates(lat=37.7750, lon=-122.4190),
            sqft=1850.0,
            bedrooms=3,
            bathrooms=2.0,
            year_built=1996,
        )

        comps = engine.find_comps(subject, max_comps=3, max_radius_km=5.0)
        assert 1 <= len(comps) <= 3
        # Check that candidate results are properly structured
        for c in comps:
            assert isinstance(c, CompCandidate)
            assert c.property.id != subject.id
            assert c.similarity_score > 0.0
            assert c.adjusted_price > 0.0

        # Verify sorted descending by similarity
        if len(comps) > 1:
            for i in range(len(comps) - 1):
                assert comps[i].similarity_score >= comps[i + 1].similarity_score

        # Valuation result
        val = engine.value(subject, max_comps=3)
        assert isinstance(val, ValuationResult)
        assert val.property_id == subject.id
        assert val.estimated_value > 0.0
        assert val.lower_bound <= val.estimated_value <= val.upper_bound
        assert 0.0 < val.confidence_score <= 1.0
        assert len(val.comps_used) > 0
        assert isinstance(val.feature_contributions, dict)

    def test_fallback_valuation_when_no_comps(self) -> None:
        """When no comps are within criteria, fallback valuation heuristic triggers."""
        engine = CompsEngine()  # empty engine
        subject = Property(
            id="SUBJ-REMOTE",
            address="Remote Island",
            coordinates=Coordinates(lat=10.0, lon=10.0),
            sqft=2000.0,
        )

        val = engine.value(subject)
        assert isinstance(val, ValuationResult)
        assert val.methodology == "fallback_sqft_heuristic"
        assert val.estimated_value == pytest.approx(2000.0 * 350.0)
        assert val.confidence_score == 0.1
        assert val.comps_used == []

    def test_parse_date_string_variants_and_fallbacks(self) -> None:
        """Test parse_date_string and engine._parse_date on multiple date formats and invalid inputs."""
        engine = CompsEngine()
        assert parse_date_string(None) is None
        assert parse_date_string("") is None
        assert parse_date_string("invalid_date_format") is None
        assert engine._parse_date("not-a-valid-date") is None

        # %Y-%m-%d
        d1 = parse_date_string("2024-03-15")
        assert d1 is not None and d1.year == 2024 and d1.month == 3 and d1.day == 15

        # %Y/%m/%d
        d2 = parse_date_string("2023/11/20")
        assert d2 is not None and d2.year == 2023 and d2.month == 11 and d2.day == 20

        # %Y-%m
        d3 = parse_date_string("2022-07")
        assert d3 is not None and d3.year == 2022 and d3.month == 7 and d3.day == 1

        # %Y
        d4 = parse_date_string("2021")
        assert d4 is not None and d4.year == 2021 and d4.month == 1 and d4.day == 1

    def test_similarity_scorer_auto_distance_and_negative_clamp(self) -> None:
        """Scorer automatically computes haversine distance if distance_km is None, and clamps negative distances."""
        scorer = SimilarityScorer()
        p1 = Property(
            id="P1",
            address="A",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            sqft=1500.0,
            year_built=2000,
        )
        p2 = Property(
            id="P2",
            address="B",
            coordinates=Coordinates(lat=37.7800, lon=-122.4100),
            sqft=1500.0,
            year_built=2000,
        )

        score_auto = scorer.score(p1, p2, distance_km=None)
        assert 0.0 < score_auto < 1.0

        # Negative distance should clamp to 0.0
        score_neg = scorer.score(p1, p1, distance_km=-5.0)
        assert score_neg == pytest.approx(1.0, abs=1e-5)

    def test_comps_engine_max_sale_age_days_filter(self) -> None:
        """Comps older than max_sale_age_days are filtered out."""
        engine = CompsEngine()
        old_date = (date.today() - timedelta(days=500)).strftime("%Y-%m-%d")
        recent_date = (date.today() - timedelta(days=50)).strftime("%Y-%m-%d")

        old_comp = Property(
            id="OLD",
            address="Old St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            sqft=1500.0,
            price=600000.0,
            sale_date=old_date,
        )
        recent_comp = Property(
            id="RECENT",
            address="Recent St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            sqft=1500.0,
            price=620000.0,
            sale_date=recent_date,
        )
        engine.fit([old_comp, recent_comp])

        subject = Property(
            id="SUBJ",
            address="Subj St",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194),
            sqft=1500.0,
        )
        comps = engine.find_comps(subject, max_comps=5, max_sale_age_days=180)
        assert len(comps) == 1
        assert comps[0].property.id == "RECENT"

    def test_comps_engine_zero_weights_equal_weight_fallback(self) -> None:
        """When comps have zero similarity scores, equal weight fallback is used."""
        engine = CompsEngine()
        comp1 = Property(
            id="C1",
            address="C1 St",
            coordinates=Coordinates(lat=37.77, lon=-122.41),
            sqft=1000.0,
            price=500000.0,
        )
        comp2 = Property(
            id="C2",
            address="C2 St",
            coordinates=Coordinates(lat=37.78, lon=-122.42),
            sqft=1000.0,
            price=600000.0,
        )
        engine.fit([comp1, comp2])

        subject = Property(
            id="SUBJ",
            address="Subj St",
            coordinates=Coordinates(lat=37.77, lon=-122.41),
            sqft=1000.0,
        )

        # Mock find_comps returning candidates with similarity_score = 0.0
        c_cand1 = CompCandidate(
            property=comp1,
            similarity_score=0.0,
            distance_km=0.1,
            time_diff_days=0,
            adjusted_price=500000.0,
        )
        c_cand2 = CompCandidate(
            property=comp2,
            similarity_score=0.0,
            distance_km=0.2,
            time_diff_days=0,
            adjusted_price=600000.0,
        )
        engine.find_comps = lambda *args, **kwargs: [c_cand1, c_cand2]  # type: ignore[method-assign]

        val = engine.value(subject)
        assert val.estimated_value == pytest.approx(550000.0)
