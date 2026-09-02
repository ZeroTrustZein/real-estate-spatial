"""Comprehensive end-to-end integration tests for the full Real Estate Spatial engine pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from real_estate_spatial.analytics.portfolio import PortfolioAssessor
from real_estate_spatial.analytics.submarket import SubmarketAnalyzer
from real_estate_spatial.cli.main import cli
from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.data.cleaner import DataCleaner
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader
from real_estate_spatial.data.validator import DataValidator
from real_estate_spatial.models.ensemble import EnsembleValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.spatial.index import KDTreeIndex
from real_estate_spatial.spatial.persistence import SpatialIndexSerializer
from real_estate_spatial.types import Coordinates, ValuationResult


class TestEndToEndValuationPipeline:
    """Full lifecycle testing from data generation, spatial indexing, modeling to valuation and analytics."""

    def test_full_pipeline_workflow(self, tmp_path: Path) -> None:
        # Step 1: Generate synthetic real estate dataset
        generator = SyntheticRealEstateGenerator(
            center_lat=37.7749,
            center_lon=-122.4194,
            base_price_sqft=700.0,
            seed=42,
        )
        raw_properties = generator.generate(n_samples=80)
        assert len(raw_properties) == 80

        # Step 2: Save to disk in CSV and GeoJSON formats
        csv_file = tmp_path / "raw_properties.csv"
        geojson_file = tmp_path / "raw_properties.geojson"
        PropertyDataLoader.save_csv(raw_properties, csv_file)
        PropertyDataLoader.save_geojson(raw_properties, geojson_file)
        assert csv_file.exists()
        assert geojson_file.exists()

        # Step 3: Load back and validate
        loaded_props = PropertyDataLoader.load_csv(csv_file)
        validator = DataValidator()
        val_report = validator.validate_dataset(loaded_props)
        assert val_report.is_valid is True

        # Step 4: Clean dataset
        cleaner = DataCleaner()
        cleaned_props, clean_report = cleaner.clean_dataset(loaded_props)
        assert len(cleaned_props) > 0

        # Step 5: Split train and test sets
        train_props = cleaned_props[:60]
        test_props = cleaned_props[60:]

        # Step 6: Spatial Indexing & Persistence
        kdtree = KDTreeIndex(train_props)
        assert len(kdtree) == len(train_props)
        index_file = tmp_path / "kdtree.json"
        SpatialIndexSerializer.save_index(kdtree, index_file, format="json")
        loaded_index = SpatialIndexSerializer.load_index(index_file)
        assert len(loaded_index) == len(train_props)

        # Step 7: Spatial Feature Extraction
        extractor = SpatialFeatureExtractor(
            landmarks={"CBD": Coordinates(lat=37.7749, lon=-122.4194)}
        )
        features = extractor.extract_all(test_props[0], train_props)
        assert "dist_CBD_km" in features
        assert "spatial_lag_price_k5" in features

        # Step 8: Valuation Orchestrator training
        orchestrator = ValuationOrchestrator(
            cleaner=cleaner,
            validator=validator,
            comps_engine=CompsEngine(),
            ensemble_model=EnsembleValuationModel(
                models=[
                    HedonicSpatialRegression(alpha=1.0),
                    SpatialGradientBoostingModel(n_estimators=15, max_depth=3),
                ],
                strategy="performance",
            ),
        )
        orchestrator.fit_pipeline(train_props)
        assert orchestrator.is_fitted is True

        # Step 9: Single property valuation and explanation
        subject = test_props[0]
        val_result = orchestrator.value_property(subject, strategy="auto")
        assert isinstance(val_result, ValuationResult)
        assert val_result.estimated_value > 0.0
        assert val_result.lower_bound <= val_result.estimated_value <= val_result.upper_bound

        explanation = orchestrator.explain_valuation(val_result)
        assert "estimated_value" in explanation
        assert "methodology" in explanation
        assert "narrative_summary" in explanation

        # Step 10: Batch valuation
        batch_results = orchestrator.batch_value(test_props)
        assert len(batch_results) == len(test_props)
        for res in batch_results:
            assert res.estimated_value > 0.0

        # Step 11: Submarket segmentation and market report
        analyzer = SubmarketAnalyzer(grid_cell_size_deg=0.03)
        submarket_groups = analyzer.identify_submarkets(cleaned_props, method="grid")
        assert len(submarket_groups) > 0

        report = analyzer.generate_market_report(cleaned_props)
        assert "total_properties" in report
        assert "submarkets" in report

        # Step 12: Portfolio assessment and stress testing
        assessor = PortfolioAssessor(orchestrator=orchestrator)
        val_res_list, portfolio_summary = assessor.assess_portfolio(test_props)
        assert portfolio_summary.total_properties == len(test_props)
        assert portfolio_summary.total_estimated_value > 0.0

        stress_report = assessor.stress_test_portfolio(
            test_props,
            valuation_results=val_res_list,
            shock_pct=-15.0,
            spatial_hotspots=[(Coordinates(lat=37.7749, lon=-122.4194), 2.0, -5.0)],
        )
        assert stress_report["baseline_total_value"] == pytest.approx(
            portfolio_summary.total_estimated_value, abs=1.0
        )
        assert stress_report["stressed_total_value"] < portfolio_summary.total_estimated_value
        assert stress_report["general_shock_pct"] == -15.0


class TestCLIIntegrationWorkflow:
    """Multi-command CLI integration workflow execution."""

    def test_full_cli_pipeline_run(self, tmp_path: Path) -> None:
        runner = CliRunner()

        # 1. Generate
        dataset_path = tmp_path / "cli_properties.json"
        res_gen = runner.invoke(
            cli,
            ["generate", "-n", "40", "-o", str(dataset_path), "--seed", "123", "--json"],
        )
        assert res_gen.exit_code == 0
        assert dataset_path.exists()

        # 2. Validate
        res_val = runner.invoke(cli, ["validate", "-d", str(dataset_path), "--json"])
        assert res_val.exit_code == 0
        val_data = json.loads(res_val.output)
        assert val_data["is_valid"] is True

        # 3. Clean
        cleaned_path = tmp_path / "cli_cleaned.json"
        res_clean = runner.invoke(
            cli,
            ["clean", "-d", str(dataset_path), "-o", str(cleaned_path), "--json"],
        )
        assert res_clean.exit_code == 0
        assert cleaned_path.exists()

        # 4. Train
        model_out = tmp_path / "trained_ensemble.pkl"
        res_train = runner.invoke(
            cli,
            ["train", "-d", str(cleaned_path), "-m", "ensemble", "-o", str(model_out), "--json"],
        )
        assert res_train.exit_code == 0
        assert model_out.exists()

        # 5. Value single property
        res_val_prop = runner.invoke(
            cli,
            [
                "value",
                "-d",
                str(cleaned_path),
                "--sqft",
                "1800",
                "--beds",
                "3",
                "--baths",
                "2.0",
                "--lat",
                "37.7749",
                "--lon",
                "-122.4194",
                "--json",
            ],
        )
        assert res_val_prop.exit_code == 0
        val_prop_data = json.loads(res_val_prop.output)
        assert val_prop_data["estimated_value"] > 0.0

        # 6. Benchmark models
        res_bench = runner.invoke(
            cli,
            ["benchmark", "-d", str(cleaned_path), "--split", "0.75", "--json"],
        )
        assert res_bench.exit_code == 0
        bench_data = json.loads(res_bench.output)
        assert "benchmark" in bench_data
        assert "hedonic" in bench_data["benchmark"]
