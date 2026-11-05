"""Tests for CLI extended commands and subsystem integration."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from real_estate_spatial.cli.main import cli
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader
from real_estate_spatial.types import Coordinates


@pytest.fixture
def cli_dataset(tmp_path: Path) -> Path:
    gen = SyntheticRealEstateGenerator(center_lat=37.7749, center_lon=-122.4194, seed=42)
    props = gen.generate(n_samples=25)
    data_file = tmp_path / "properties.json"
    PropertyDataLoader.save_json(props, data_file)
    return data_file


@pytest.fixture
def cli_target_dataset(tmp_path: Path) -> Path:
    gen = SyntheticRealEstateGenerator(center_lat=37.7749, center_lon=-122.4194, seed=99)
    props = gen.generate(n_samples=5)
    target_file = tmp_path / "targets.json"
    PropertyDataLoader.save_json(props, target_file)
    return target_file


class TestCLISubsystems:
    def test_version_and_info(self):
        runner = CliRunner()
        res_v = runner.invoke(cli, ["version", "--json"])
        assert res_v.exit_code == 0
        v_data = json.loads(res_v.output)
        assert v_data["name"] == "real-estate-spatial"

        res_info = runner.invoke(cli, ["info", "--json"])
        assert res_info.exit_code == 0
        info_data = json.loads(res_info.output)
        assert "capabilities" in info_data

    def test_validate_command(self, cli_dataset: Path):
        runner = CliRunner()
        res = runner.invoke(cli, ["validate", "-d", str(cli_dataset), "--json"])
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert "total_records" in data
        assert data["total_records"] == 25

        res_text = runner.invoke(cli, ["validate", "-d", str(cli_dataset)])
        assert res_text.exit_code == 0
        assert "Dataset Validation Summary" in res_text.output

    def test_clean_command(self, cli_dataset: Path, tmp_path: Path):
        runner = CliRunner()
        out_file = tmp_path / "cleaned.json"
        res = runner.invoke(cli, ["clean", "-d", str(cli_dataset), "-o", str(out_file), "--json"])
        assert res.exit_code == 0
        assert out_file.exists()
        data = json.loads(res.output)
        assert data["status"] == "success"
        assert "report" in data

    def test_batch_value_command(self, cli_dataset: Path, cli_target_dataset: Path, tmp_path: Path):
        runner = CliRunner()
        out_file = tmp_path / "valuations.csv"
        res = runner.invoke(
            cli,
            [
                "batch-value",
                "-d",
                str(cli_dataset),
                "-i",
                str(cli_target_dataset),
                "-o",
                str(out_file),
                "-s",
                "auto",
            ],
        )
        assert res.exit_code == 0
        assert out_file.exists()

        # Test JSON mode
        res_json = runner.invoke(
            cli,
            [
                "batch-value",
                "-d",
                str(cli_dataset),
                "-i",
                str(cli_target_dataset),
                "--json",
            ],
        )
        assert res_json.exit_code == 0
        data = json.loads(res_json.output)
        assert data["count"] == 5
        assert len(data["results"]) == 5

    def test_portfolio_command(self, cli_dataset: Path, cli_target_dataset: Path, tmp_path: Path):
        runner = CliRunner()
        out_file = tmp_path / "portfolio_report.json"
        res = runner.invoke(
            cli,
            [
                "portfolio",
                "-d",
                str(cli_dataset),
                "-p",
                str(cli_target_dataset),
                "-o",
                str(out_file),
                "--stress-shock",
                "-12.5",
                "--json",
            ],
        )
        assert res.exit_code == 0
        assert out_file.exists()
        data = json.loads(res.output)
        assert data["total_properties"] == 5
        assert "stress_test" in data
        assert data["stress_test"]["general_shock_pct"] == -12.5

        # Test Rich text mode
        res_text = runner.invoke(
            cli,
            [
                "portfolio",
                "-d",
                str(cli_dataset),
                "-p",
                str(cli_target_dataset),
                "--stress-shock",
                "-10.0",
            ],
        )
        assert res_text.exit_code == 0
        assert "Portfolio Assessment Summary" in res_text.output
        assert "Macroeconomic Stress Test Result" in res_text.output

    def test_submarket_command(self, cli_dataset: Path, tmp_path: Path):
        runner = CliRunner()
        out_file = tmp_path / "submarkets.json"
        res = runner.invoke(
            cli,
            [
                "submarket",
                "-d",
                str(cli_dataset),
                "-o",
                str(out_file),
                "-m",
                "grid",
                "--json",
            ],
        )
        assert res.exit_code == 0
        assert out_file.exists()
        data = json.loads(res.output)
        assert "submarkets" in data
        assert "overall_median_price" in data

        res_text = runner.invoke(cli, ["submarket", "-d", str(cli_dataset)])
        assert res_text.exit_code == 0
        assert "Market Overview & Submarkets" in res_text.output

    def test_validate_and_clean_with_issues_and_outliers(self, tmp_path: Path):
        runner = CliRunner()
        # Create dataset with invalid property and outlier
        invalid_props = [
            {
                "id": "VALID1",
                "latitude": 37.77,
                "longitude": -122.42,
                "sqft": 1500,
                "price": 800000,
            },
            {
                "id": "VALID2",
                "latitude": 37.78,
                "longitude": -122.41,
                "sqft": 1600,
                "price": 850000,
            },
            {
                "id": "VALID3",
                "latitude": 37.76,
                "longitude": -122.43,
                "sqft": 1400,
                "price": 750000,
            },
            {
                "id": "VALID4",
                "latitude": 37.775,
                "longitude": -122.425,
                "sqft": 1550,
                "price": 820000,
            },
            {"id": "INVALID", "latitude": 37.77, "longitude": -122.42, "sqft": 50, "price": 10},
            {
                "id": "OUTLIER",
                "latitude": 37.772,
                "longitude": -122.422,
                "sqft": 1500,
                "price": 10000000,
            },
        ]
        bad_file = tmp_path / "bad_props.json"
        with bad_file.open("w", encoding="utf-8") as f:
            json.dump(invalid_props, f)

        # Validate in rich text mode triggering issues table, spatial outliers table, and failed panel
        res_val = runner.invoke(cli, ["validate", "-d", str(bad_file)])
        assert res_val.exit_code == 0
        assert "Validation Issues" in res_val.output
        assert "Validation FAILED" in res_val.output

        # Clean in rich text mode triggering dropped and imputed stats
        cleaned_file = tmp_path / "cleaned_bad.json"
        res_clean = runner.invoke(
            cli,
            [
                "clean",
                "-d",
                str(bad_file),
                "-o",
                str(cleaned_file),
                "--drop-invalid",
                "--drop-outliers",
                "--impute",
            ],
        )
        assert res_clean.exit_code == 0
        assert "Dataset Cleaning Report" in res_clean.output
        assert cleaned_file.exists()

    def test_comps_empty_and_not_found(self, tmp_path: Path):
        runner = CliRunner()
        # Empty dataset
        empty_file = tmp_path / "empty.json"
        with empty_file.open("w", encoding="utf-8") as f:
            json.dump([], f)

        res_empty = runner.invoke(
            cli,
            [
                "comps",
                "-d",
                str(empty_file),
                "--lat",
                "37.77",
                "--lon",
                "-122.42",
                "--sqft",
                "1500",
            ],
        )
        assert res_empty.exit_code == 0
        assert "Dataset is empty" in res_empty.output

        res_empty_json = runner.invoke(
            cli,
            [
                "comps",
                "-d",
                str(empty_file),
                "--lat",
                "37.77",
                "--lon",
                "-122.42",
                "--sqft",
                "1500",
                "--json",
            ],
        )
        assert res_empty_json.exit_code == 0
        assert "Dataset is empty" in res_empty_json.output

        # Comps with zero radius (no comps found)
        valid_props = [
            {"id": "P1", "latitude": 37.77, "longitude": -122.42, "sqft": 1500, "price": 800000}
        ]
        props_file = tmp_path / "props.json"
        with props_file.open("w", encoding="utf-8") as f:
            json.dump(valid_props, f)

        res_not_found = runner.invoke(
            cli,
            [
                "comps",
                "-d",
                str(props_file),
                "--lat",
                "0.0",
                "--lon",
                "0.0",
                "--sqft",
                "1500",
                "-r",
                "0.001",
            ],
        )
        assert res_not_found.exit_code == 0
        assert "No comparable properties found" in res_not_found.output

    def test_batch_value_save_json_and_file_outputs(
        self, cli_dataset: Path, cli_target_dataset: Path, tmp_path: Path
    ):
        runner = CliRunner()
        out_json = tmp_path / "valuations.json"
        res = runner.invoke(
            cli,
            [
                "batch-value",
                "-d",
                str(cli_dataset),
                "-i",
                str(cli_target_dataset),
                "-o",
                str(out_json),
                "--json",
            ],
        )
        assert res.exit_code == 0
        assert out_json.exists()
        with out_json.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert len(saved_data) == 5

        # Submarket output file save in text mode
        sub_out = tmp_path / "submarket_report.json"
        res_sub = runner.invoke(cli, ["submarket", "-d", str(cli_dataset), "-o", str(sub_out)])
        assert res_sub.exit_code == 0
        assert sub_out.exists()

    def test_load_dataset_csv_and_geojson_formats(self, tmp_path: Path):
        runner = CliRunner()
        gen = SyntheticRealEstateGenerator(seed=101)
        props = gen.generate(n_samples=15)

        # CSV format
        csv_file = tmp_path / "props.csv"
        PropertyDataLoader.save_csv(props, csv_file)
        res_csv = runner.invoke(cli, ["validate", "-d", str(csv_file), "--json"])
        assert res_csv.exit_code == 0
        data_csv = json.loads(res_csv.output)
        assert data_csv["total_records"] == 15

        # GeoJSON format
        geo_file = tmp_path / "props.geojson"
        PropertyDataLoader.save_geojson(props, geo_file)
        res_geo = runner.invoke(cli, ["validate", "-d", str(geo_file), "--json"])
        assert res_geo.exit_code == 0
        data_geo = json.loads(res_geo.output)
        assert data_geo["total_records"] == 15

    def test_load_dataset_missing_and_corrupt_files(self, tmp_path: Path):
        runner = CliRunner()
        # Missing file in text mode
        res_text_missing = runner.invoke(
            cli, ["validate", "-d", str(tmp_path / "does_not_exist.json")]
        )
        assert res_text_missing.exit_code != 0
        assert "File not found" in res_text_missing.output

        # Corrupt file in text mode
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("invalid json format content {[[", encoding="utf-8")
        res_text_corrupt = runner.invoke(cli, ["validate", "-d", str(corrupt_file)])
        assert res_text_corrupt.exit_code != 0
        assert "Error loading dataset" in res_text_corrupt.output

        # Corrupt file in JSON mode
        res_json_corrupt = runner.invoke(cli, ["validate", "-d", str(corrupt_file), "--json"])
        assert res_json_corrupt.exit_code != 0
        data_corrupt = json.loads(res_json_corrupt.output)
        assert data_corrupt["status"] == "error"
        assert "Failed to load" in data_corrupt["error"]

    def test_batch_value_large_result_truncation(self, cli_dataset: Path, tmp_path: Path):
        runner = CliRunner()
        gen = SyntheticRealEstateGenerator(seed=12)
        target_props = gen.generate(n_samples=55)
        target_file = tmp_path / "targets_55.json"
        PropertyDataLoader.save_json(target_props, target_file)

        res = runner.invoke(cli, ["batch-value", "-d", str(cli_dataset), "-i", str(target_file)])
        assert res.exit_code == 0
        assert "Showing first 50 of 55 records" in res.output

    def test_train_command_split_and_save_options(self, cli_dataset: Path, tmp_path: Path):
        runner = CliRunner()
        out_metrics = tmp_path / "metrics_saved.json"

        # Split 1.0 (train_props = eval_props = props) with saved metrics in text mode
        res = runner.invoke(
            cli,
            [
                "train",
                "-d",
                str(cli_dataset),
                "-m",
                "hedonic",
                "-o",
                str(out_metrics),
                "--split",
                "1.0",
            ],
        )
        assert res.exit_code == 0
        assert out_metrics.exists()
        assert "Metrics saved to" in res.output

        # Direct invocation with invalid method raises ValueError
        from real_estate_spatial.cli.main import train_cmd

        with pytest.raises(ValueError, match="Unknown method"):
            train_cmd.callback(
                data=str(cli_dataset),
                method="invalid_model",
                output=None,
                split=0.8,
                auto_clean=False,
                json_mode=False,
            )

    def test_benchmark_small_dataset_and_edge_splits(self, tmp_path: Path):
        runner = CliRunner()
        gen = SyntheticRealEstateGenerator(seed=3)
        small_props = gen.generate(n_samples=5)
        small_file = tmp_path / "small.json"
        PropertyDataLoader.save_json(small_props, small_file)

        # < 10 properties in text mode
        res_text = runner.invoke(cli, ["benchmark", "-d", str(small_file)])
        assert res_text.exit_code == 0
        assert "Need at least 10 properties" in res_text.output

        # < 10 properties in JSON mode
        res_json = runner.invoke(cli, ["benchmark", "-d", str(small_file), "--json"])
        assert res_json.exit_code == 0
        data = json.loads(res_json.output)
        assert data["status"] == "error"
        assert "Need at least 10 properties" in data["error"]

        # High split ratio reaching n_train >= len(props)
        valid_props = gen.generate(n_samples=10)
        ten_file = tmp_path / "ten.json"
        PropertyDataLoader.save_json(valid_props, ten_file)
        res_high_split = runner.invoke(
            cli, ["benchmark", "-d", str(ten_file), "--split", "1.0", "--json"]
        )
        assert res_high_split.exit_code == 0
        assert "benchmark" in json.loads(res_high_split.output)

    def test_portfolio_text_mode_output_saving(
        self, cli_dataset: Path, cli_target_dataset: Path, tmp_path: Path
    ):
        runner = CliRunner()
        out_file = tmp_path / "portfolio_text_out.json"
        res = runner.invoke(
            cli,
            [
                "portfolio",
                "-d",
                str(cli_dataset),
                "-p",
                str(cli_target_dataset),
                "-o",
                str(out_file),
            ],
        )
        assert res.exit_code == 0
        assert out_file.exists()
        assert "Assessment report saved to" in res.output

    def test_submarket_auto_clean_and_large_clusters(self, tmp_path: Path):
        runner = CliRunner()
        gen = SyntheticRealEstateGenerator(seed=77)
        props = gen.generate(n_samples=45)
        for i, p in enumerate(props):
            p.coordinates = Coordinates(lat=37.0 + (i * 0.05), lon=-122.0 + (i * 0.05))
        scattered_file = tmp_path / "scattered.json"
        PropertyDataLoader.save_json(props, scattered_file)

        res = runner.invoke(
            cli,
            [
                "submarket",
                "-d",
                str(scattered_file),
                "--cell-size",
                "0.01",
                "--min-size",
                "1",
                "--auto-clean",
            ],
        )
        assert res.exit_code == 0
        assert "Showing first 30 of" in res.output

    def test_comps_and_value_property_type_option(self, cli_dataset: Path):
        runner = CliRunner()
        # comps with property-type condo
        res_comps = runner.invoke(
            cli,
            [
                "comps",
                "-d",
                str(cli_dataset),
                "--lat",
                "37.77",
                "--lon",
                "-122.42",
                "--sqft",
                "1200",
                "-t",
                "condo",
                "--json",
            ],
        )
        assert res_comps.exit_code == 0
        data_comps = json.loads(res_comps.output)
        assert data_comps["subject"]["property_type"] == "condo"

        # value with property-type townhouse
        res_value = runner.invoke(
            cli,
            [
                "value",
                "-d",
                str(cli_dataset),
                "--lat",
                "37.77",
                "--lon",
                "-122.42",
                "--sqft",
                "1400",
                "-t",
                "townhouse",
                "-m",
                "hedonic",
                "--json",
            ],
        )
        assert res_value.exit_code == 0
        data_val = json.loads(res_value.output)
        assert "estimated_value" in data_val

    @pytest.mark.parametrize("module", ["real_estate_spatial.cli.main", "real_estate_spatial"])
    def test_main_entry_module_execution(self, module):
        import runpy
        import sys
        import warnings

        old_argv = sys.argv
        try:
            sys.argv = ["real-estate-spatial", "--help"]
            with pytest.raises(SystemExit) as exc_info:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    runpy.run_module(module, run_name="__main__")
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv
