"""Tests for CLI extended commands and subsystem integration."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from real_estate_spatial.cli.main import cli
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader


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
