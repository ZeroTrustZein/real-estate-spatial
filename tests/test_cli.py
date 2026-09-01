"""Comprehensive tests for real-estate-spatial CLI commands and flags."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from real_estate_spatial.cli.main import cli


def test_cli_version(tmp_path: Path) -> None:
    runner = CliRunner()
    # Plain text
    res = runner.invoke(cli, ["version"])
    assert res.exit_code == 0
    assert "real-estate-spatial version" in res.output

    # JSON mode
    res_json = runner.invoke(cli, ["version", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["name"] == "real-estate-spatial"
    assert "version" in data


def test_cli_info() -> None:
    runner = CliRunner()
    # Plain text
    res = runner.invoke(cli, ["info"])
    assert res.exit_code == 0
    assert "Real Estate Spatial Engine Capabilities" in res.output

    # JSON mode
    res_json = runner.invoke(cli, ["info", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["package"] == "real-estate-spatial"
    assert len(data["capabilities"]) >= 5


def test_cli_generate_formats_and_json(tmp_path: Path) -> None:
    runner = CliRunner()

    # CSV output
    csv_file = tmp_path / "data.csv"
    res_csv = runner.invoke(cli, ["generate", "-n", "25", "-o", str(csv_file), "--radius-km", "8.0"])
    assert res_csv.exit_code == 0
    assert csv_file.exists()

    # GeoJSON output
    geojson_file = tmp_path / "data.geojson"
    res_geo = runner.invoke(cli, ["generate", "-n", "20", "-o", str(geojson_file)])
    assert res_geo.exit_code == 0
    assert geojson_file.exists()

    # JSON output with --json flag
    json_file = tmp_path / "data.json"
    res_json = runner.invoke(
        cli,
        ["generate", "-n", "30", "-o", str(json_file), "--lat", "37.77", "--lon", "-122.42", "--json"],
    )
    assert res_json.exit_code == 0
    assert json_file.exists()
    payload = json.loads(res_json.output)
    assert payload["status"] == "success"
    assert payload["count"] == 30


def test_cli_validate(tmp_path: Path) -> None:
    runner = CliRunner()
    data_file = tmp_path / "test_props.json"
    runner.invoke(cli, ["generate", "-n", "30", "-o", str(data_file), "--seed", "42"])

    # Normal validate
    res = runner.invoke(cli, ["validate", "-d", str(data_file)])
    assert res.exit_code == 0
    assert "Dataset Validation Summary" in res.output

    # Validate with --json and --no-outliers
    res_json = runner.invoke(
        cli, ["validate", "-d", str(data_file), "--no-outliers", "--json"]
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "total_records" in data
    assert data["total_records"] == 30
    assert data["spatial_outliers_count"] == 0

    # Missing file error handling
    res_err = runner.invoke(cli, ["validate", "-d", "non_existent_file.json", "--json"])
    assert res_err.exit_code != 0
    err_data = json.loads(res_err.output)
    assert "File not found" in err_data["error"]


def test_cli_clean(tmp_path: Path) -> None:
    runner = CliRunner()
    data_file = tmp_path / "raw.json"
    cleaned_file = tmp_path / "cleaned.json"
    runner.invoke(cli, ["generate", "-n", "30", "-o", str(data_file)])

    # Normal clean
    res = runner.invoke(
        cli,
        [
            "clean",
            "-d",
            str(data_file),
            "-o",
            str(cleaned_file),
            "--drop-invalid",
            "--drop-outliers",
            "--impute",
        ],
    )
    assert res.exit_code == 0
    assert "Dataset Cleaning Report" in res.output
    assert cleaned_file.exists()

    # Clean with --json
    cleaned_json_file = tmp_path / "cleaned_2.json"
    res_json = runner.invoke(
        cli,
        [
            "clean",
            "-d",
            str(data_file),
            "-o",
            str(cleaned_json_file),
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["status"] == "success"
    assert "report" in data
    assert data["report"]["cleaned_count"] > 0


def test_cli_comps(tmp_path: Path) -> None:
    runner = CliRunner()
    data_file = tmp_path / "props.json"
    runner.invoke(cli, ["generate", "-n", "40", "-o", str(data_file)])

    # Normal comps
    res = runner.invoke(
        cli,
        [
            "comps",
            "-d",
            str(data_file),
            "--lat",
            "37.7749",
            "--lon",
            "-122.4194",
            "--sqft",
            "2000",
            "--beds",
            "3",
            "--baths",
            "2.5",
            "--lot-size",
            "4000",
            "--year-built",
            "2000",
            "-k",
            "4",
            "-r",
            "10.0",
            "--auto-clean",
        ],
    )
    assert res.exit_code == 0
    assert "Comparable Properties" in res.output

    # Comps with --json
    res_json = runner.invoke(
        cli,
        [
            "comps",
            "-d",
            str(data_file),
            "--lat",
            "37.7749",
            "--lon",
            "-122.4194",
            "--sqft",
            "2000",
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["comps_count"] > 0
    assert len(data["comps"]) > 0


def test_cli_value(tmp_path: Path) -> None:
    runner = CliRunner()
    data_file = tmp_path / "props.json"
    runner.invoke(cli, ["generate", "-n", "40", "-o", str(data_file)])

    # Value with ensemble & explain
    res_exp = runner.invoke(
        cli,
        [
            "value",
            "-d",
            str(data_file),
            "--lat",
            "37.7749",
            "--lon",
            "-122.4194",
            "--sqft",
            "1800",
            "-m",
            "ensemble",
            "--explain",
            "--auto-clean",
        ],
    )
    assert res_exp.exit_code == 0
    assert "Property Valuation" in res_exp.output
    assert "Valuation Narrative" in res_exp.output

    # Value with --json and --explain
    res_json = runner.invoke(
        cli,
        [
            "value",
            "-d",
            str(data_file),
            "--lat",
            "37.7749",
            "--lon",
            "-122.4194",
            "--sqft",
            "1800",
            "-m",
            "auto",
            "--explain",
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "estimated_value" in data
    assert "narrative_summary" in data


def test_cli_batch_value(tmp_path: Path) -> None:
    runner = CliRunner()
    train_file = tmp_path / "train.json"
    target_file = tmp_path / "target.json"
    out_csv = tmp_path / "valuations.csv"

    runner.invoke(cli, ["generate", "-n", "30", "-o", str(train_file), "--seed", "1"])
    runner.invoke(cli, ["generate", "-n", "10", "-o", str(target_file), "--seed", "2"])

    # Batch value to CSV
    res = runner.invoke(
        cli,
        [
            "batch-value",
            "-d",
            str(train_file),
            "-i",
            str(target_file),
            "-o",
            str(out_csv),
            "-s",
            "ensemble",
            "--auto-clean",
        ],
    )
    assert res.exit_code == 0
    assert "Batch Valuation Results" in res.output
    assert out_csv.exists()

    # Batch value with --json
    res_json = runner.invoke(
        cli,
        [
            "batch-value",
            "-d",
            str(train_file),
            "-i",
            str(target_file),
            "-s",
            "hedonic",
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["count"] == 10
    assert len(data["results"]) == 10


def test_cli_train(tmp_path: Path) -> None:
    runner = CliRunner()
    data_file = tmp_path / "props.json"
    metrics_file = tmp_path / "ensemble_metrics.json"
    runner.invoke(cli, ["generate", "-n", "35", "-o", str(data_file)])

    # Train ensemble
    res = runner.invoke(
        cli,
        [
            "train",
            "-d",
            str(data_file),
            "-m",
            "ensemble",
            "-o",
            str(metrics_file),
            "--split",
            "0.8",
            "--auto-clean",
        ],
    )
    assert res.exit_code == 0
    assert "Training Metrics (ENSEMBLE)" in res.output
    assert metrics_file.exists()

    # Train with --json
    res_json = runner.invoke(
        cli,
        [
            "train",
            "-d",
            str(data_file),
            "-m",
            "gbm",
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["method"] == "gbm"
    assert "mape" in data["metrics"]


def test_cli_benchmark(tmp_path: Path) -> None:
    runner = CliRunner()
    data_file = tmp_path / "props.json"
    runner.invoke(cli, ["generate", "-n", "35", "-o", str(data_file)])

    # Benchmark plain
    res = runner.invoke(cli, ["benchmark", "-d", str(data_file), "--split", "0.75", "--auto-clean"])
    assert res.exit_code == 0
    assert "Model Benchmark Comparison" in res.output

    # Benchmark --json
    res_json = runner.invoke(cli, ["benchmark", "-d", str(data_file), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "benchmark" in data
    assert "ensemble" in data["benchmark"]
    assert "comps" in data["benchmark"]


def test_cli_benchmark_with_unpriced_test_props(tmp_path: Path) -> None:
    """Benchmark should correctly ignore unpriced properties in test split without dimension error."""
    runner = CliRunner()
    data_file = tmp_path / "props_unpriced.json"
    runner.invoke(cli, ["generate", "-n", "30", "-o", str(data_file)])

    # Inject unpriced property at end of file
    with data_file.open("r", encoding="utf-8") as f:
        props = json.load(f)
    props[-1]["price"] = None
    props[-2]["price"] = 0.0
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(props, f)

    res = runner.invoke(cli, ["benchmark", "-d", str(data_file), "--split", "0.7", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert "benchmark" in data
    assert "comps" in data["benchmark"]
