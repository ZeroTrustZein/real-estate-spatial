"""Command-line interface for real-estate-spatial."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel

import real_estate_spatial
from real_estate_spatial.analytics.portfolio import PortfolioAssessor
from real_estate_spatial.analytics.submarket import SubmarketAnalyzer
from real_estate_spatial.cli.formatters import (
    create_batch_valuation_table,
    create_benchmark_table,
    create_capabilities_table,
    create_cleaning_report_table,
    create_comps_table,
    create_portfolio_summary_table,
    create_portfolio_type_breakdown_table,
    create_spatial_outliers_table,
    create_stress_test_panel,
    create_submarket_table,
    create_training_metrics_table,
    create_validation_issues_table,
    create_validation_summary_table,
    create_valuation_drivers_table,
    create_valuation_panel,
)
from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.data.cleaner import DataCleaner
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader
from real_estate_spatial.data.validator import DataValidator
from real_estate_spatial.models.ensemble import EnsembleValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.types import Coordinates, Property, ValuationResult

console = Console()


def _load_dataset(filepath: Union[str, Path], json_mode: bool = False) -> List[Property]:
    """Load property dataset with automatic format detection and robust error reporting."""
    path = Path(filepath)
    if not path.exists():
        if json_mode:
            click.echo(json.dumps({"error": f"File not found: {filepath}", "status": "error"}, indent=2))
            raise click.exceptions.Exit(1)
        else:
            console.print(f"[bold red]File not found: {filepath}[/bold red]")
            raise click.Abort()

    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return PropertyDataLoader.load_csv(path)
        elif ext == ".geojson":
            return PropertyDataLoader.load_geojson(path)
        return PropertyDataLoader.load_json(path)
    except Exception as e:
        if json_mode:
            click.echo(json.dumps({"error": f"Failed to load {filepath}: {str(e)}", "status": "error"}, indent=2))
            raise click.exceptions.Exit(1)
        else:
            console.print(f"[bold red]Error loading dataset {filepath}: {str(e)}[/bold red]")
            raise click.Abort()


def _save_dataset(properties: Sequence[Property], filepath: Union[str, Path]) -> None:
    """Save properties according to file extension (.csv, .geojson, .json)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".csv":
        PropertyDataLoader.save_csv(properties, path)
    elif ext == ".geojson":
        PropertyDataLoader.save_geojson(properties, path)
    else:
        PropertyDataLoader.save_json(properties, path)


def _save_valuations(results: Sequence[ValuationResult], filepath: Union[str, Path]) -> None:
    """Save valuation results to JSON or CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".csv":
        fieldnames = [
            "property_id",
            "estimated_value",
            "confidence_score",
            "lower_bound",
            "upper_bound",
            "spread_pct",
            "methodology",
            "comps_count",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "property_id": r.property_id,
                        "estimated_value": round(r.estimated_value, 2),
                        "confidence_score": round(r.confidence_score, 4),
                        "lower_bound": round(r.lower_bound, 2),
                        "upper_bound": round(r.upper_bound, 2),
                        "spread_pct": round(r.spread_pct, 4),
                        "methodology": r.methodology,
                        "comps_count": len(r.comps_used) if r.comps_used else 0,
                    }
                )
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)


@click.group()
def cli() -> None:
    """Real Estate Spatial Valuation & Comparables Engine CLI."""
    pass


@cli.command(name="version")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def version_cmd(json_mode: bool) -> None:
    """Print package version."""
    if json_mode:
        click.echo(
            json.dumps(
                {"name": "real-estate-spatial", "version": real_estate_spatial.__version__},
                indent=2,
            )
        )
    else:
        console.print(f"[bold cyan]real-estate-spatial[/bold cyan] version [green]{real_estate_spatial.__version__}[/green]")


@cli.command(name="info")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def info_cmd(json_mode: bool) -> None:
    """Display package overview and capability info."""
    capabilities = [
        {
            "module": "Spatial Indexing",
            "components": "KDTreeIndex, GridIndex",
            "description": "Fast spherical nearest neighbor and radius spatial queries",
        },
        {
            "module": "Automated Comps",
            "components": "CompsEngine, SimilarityScorer, CompAdjustmentCalculator",
            "description": "Finds top comparable sales, calculates appraisal adjustments",
        },
        {
            "module": "Hedonic Regression",
            "components": "HedonicSpatialRegression",
            "description": "L2 Regularized spatial regression with coordinate interaction",
        },
        {
            "module": "Spatial GBM",
            "components": "SpatialGradientBoostingModel",
            "description": "Tree ensemble model with spatial lag and density features",
        },
        {
            "module": "Ensemble Valuation",
            "components": "EnsembleValuationModel, ValuationOrchestrator",
            "description": "Blended valuation combining comps, hedonic ridge, and spatial boosting",
        },
        {
            "module": "Data Engine & Validation",
            "components": "SyntheticRealEstateGenerator, PropertyDataLoader, DataValidator, DataCleaner",
            "description": "Market simulation, multi-format I/O (CSV/JSON/GeoJSON), validation and cleaning",
        },
    ]

    if json_mode:
        click.echo(
            json.dumps(
                {
                    "package": "real-estate-spatial",
                    "version": real_estate_spatial.__version__,
                    "capabilities": capabilities,
                },
                indent=2,
            )
        )
        return

    console.print(create_capabilities_table(capabilities))


@cli.command(name="generate")
@click.option("-n", "--count", default=200, type=int, help="Number of synthetic properties to generate")
@click.option("-o", "--output", default="properties.json", help="Output file path (.json, .csv, .geojson)")
@click.option("--lat", default=37.7749, type=float, help="Center latitude (e.g. 37.7749 for SF)")
@click.option("--lon", default=-122.4194, type=float, help="Center longitude (e.g. -122.4194 for SF)")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("--radius-km", "radius_km", default=10.0, type=float, help="Generation radius in km around center")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def generate_cmd(
    count: int,
    output: str,
    lat: float,
    lon: float,
    seed: int,
    radius_km: float,
    json_mode: bool,
) -> None:
    """Generate synthetic spatial real estate market data."""
    if not json_mode:
        console.print(
            f"[bold green]Generating {count} synthetic properties centered around ({lat}, {lon})...[/bold green]"
        )

    gen = SyntheticRealEstateGenerator(center_lat=lat, center_lon=lon, seed=seed)
    props = gen.generate(n_samples=count)

    _save_dataset(props, output)

    if json_mode:
        click.echo(
            json.dumps(
                {
                    "status": "success",
                    "count": len(props),
                    "output": output,
                    "center": {"lat": lat, "lon": lon},
                    "seed": seed,
                    "radius_km": radius_km,
                },
                indent=2,
            )
        )
    else:
        console.print(f"[bold green]Successfully saved {len(props)} properties to [cyan]{output}[/cyan][/bold green]")


@cli.command(name="validate")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option("--outliers/--no-outliers", default=True, help="Enable/disable spatial outlier detection")
@click.option("--k-neighbors", default=15, type=int, help="Number of k-nearest neighbors for outlier detection")
@click.option("--z-threshold", default=3.0, type=float, help="Z-score threshold for price/sqft anomalies")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def validate_cmd(
    data: str,
    outliers: bool,
    k_neighbors: int,
    z_threshold: float,
    json_mode: bool,
) -> None:
    """Validate property dataset against schema and spatial consistency rules."""
    props = _load_dataset(data, json_mode=json_mode)
    validator = DataValidator()
    val_res = validator.validate_dataset(props)

    spatial_outliers = []
    if outliers and len(props) >= 3:
        spatial_outliers = validator.detect_spatial_outliers(
            props, k_neighbors=k_neighbors, z_threshold=z_threshold
        )

    is_clean = val_res.is_valid and len(spatial_outliers) == 0

    if json_mode:
        payload = {
            "is_valid": is_clean,
            "total_records": val_res.total_records,
            "valid_records": val_res.valid_records,
            "invalid_records": val_res.invalid_records,
            "issues_count": len(val_res.issues),
            "issues": [i.to_dict() for i in val_res.issues],
            "summary": val_res.summary,
            "spatial_outliers_count": len(spatial_outliers),
            "spatial_outliers": [
                {"property_id": p.id, "z_score": round(z, 2), "reason": reason}
                for p, z, reason in spatial_outliers
            ],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    console.print(create_validation_summary_table(val_res, len(spatial_outliers), data))

    if val_res.issues:
        console.print(create_validation_issues_table(val_res.issues, max_rows=15))

    if spatial_outliers:
        console.print(create_spatial_outliers_table(spatial_outliers, max_rows=15))

    if is_clean:
        console.print(Panel("[bold green]Validation PASSED: All records meet quality and spatial criteria.[/bold green]"))
    else:
        console.print(
            Panel(
                f"[bold red]Validation FAILED: {val_res.invalid_records} invalid records, {len(spatial_outliers)} spatial outliers.[/bold red]"
            )
        )


@cli.command(name="clean")
@click.option("-d", "--data", default="properties.json", help="Input dataset file path")
@click.option("-o", "--output", default="cleaned_properties.json", help="Output cleaned dataset file path")
@click.option("--drop-invalid/--keep-invalid", default=True, help="Drop records with validation errors")
@click.option("--drop-outliers/--keep-outliers", default=False, help="Drop detected spatial price outliers")
@click.option("--impute/--no-impute", default=True, help="Impute missing year_built and lot_size values")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def clean_cmd(
    data: str,
    output: str,
    drop_invalid: bool,
    drop_outliers: bool,
    impute: bool,
    json_mode: bool,
) -> None:
    """Clean dataset by removing invalid records, outliers, and imputing missing data."""
    props = _load_dataset(data, json_mode=json_mode)
    cleaner = DataCleaner(
        drop_invalid=drop_invalid,
        drop_outliers=drop_outliers,
        impute_missing=impute,
    )
    cleaned_props, report = cleaner.clean_dataset(props)
    _save_dataset(cleaned_props, output)

    if json_mode:
        click.echo(
            json.dumps(
                {
                    "status": "success",
                    "output": output,
                    "report": report.to_dict(),
                },
                indent=2,
            )
        )
        return

    console.print(create_cleaning_report_table(report, data, output))
    console.print(f"[bold green]Saved cleaned dataset ({len(cleaned_props)} records) to [cyan]{output}[/cyan][/bold green]")


@cli.command(name="comps")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option("--lat", required=True, type=float, help="Subject latitude")
@click.option("--lon", required=True, type=float, help="Subject longitude")
@click.option("--sqft", required=True, type=float, help="Subject square footage")
@click.option("--beds", default=3, type=int, help="Subject bedrooms")
@click.option("--baths", default=2.0, type=float, help="Subject bathrooms")
@click.option("--lot-size", default=None, type=float, help="Subject lot size in sqft")
@click.option("--year-built", default=None, type=int, help="Subject year built")
@click.option("-k", "--max-comps", default=5, type=int, help="Number of comps to retrieve")
@click.option("-r", "--radius", default=5.0, type=float, help="Search radius in km")
@click.option("--auto-clean", is_flag=True, default=False, help="Automatically clean dataset before search")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def comps_cmd(
    data: str,
    lat: float,
    lon: float,
    sqft: float,
    beds: int,
    baths: float,
    lot_size: Optional[float],
    year_built: Optional[int],
    max_comps: int,
    radius: float,
    auto_clean: bool,
    json_mode: bool,
) -> None:
    """Find and display comparable properties for a subject property."""
    props = _load_dataset(data, json_mode=json_mode)
    if auto_clean:
        props, _ = DataCleaner().clean_dataset(props)

    if not props:
        if json_mode:
            click.echo(json.dumps({"error": "Dataset is empty", "comps": []}, indent=2))
        else:
            console.print("[yellow]Dataset is empty.[/yellow]")
        return

    engine = CompsEngine()
    engine.fit(props)

    subject = Property(
        id="SUBJECT",
        address="Subject Property",
        coordinates=Coordinates(lat=lat, lon=lon),
        sqft=sqft,
        bedrooms=beds,
        bathrooms=baths,
        lot_size_sqft=lot_size,
        year_built=year_built,
    )

    comps = engine.find_comps(subject, max_comps=max_comps, max_radius_km=radius)

    if json_mode:
        payload = {
            "subject": subject.to_dict(),
            "comps_count": len(comps),
            "radius_km": radius,
            "comps": [c.to_dict() for c in comps],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    if not comps:
        console.print("[yellow]No comparable properties found within given search radius.[/yellow]")
        return

    console.print(create_comps_table(comps, radius))


@cli.command(name="value")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option("--lat", required=True, type=float, help="Subject latitude")
@click.option("--lon", required=True, type=float, help="Subject longitude")
@click.option("--sqft", required=True, type=float, help="Subject square footage")
@click.option("--beds", default=3, type=int, help="Subject bedrooms")
@click.option("--baths", default=2.0, type=float, help="Subject bathrooms")
@click.option("--lot-size", default=None, type=float, help="Subject lot size in sqft")
@click.option("--year-built", default=None, type=int, help="Subject year built")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["auto", "ensemble", "comps", "hedonic", "gbm"], case_sensitive=False),
    default="auto",
    help="Valuation methodology",
)
@click.option("--auto-clean", is_flag=True, default=False, help="Automatically clean dataset before valuation")
@click.option("--explain", is_flag=True, default=False, help="Generate explainability breakdown")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def value_cmd(
    data: str,
    lat: float,
    lon: float,
    sqft: float,
    beds: int,
    baths: float,
    lot_size: Optional[float],
    year_built: Optional[int],
    method: str,
    auto_clean: bool,
    explain: bool,
    json_mode: bool,
) -> None:
    """Estimate valuation for a property using chosen methodology."""
    props = _load_dataset(data, json_mode=json_mode)

    subject = Property(
        id="SUBJECT",
        address="Subject Property",
        coordinates=Coordinates(lat=lat, lon=lon),
        sqft=sqft,
        bedrooms=beds,
        bathrooms=baths,
        lot_size_sqft=lot_size,
        year_built=year_built,
    )

    orchestrator = ValuationOrchestrator()
    orchestrator.fit_pipeline(props, auto_clean=auto_clean)
    result = orchestrator.value_property(subject, strategy=method.lower())

    if json_mode:
        if explain:
            explanation = orchestrator.explain_valuation(result)
            explanation["result"] = result.to_dict()
            click.echo(json.dumps(explanation, indent=2))
        else:
            click.echo(json.dumps(result.to_dict(), indent=2))
        return

    console.print(create_valuation_panel(result, method))

    if explain:
        exp = orchestrator.explain_valuation(result)
        console.print(Panel(exp["narrative_summary"], title="Valuation Narrative", expand=False))
        if exp["key_drivers"]:
            console.print(create_valuation_drivers_table(exp["key_drivers"]))


@cli.command(name="batch-value")
@click.option("-d", "--data", default="properties.json", help="Training dataset file path")
@click.option("-i", "--input", "input_file", required=True, help="Input properties file to value (.json, .csv, .geojson)")
@click.option("-o", "--output", default=None, help="Optional output file path for valuations (.json, .csv)")
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["auto", "ensemble", "comps", "hedonic", "gbm"], case_sensitive=False),
    default="auto",
    help="Valuation strategy",
)
@click.option("--auto-clean", is_flag=True, default=False, help="Automatically clean dataset before valuation")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def batch_value_cmd(
    data: str,
    input_file: str,
    output: Optional[str],
    strategy: str,
    auto_clean: bool,
    json_mode: bool,
) -> None:
    """Value multiple target properties in batch mode."""
    train_props = _load_dataset(data, json_mode=json_mode)
    target_props = _load_dataset(input_file, json_mode=json_mode)

    orchestrator = ValuationOrchestrator()
    orchestrator.fit_pipeline(train_props, auto_clean=auto_clean)
    results = orchestrator.batch_value(target_props, strategy=strategy.lower())

    if output:
        _save_valuations(results, output)

    if json_mode:
        payload = {
            "count": len(results),
            "strategy": strategy.lower(),
            "results": [r.to_dict() for r in results],
        }
        if output:
            payload["output"] = output
        click.echo(json.dumps(payload, indent=2))
        return

    console.print(create_batch_valuation_table(results, max_rows=50))
    if len(results) > 50:
        console.print(f"[dim]Showing first 50 of {len(results)} records.[/dim]")
    if output:
        console.print(f"[bold green]Saved {len(results)} valuations to [cyan]{output}[/cyan][/bold green]")


@cli.command(name="train")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["hedonic", "gbm", "ensemble"], case_sensitive=False),
    default="hedonic",
    help="Model type to train",
)
@click.option("-o", "--output", default=None, help="Output metrics path (.json)")
@click.option("--split", default=0.8, type=float, help="Train/test split ratio (1.0 evaluates on training set)")
@click.option("--auto-clean", is_flag=True, default=False, help="Automatically clean dataset before training")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def train_cmd(
    data: str,
    method: str,
    output: Optional[str],
    split: float,
    auto_clean: bool,
    json_mode: bool,
) -> None:
    """Train valuation model on dataset and output evaluation metrics."""
    props = _load_dataset(data, json_mode=json_mode)
    if auto_clean:
        props, _ = DataCleaner().clean_dataset(props)

    if not json_mode:
        console.print(f"[cyan]Training {method} model on {len(props)} properties...[/cyan]")

    if 0.0 < split < 1.0 and len(props) >= 10:
        n_train = max(5, int(len(props) * split))
        train_props = props[:n_train]
        eval_props = props[n_train:]
    else:
        train_props = props
        eval_props = props

    m_lower = method.lower()
    if m_lower == "hedonic":
        model = HedonicSpatialRegression().fit(train_props)
    elif m_lower == "gbm":
        model = SpatialGradientBoostingModel().fit(train_props)
    elif m_lower == "ensemble":
        model = EnsembleValuationModel().fit(train_props)
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = model.evaluate(eval_props)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

    if json_mode:
        payload = {
            "method": m_lower,
            "train_samples": len(train_props),
            "eval_samples": len(eval_props),
            "metrics": metrics.to_dict(),
        }
        if output:
            payload["output"] = output
        click.echo(json.dumps(payload, indent=2))
        return

    console.print(create_training_metrics_table(metrics, method))
    if output:
        console.print(f"[dim]Metrics saved to {output}[/dim]")


@cli.command(name="benchmark")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option("--split", default=0.8, type=float, help="Train/test split ratio")
@click.option("--auto-clean", is_flag=True, default=False, help="Automatically clean dataset before benchmarking")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def benchmark_cmd(
    data: str,
    split: float,
    auto_clean: bool,
    json_mode: bool,
) -> None:
    """Benchmark all valuation methods side-by-side with out-of-sample evaluation."""
    props = _load_dataset(data, json_mode=json_mode)
    if auto_clean:
        props, _ = DataCleaner().clean_dataset(props)

    if len(props) < 10:
        if json_mode:
            click.echo(json.dumps({"error": "Need at least 10 properties for benchmarking", "status": "error"}, indent=2))
        else:
            console.print("[red]Need at least 10 properties for benchmarking[/red]")
        return

    n_train = max(5, int(len(props) * split))
    if n_train >= len(props):
        n_train = len(props) - 1

    train_props = props[:n_train]
    test_props = props[n_train:]

    if not json_mode:
        console.print(f"[cyan]Benchmarking on {len(train_props)} train / {len(test_props)} test samples...[/cyan]")

    valid_test = [p for p in test_props if p.price is not None and p.price > 0]

    # 1. Comps Engine
    comps_engine = CompsEngine().fit(train_props)
    comps_preds = [comps_engine.value(tp).estimated_value for tp in valid_test]
    y_test = np.array([p.price for p in valid_test], dtype=np.float64)
    comps_metrics = HedonicSpatialRegression.compute_metrics(y_test, np.array(comps_preds))

    # 2. Hedonic Model
    hedonic = HedonicSpatialRegression().fit(train_props)
    hedonic_metrics = hedonic.evaluate(valid_test)

    # 3. GBM Model
    gbm = SpatialGradientBoostingModel().fit(train_props)
    gbm_metrics = gbm.evaluate(valid_test)

    # 4. Ensemble Model
    ensemble = EnsembleValuationModel().fit(train_props)
    ensemble_metrics = ensemble.evaluate(valid_test)

    if json_mode:
        payload = {
            "total_samples": len(props),
            "train_samples": len(train_props),
            "test_samples": len(test_props),
            "split": split,
            "benchmark": {
                "comps": comps_metrics.to_dict(),
                "hedonic": hedonic_metrics.to_dict(),
                "gbm": gbm_metrics.to_dict(),
                "ensemble": ensemble_metrics.to_dict(),
            },
        }
        click.echo(json.dumps(payload, indent=2))
        return

    benchmark_items = [
        ("Automated Comps", comps_metrics),
        ("Hedonic Spatial Ridge", hedonic_metrics),
        ("Spatial Gradient Boosting", gbm_metrics),
        ("Ensemble Model", ensemble_metrics),
    ]
    console.print(create_benchmark_table(benchmark_items))


@cli.command(name="portfolio")
@click.option("-d", "--data", default="properties.json", help="Training/market dataset file path")
@click.option("-p", "--portfolio", "portfolio_file", required=True, help="Portfolio properties file (.json, .csv, .geojson)")
@click.option("-o", "--output", default=None, help="Output file path for portfolio risk report (.json)")
@click.option("--stress-shock", default=0.0, type=float, help="Macroeconomic price shock percentage (e.g. -10.0 for -10%)")
@click.option("--auto-clean", is_flag=True, default=False, help="Automatically clean dataset before analysis")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def portfolio_cmd(
    data: str,
    portfolio_file: str,
    output: Optional[str],
    stress_shock: float,
    auto_clean: bool,
    json_mode: bool,
) -> None:
    """Assess real estate portfolio valuation, value dispersion, and risk metrics."""
    train_props = _load_dataset(data, json_mode=json_mode)
    port_props = _load_dataset(portfolio_file, json_mode=json_mode)

    orchestrator = ValuationOrchestrator()
    orchestrator.fit_pipeline(train_props, auto_clean=auto_clean)

    assessor = PortfolioAssessor(orchestrator=orchestrator)
    val_results, summary = assessor.assess_portfolio(port_props)

    stress_report = None
    if stress_shock != 0.0:
        stress_report = assessor.stress_test_portfolio(
            port_props, valuation_results=val_results, shock_pct=stress_shock
        )

    summary_dict = summary.to_dict()
    if stress_report:
        summary_dict["stress_test"] = stress_report

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

    if json_mode:
        click.echo(json.dumps(summary_dict, indent=2))
        return

    console.print(create_portfolio_summary_table(summary, len(port_props)))

    if summary.property_type_breakdown:
        console.print(create_portfolio_type_breakdown_table(summary.property_type_breakdown))

    if stress_report:
        console.print(create_stress_test_panel(stress_report, stress_shock))

    if output:
        console.print(f"[dim]Assessment report saved to {output}[/dim]")


@cli.command(name="submarket")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option("-o", "--output", default=None, help="Output report file path (.json)")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["grid", "spatial_clustering"], case_sensitive=False),
    default="grid",
    help="Submarket clustering methodology",
)
@click.option("--cell-size", default=0.05, type=float, help="Grid cell size in degrees (approx ~5.5km)")
@click.option("--min-size", default=2, type=int, help="Minimum property count per submarket cluster")
@click.option("--auto-clean", is_flag=True, default=False, help="Automatically clean dataset before analysis")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output in JSON format")
def submarket_cmd(
    data: str,
    output: Optional[str],
    method: str,
    cell_size: float,
    min_size: int,
    auto_clean: bool,
    json_mode: bool,
) -> None:
    """Analyze spatial submarkets, micro-market pricing trends, and market volatility."""
    props = _load_dataset(data, json_mode=json_mode)
    if auto_clean:
        props, _ = DataCleaner().clean_dataset(props)

    analyzer = SubmarketAnalyzer(grid_cell_size_deg=cell_size, min_cluster_size=min_size)
    report = analyzer.generate_market_report(props)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if json_mode:
        click.echo(json.dumps(report, indent=2))
        return

    console.print(create_submarket_table(report, max_rows=30))
    if report["submarket_count"] > 30:
        console.print(f"[dim]Showing first 30 of {report['submarket_count']} submarkets.[/dim]")
    if output:
        console.print(f"[dim]Full market report saved to {output}[/dim]")


if __name__ == "__main__":
    cli()

