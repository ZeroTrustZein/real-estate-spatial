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
from rich.table import Table
from rich.text import Text

import real_estate_spatial
from real_estate_spatial.analytics.portfolio import PortfolioAssessor
from real_estate_spatial.analytics.submarket import SubmarketAnalyzer
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

    table = Table(title="Real Estate Spatial Engine Capabilities", show_header=True)
    table.add_column("Module", style="cyan")
    table.add_column("Components", style="white")
    table.add_column("Description", style="dim")

    for cap in capabilities:
        table.add_row(cap["module"], cap["components"], cap["description"])

    console.print(table)


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

    table = Table(title=f"Dataset Validation Summary ({data})", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    table.add_row("Total Records", str(val_res.total_records))
    table.add_row("Valid Records", f"[green]{val_res.valid_records}[/green]")
    table.add_row("Invalid Records", f"[red]{val_res.invalid_records}[/red]" if val_res.invalid_records else "0")
    table.add_row("Total Issues", str(len(val_res.issues)))
    table.add_row("Spatial Outliers", str(len(spatial_outliers)))

    console.print(table)

    if val_res.issues:
        issue_table = Table(title="Validation Issues (Sample)", show_header=True)
        issue_table.add_column("Property ID", style="cyan")
        issue_table.add_column("Field", style="yellow")
        issue_table.add_column("Rule", style="dim")
        issue_table.add_column("Severity", justify="center")
        issue_table.add_column("Message")

        for issue in val_res.issues[:15]:
            sev_style = "[red]ERROR[/red]" if issue.severity == "error" else "[yellow]WARN[/yellow]"
            issue_table.add_row(issue.property_id, issue.field, issue.rule, sev_style, issue.message)

        console.print(issue_table)

    if spatial_outliers:
        outlier_table = Table(title="Spatial Outliers Detected", show_header=True)
        outlier_table.add_column("Property ID", style="cyan")
        outlier_table.add_column("Z-Score", justify="right", style="magenta")
        outlier_table.add_column("Reason", style="dim")
        for p, z, reason in spatial_outliers[:15]:
            outlier_table.add_row(p.id, f"{z:.2f}", reason)
        console.print(outlier_table)

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

    table = Table(title=f"Dataset Cleaning Report ({data} -> {output})", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="white")

    table.add_row("Initial Records", str(report.initial_count))
    table.add_row("Cleaned Records", f"[bold green]{report.cleaned_count}[/bold green]")
    table.add_row("Dropped Records", f"[bold red]{report.dropped_count}[/bold red]" if report.dropped_count else "0")

    for reason, cnt in report.dropped_reasons.items():
        table.add_row(f"  Dropped: {reason}", str(cnt))

    for field_name, cnt in report.imputed_fields.items():
        table.add_row(f"  Imputed: {field_name}", f"[yellow]{cnt}[/yellow]")

    console.print(table)
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

    table = Table(title=f"Top {len(comps)} Comparable Properties (Radius: {radius} km)")
    table.add_column("Rank", justify="center", style="dim")
    table.add_column("Comp ID", style="cyan")
    table.add_column("Distance", justify="right")
    table.add_column("Similarity", justify="right", style="green")
    table.add_column("Specs (Sqft/Bed/Bath)", justify="center")
    table.add_column("Sale Price", justify="right")
    table.add_column("Adjusted Price", justify="right", style="bold yellow")

    for i, c in enumerate(comps, 1):
        p = c.property
        specs = f"{int(p.sqft)} sqft | {p.bedrooms}b/{p.bathrooms}ba"
        sale_str = f"${p.price:,.0f}" if p.price else "N/A"
        adj_str = f"${c.adjusted_price:,.0f}"
        table.add_row(
            str(i),
            p.id,
            f"{c.distance_km:.2f} km",
            f"{c.similarity_score*100:.1f}%",
            specs,
            sale_str,
            adj_str,
        )

    console.print(table)


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

    panel_content = (
        f"[bold cyan]Estimated Value:[/bold cyan] [bold green]${result.estimated_value:,.2f}[/bold green]\n"
        f"[bold]Confidence Score:[/bold] {result.confidence_score * 100:.1f}%\n"
        f"[bold]Valuation Range:[/bold] ${result.lower_bound:,.2f} - ${result.upper_bound:,.2f} "
        f"([dim]spread: {result.spread_pct * 100:.1f}%[/dim])\n"
        f"[bold]Methodology:[/bold] {result.methodology}\n"
        f"[bold]Comps Used:[/bold] {len(result.comps_used) if result.comps_used else 0}"
    )

    console.print(
        Panel(panel_content, title=f"Property Valuation ({method.upper()})", expand=False)
    )

    if explain:
        exp = orchestrator.explain_valuation(result)
        console.print(Panel(exp["narrative_summary"], title="Valuation Narrative", expand=False))
        if exp["key_drivers"]:
            d_table = Table(title="Top Valuation Drivers", show_header=True)
            d_table.add_column("Feature / Component", style="cyan")
            d_table.add_column("Impact / Weight", justify="right", style="bold green")
            for k, v in exp["key_drivers"].items():
                val_fmt = f"${v:,.2f}" if abs(v) > 100 else f"{v:+.2f}"
                d_table.add_row(k, val_fmt)
            console.print(d_table)


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

    table = Table(title=f"Batch Valuation Results ({len(results)} properties)", show_header=True)
    table.add_column("Property ID", style="cyan")
    table.add_column("Estimated Value", justify="right", style="bold green")
    table.add_column("Confidence", justify="right")
    table.add_column("Valuation Range", justify="right", style="dim")
    table.add_column("Methodology", style="white")

    for r in results[:50]:
        val_str = f"${r.estimated_value:,.2f}"
        conf_str = f"{r.confidence_score * 100:.1f}%"
        range_str = f"${r.lower_bound:,.0f} - ${r.upper_bound:,.0f}"
        table.add_row(r.property_id, val_str, conf_str, range_str, r.methodology)

    console.print(table)
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

    table = Table(title=f"Training Metrics ({method.upper()})")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold green")

    table.add_row("MAPE (%)", f"{metrics.mape:.2f}%")
    table.add_row("MdAPE (%)", f"{metrics.mdape:.2f}%")
    table.add_row("MAE ($)", f"${metrics.mae:,.2f}")
    table.add_row("RMSE ($)", f"${metrics.rmse:,.2f}")
    table.add_row("R² Score", f"{metrics.r2:.4f}")
    table.add_row("Within 5% of True Price", f"{metrics.within_5_pct:.1f}%")
    table.add_row("Within 10% of True Price", f"{metrics.within_10_pct:.1f}%")
    table.add_row("Within 20% of True Price", f"{metrics.within_20_pct:.1f}%")

    console.print(table)
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

    table = Table(title="Model Benchmark Comparison (Out-of-Sample)")
    table.add_column("Model / Method", style="cyan")
    table.add_column("MAPE (%)", justify="right")
    table.add_column("MdAPE (%)", justify="right")
    table.add_column("MAE ($)", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Within 10%", justify="right", style="green")

    for name, m in [
        ("Automated Comps", comps_metrics),
        ("Hedonic Spatial Ridge", hedonic_metrics),
        ("Spatial Gradient Boosting", gbm_metrics),
        ("Ensemble Model", ensemble_metrics),
    ]:
        table.add_row(
            name,
            f"{m.mape:.2f}%",
            f"{m.mdape:.2f}%",
            f"${m.mae:,.0f}",
            f"{m.r2:.3f}",
            f"{m.within_10_pct:.1f}%",
        )

    console.print(table)


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

    table = Table(title=f"Portfolio Assessment Summary ({len(port_props)} properties)", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold green")

    table.add_row("Total Properties", str(summary.total_properties))
    table.add_row("Total Estimated Value", f"${summary.total_estimated_value:,.2f}")
    table.add_row("Mean Property Value", f"${summary.mean_estimated_value:,.2f}")
    table.add_row("Median Property Value", f"${summary.median_estimated_value:,.2f}")
    table.add_row("Average Confidence", f"{summary.average_confidence * 100:.1f}%")
    table.add_row("High-Risk Count", f"[red]{summary.high_risk_count}[/red]" if summary.high_risk_count else "0")
    table.add_row("Valuation Range", f"${summary.total_lower_bound:,.0f} - ${summary.total_upper_bound:,.0f}")
    table.add_row("Portfolio Spread", f"{summary.portfolio_spread_pct:.1f}%")

    console.print(table)

    if summary.property_type_breakdown:
        type_table = Table(title="Portfolio Breakdown by Property Type", show_header=True)
        type_table.add_column("Property Type", style="cyan")
        type_table.add_column("Count", justify="right")
        type_table.add_column("Total Value", justify="right", style="green")
        type_table.add_column("Mean Value", justify="right")
        type_table.add_column("Mean Sqft", justify="right")

        for ptype, stats in summary.property_type_breakdown.items():
            type_table.add_row(
                ptype,
                str(stats["count"]),
                f"${stats['total_value']:,.0f}",
                f"${stats['mean_value']:,.0f}",
                f"{stats['mean_sqft']:,.0f}",
            )
        console.print(type_table)

    if stress_report:
        stress_panel = (
            f"[bold]Stress Shock:[/bold] {stress_shock:+.1f}%\n"
            f"[bold]Baseline Total:[/bold] ${stress_report['baseline_total_value']:,.2f}\n"
            f"[bold]Stressed Total:[/bold] [bold yellow]${stress_report['stressed_total_value']:,.2f}[/bold yellow]\n"
            f"[bold]Value Delta:[/bold] [bold red]${stress_report['value_delta']:,.2f}[/bold red] "
            f"({stress_report['percentage_change']:+.2f}%)"
        )
        console.print(Panel(stress_panel, title="Macroeconomic Stress Test Result", expand=False))

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

    table = Table(title=f"Market Overview & Submarkets ({report['submarket_count']} clusters)", show_header=True)
    table.add_column("Submarket ID", style="cyan")
    table.add_column("Properties", justify="right")
    table.add_column("Median Price", justify="right", style="bold green")
    table.add_column("Median $/Sqft", justify="right")
    table.add_column("Price Range", justify="right", style="dim")
    table.add_column("Volatility (%)", justify="right", style="yellow")

    for sm in report["submarkets"][:30]:
        price_range = f"${sm['min_price']:,.0f} - ${sm['max_price']:,.0f}"
        table.add_row(
            sm["submarket_id"],
            str(sm["property_count"]),
            f"${sm['median_price']:,.0f}",
            f"${sm['median_price_per_sqft']:,.0f}",
            price_range,
            f"{sm['price_volatility_pct']:.1f}%",
        )

    console.print(table)
    if report["submarket_count"] > 30:
        console.print(f"[dim]Showing first 30 of {report['submarket_count']} submarkets.[/dim]")
    if output:
        console.print(f"[dim]Full market report saved to {output}[/dim]")


if __name__ == "__main__":
    cli()

