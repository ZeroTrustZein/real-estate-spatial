"""Command-line interface for real-estate-spatial."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import real_estate_spatial
from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.types import Coordinates, Property

console = Console()


@click.group()
def cli() -> None:
    """Real Estate Spatial Valuation & Comparables Engine CLI."""
    pass


@cli.command(name="version")
def version_cmd() -> None:
    """Print package version."""
    console.print(f"[bold cyan]real-estate-spatial[/bold cyan] version [green]{real_estate_spatial.__version__}[/green]")


@cli.command(name="info")
def info_cmd() -> None:
    """Display package overview and capability info."""
    table = Table(title="Real Estate Spatial Engine Capabilities", show_header=True)
    table.add_column("Module", style="cyan")
    table.add_column("Components", style="white")
    table.add_column("Description", style="dim")

    table.add_row(
        "Spatial Indexing",
        "KDTreeIndex, GridIndex",
        "Fast spherical nearest neighbor and radius spatial queries",
    )
    table.add_row(
        "Automated Comps",
        "CompsEngine, SimilarityScorer, CompAdjustmentCalculator",
        "Finds top comparable sales, calculates appraisal adjustments",
    )
    table.add_row(
        "Hedonic Regression",
        "HedonicSpatialRegression",
        "L2 Regularized spatial regression with coordinate interaction",
    )
    table.add_row(
        "Spatial GBM",
        "SpatialGradientBoostingModel",
        "Tree ensemble model with spatial lag and density features",
    )
    table.add_row(
        "Data Engine",
        "SyntheticRealEstateGenerator, PropertyDataLoader",
        "Market simulation and multi-format I/O (CSV, JSON, GeoJSON)",
    )

    console.print(table)


@cli.command(name="generate")
@click.option("-n", "--count", default=200, help="Number of synthetic properties to generate")
@click.option("-o", "--output", default="properties.json", help="Output file path (.json, .csv, .geojson)")
@click.option("--lat", default=37.7749, help="Center latitude (e.g. 37.7749 for SF)")
@click.option("--lon", default=-122.4194, help="Center longitude (e.g. -122.4194 for SF)")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def generate_cmd(count: int, output: str, lat: float, lon: float, seed: int) -> None:
    """Generate synthetic spatial real estate market data."""
    console.print(f"[bold green]Generating {count} synthetic properties centered around ({lat}, {lon})...[/bold green]")
    gen = SyntheticRealEstateGenerator(center_lat=lat, center_lon=lon, seed=seed)
    props = gen.generate(n_samples=count)

    out_path = Path(output)
    if out_path.suffix.lower() == ".csv":
        PropertyDataLoader.save_csv(props, out_path)
    elif out_path.suffix.lower() in [".geojson"]:
        PropertyDataLoader.save_geojson(props, out_path)
    else:
        PropertyDataLoader.save_json(props, out_path)

    console.print(f"[bold green]Successfully saved {len(props)} properties to [cyan]{output}[/cyan][/bold green]")


def _load_dataset(filepath: str) -> list[Property]:
    path = Path(filepath)
    if not path.exists():
        console.print(f"[bold red]File not found: {filepath}[/bold red]")
        raise click.Abort()

    if path.suffix.lower() == ".csv":
        return PropertyDataLoader.load_csv(path)
    elif path.suffix.lower() == ".geojson":
        return PropertyDataLoader.load_geojson(path)
    return PropertyDataLoader.load_json(path)


@cli.command(name="comps")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option("--lat", required=True, type=float, help="Subject latitude")
@click.option("--lon", required=True, type=float, help="Subject longitude")
@click.option("--sqft", required=True, type=float, help="Subject square footage")
@click.option("--beds", default=3, type=int, help="Subject bedrooms")
@click.option("--baths", default=2.0, type=float, help="Subject bathrooms")
@click.option("-k", "--max-comps", default=5, type=int, help="Number of comps to retrieve")
@click.option("-r", "--radius", default=5.0, type=float, help="Search radius in km")
def comps_cmd(
    data: str,
    lat: float,
    lon: float,
    sqft: float,
    beds: int,
    baths: float,
    max_comps: int,
    radius: float,
) -> None:
    """Find and display comparable properties for a subject property."""
    props = _load_dataset(data)
    engine = CompsEngine()
    engine.fit(props)

    subject = Property(
        id="SUBJECT",
        address="Subject Property",
        coordinates=Coordinates(lat=lat, lon=lon),
        sqft=sqft,
        bedrooms=beds,
        bathrooms=baths,
    )

    comps = engine.find_comps(subject, max_comps=max_comps, max_radius_km=radius)

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
@click.option(
    "-m",
    "--method",
    type=click.Choice(["comps", "hedonic", "gbm"], case_sensitive=False),
    default="comps",
    help="Valuation methodology",
)
def value_cmd(
    data: str,
    lat: float,
    lon: float,
    sqft: float,
    beds: int,
    baths: float,
    method: str,
) -> None:
    """Estimate valuation for a property using chosen methodology."""
    props = _load_dataset(data)

    subject = Property(
        id="SUBJECT",
        address="Subject Property",
        coordinates=Coordinates(lat=lat, lon=lon),
        sqft=sqft,
        bedrooms=beds,
        bathrooms=baths,
    )

    if method == "comps":
        engine = CompsEngine().fit(props)
        result = engine.value(subject)
    elif method == "hedonic":
        model = HedonicSpatialRegression().fit(props)
        result = model.predict_one(subject)
    else:  # gbm
        gbm = SpatialGradientBoostingModel().fit(props)
        result = gbm.predict_one(subject)

    panel_content = (
        f"[bold cyan]Estimated Value:[/bold cyan] [bold green]${result.estimated_value:,.2f}[/bold green]\n"
        f"[bold]Confidence Score:[/bold] {result.confidence_score * 100:.1f}%\n"
        f"[bold]Valuation Range:[/bold] ${result.lower_bound:,.2f} - ${result.upper_bound:,.2f}\n"
        f"[bold]Methodology:[/bold] {result.methodology}"
    )

    console.print(
        Panel(panel_content, title=f"Property Valuation ({method.upper()})", expand=False)
    )


@cli.command(name="train")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["hedonic", "gbm"], case_sensitive=False),
    default="hedonic",
    help="Model type to train",
)
@click.option("-o", "--output", default="model_metrics.json", help="Output metrics path")
def train_cmd(data: str, method: str, output: str) -> None:
    """Train valuation model on dataset and output evaluation metrics."""
    props = _load_dataset(data)
    console.print(f"[cyan]Training {method} model on {len(props)} properties...[/cyan]")

    if method == "hedonic":
        model = HedonicSpatialRegression().fit(props)
    else:
        model = SpatialGradientBoostingModel().fit(props)

    metrics = model.evaluate(props)

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

    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    console.print(f"[dim]Metrics saved to {output}[/dim]")


@cli.command(name="benchmark")
@click.option("-d", "--data", default="properties.json", help="Dataset file path")
@click.option("--split", default=0.8, help="Train/test split ratio")
def benchmark_cmd(data: str, split: float) -> None:
    """Benchmark all valuation methods side-by-side with out-of-sample evaluation."""
    props = _load_dataset(data)
    if len(props) < 20:
        console.print("[red]Need at least 20 properties for benchmarking[/red]")
        return

    n_train = int(len(props) * split)
    train_props = props[:n_train]
    test_props = props[n_train:]

    console.print(f"[cyan]Benchmarking on {len(train_props)} train / {len(test_props)} test samples...[/cyan]")

    # 1. Comps Engine
    comps_engine = CompsEngine().fit(train_props)
    comps_preds = []
    for tp in test_props:
        res = comps_engine.value(tp)
        comps_preds.append(res.estimated_value)
    import numpy as np

    y_test = np.array([p.price for p in test_props], dtype=np.float64)
    comps_metrics = HedonicSpatialRegression.compute_metrics(y_test, np.array(comps_preds))

    # 2. Hedonic Model
    hedonic = HedonicSpatialRegression().fit(train_props)
    hedonic_metrics = hedonic.evaluate(test_props)

    # 3. GBM Model
    gbm = SpatialGradientBoostingModel().fit(train_props)
    gbm_metrics = gbm.evaluate(test_props)

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


if __name__ == "__main__":
    cli()
