"""Reusable Rich table and panel formatting helpers for real-estate-spatial CLI."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from rich.panel import Panel
from rich.table import Table

from real_estate_spatial.types import CompCandidate, ModelMetrics, Property, ValuationResult


def create_capabilities_table(capabilities: Sequence[Dict[str, str]]) -> Table:
    """Format package capabilities overview table."""
    table = Table(title="Real Estate Spatial Engine Capabilities", show_header=True)
    table.add_column("Module", style="cyan")
    table.add_column("Components", style="white")
    table.add_column("Description", style="dim")

    for cap in capabilities:
        table.add_row(cap["module"], cap["components"], cap["description"])
    return table


def create_validation_summary_table(
    val_res: Any, spatial_outliers_count: int, dataset_name: str
) -> Table:
    """Format dataset validation summary table."""
    table = Table(title=f"Dataset Validation Summary ({dataset_name})", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    table.add_row("Total Records", str(val_res.total_records))
    table.add_row("Valid Records", f"[green]{val_res.valid_records}[/green]")
    table.add_row(
        "Invalid Records",
        f"[red]{val_res.invalid_records}[/red]" if val_res.invalid_records else "0",
    )
    table.add_row("Total Issues", str(len(val_res.issues)))
    table.add_row("Spatial Outliers", str(spatial_outliers_count))
    return table


def create_validation_issues_table(
    issues: Sequence[Any], max_rows: int = 15
) -> Table:
    """Format sample validation issues table."""
    table = Table(title="Validation Issues (Sample)", show_header=True)
    table.add_column("Property ID", style="cyan")
    table.add_column("Field", style="yellow")
    table.add_column("Rule", style="dim")
    table.add_column("Severity", justify="center")
    table.add_column("Message")

    for issue in issues[:max_rows]:
        sev_style = "[red]ERROR[/red]" if issue.severity == "error" else "[yellow]WARN[/yellow]"
        table.add_row(issue.property_id, issue.field, issue.rule, sev_style, issue.message)
    return table


def create_spatial_outliers_table(
    spatial_outliers: Sequence[Tuple[Property, float, str]], max_rows: int = 15
) -> Table:
    """Format detected spatial outliers table."""
    table = Table(title="Spatial Outliers Detected", show_header=True)
    table.add_column("Property ID", style="cyan")
    table.add_column("Z-Score", justify="right", style="magenta")
    table.add_column("Reason", style="dim")

    for p, z, reason in spatial_outliers[:max_rows]:
        table.add_row(p.id, f"{z:.2f}", reason)
    return table


def create_cleaning_report_table(
    report: Any, data_name: str, output_name: str
) -> Table:
    """Format dataset cleaning metrics table."""
    table = Table(
        title=f"Dataset Cleaning Report ({data_name} -> {output_name})",
        show_header=True,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="white")

    table.add_row("Initial Records", str(report.initial_count))
    table.add_row("Cleaned Records", f"[bold green]{report.cleaned_count}[/bold green]")
    table.add_row(
        "Dropped Records",
        f"[bold red]{report.dropped_count}[/bold red]" if report.dropped_count else "0",
    )

    for reason, cnt in report.dropped_reasons.items():
        table.add_row(f"  Dropped: {reason}", str(cnt))

    for field_name, cnt in report.imputed_fields.items():
        table.add_row(f"  Imputed: {field_name}", f"[yellow]{cnt}[/yellow]")

    return table


def create_comps_table(
    comps: Sequence[CompCandidate], radius_km: float
) -> Table:
    """Format top comparable sales table."""
    table = Table(title=f"Top {len(comps)} Comparable Properties (Radius: {radius_km} km)")
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
    return table


def create_valuation_panel(result: ValuationResult, method: str) -> Panel:
    """Format single property valuation result panel."""
    panel_content = (
        f"[bold cyan]Estimated Value:[/bold cyan] [bold green]${result.estimated_value:,.2f}[/bold green]\n"
        f"[bold]Confidence Score:[/bold] {result.confidence_score * 100:.1f}%\n"
        f"[bold]Valuation Range:[/bold] ${result.lower_bound:,.2f} - ${result.upper_bound:,.2f} "
        f"([dim]spread: {result.spread_pct * 100:.1f}%[/dim])\n"
        f"[bold]Methodology:[/bold] {result.methodology}\n"
        f"[bold]Comps Used:[/bold] {len(result.comps_used) if result.comps_used else 0}"
    )
    return Panel(panel_content, title=f"Property Valuation ({method.upper()})", expand=False)


def create_valuation_drivers_table(key_drivers: Dict[str, float]) -> Table:
    """Format top valuation drivers table."""
    table = Table(title="Top Valuation Drivers", show_header=True)
    table.add_column("Feature / Component", style="cyan")
    table.add_column("Impact / Weight", justify="right", style="bold green")
    for k, v in key_drivers.items():
        val_fmt = f"${v:,.2f}" if abs(v) > 100 else f"{v:+.2f}"
        table.add_row(k, val_fmt)
    return table


def create_batch_valuation_table(
    results: Sequence[ValuationResult], max_rows: int = 50
) -> Table:
    """Format batch valuation summary table."""
    table = Table(
        title=f"Batch Valuation Results ({len(results)} properties)",
        show_header=True,
    )
    table.add_column("Property ID", style="cyan")
    table.add_column("Estimated Value", justify="right", style="bold green")
    table.add_column("Confidence", justify="right")
    table.add_column("Valuation Range", justify="right", style="dim")
    table.add_column("Methodology", style="white")

    for r in results[:max_rows]:
        val_str = f"${r.estimated_value:,.2f}"
        conf_str = f"{r.confidence_score * 100:.1f}%"
        range_str = f"${r.lower_bound:,.0f} - ${r.upper_bound:,.0f}"
        table.add_row(r.property_id, val_str, conf_str, range_str, r.methodology)
    return table


def create_training_metrics_table(metrics: ModelMetrics, method: str) -> Table:
    """Format model training evaluation metrics table."""
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
    return table


def create_benchmark_table(
    metrics_items: Sequence[Tuple[str, ModelMetrics]]
) -> Table:
    """Format side-by-side model benchmark comparison table."""
    table = Table(title="Model Benchmark Comparison (Out-of-Sample)")
    table.add_column("Model / Method", style="cyan")
    table.add_column("MAPE (%)", justify="right")
    table.add_column("MdAPE (%)", justify="right")
    table.add_column("MAE ($)", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Within 10%", justify="right", style="green")

    for name, m in metrics_items:
        table.add_row(
            name,
            f"{m.mape:.2f}%",
            f"{m.mdape:.2f}%",
            f"${m.mae:,.0f}",
            f"{m.r2:.3f}",
            f"{m.within_10_pct:.1f}%",
        )
    return table


def create_portfolio_summary_table(summary: Any, property_count: int) -> Table:
    """Format portfolio overview summary table."""
    table = Table(
        title=f"Portfolio Assessment Summary ({property_count} properties)",
        show_header=True,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold green")

    table.add_row("Total Properties", str(summary.total_properties))
    table.add_row("Total Estimated Value", f"${summary.total_estimated_value:,.2f}")
    table.add_row("Mean Property Value", f"${summary.mean_estimated_value:,.2f}")
    table.add_row("Median Property Value", f"${summary.median_estimated_value:,.2f}")
    table.add_row("Average Confidence", f"{summary.average_confidence * 100:.1f}%")
    table.add_row(
        "High-Risk Count",
        f"[red]{summary.high_risk_count}[/red]" if summary.high_risk_count else "0",
    )
    table.add_row(
        "Valuation Range",
        f"${summary.total_lower_bound:,.0f} - ${summary.total_upper_bound:,.0f}",
    )
    table.add_row("Portfolio Spread", f"{summary.portfolio_spread_pct:.1f}%")
    return table


def create_portfolio_type_breakdown_table(
    breakdown: Dict[str, Dict[str, Any]]
) -> Table:
    """Format portfolio property type breakdown table."""
    type_table = Table(title="Portfolio Breakdown by Property Type", show_header=True)
    type_table.add_column("Property Type", style="cyan")
    type_table.add_column("Count", justify="right")
    type_table.add_column("Total Value", justify="right", style="green")
    type_table.add_column("Mean Value", justify="right")
    type_table.add_column("Mean Sqft", justify="right")

    for ptype, stats in breakdown.items():
        type_table.add_row(
            ptype,
            str(stats["count"]),
            f"${stats['total_value']:,.0f}",
            f"${stats['mean_value']:,.0f}",
            f"{stats['mean_sqft']:,.0f}",
        )
    return type_table


def create_stress_test_panel(
    stress_report: Dict[str, Any], stress_shock: float
) -> Panel:
    """Format macroeconomic stress test results panel."""
    stress_content = (
        f"[bold]Stress Shock:[/bold] {stress_shock:+.1f}%\n"
        f"[bold]Baseline Total:[/bold] ${stress_report['baseline_total_value']:,.2f}\n"
        f"[bold]Stressed Total:[/bold] [bold yellow]${stress_report['stressed_total_value']:,.2f}[/bold yellow]\n"
        f"[bold]Value Delta:[/bold] [bold red]${stress_report['value_delta']:,.2f}[/bold red] "
        f"({stress_report['percentage_change']:+.2f}%)"
    )
    return Panel(stress_content, title="Macroeconomic Stress Test Result", expand=False)


def create_submarket_table(
    report: Dict[str, Any], max_rows: int = 30
) -> Table:
    """Format submarket analytics summary table."""
    table = Table(
        title=f"Market Overview & Submarkets ({report['submarket_count']} clusters)",
        show_header=True,
    )
    table.add_column("Submarket ID", style="cyan")
    table.add_column("Properties", justify="right")
    table.add_column("Median Price", justify="right", style="bold green")
    table.add_column("Median $/Sqft", justify="right")
    table.add_column("Price Range", justify="right", style="dim")
    table.add_column("Volatility (%)", justify="right", style="yellow")

    for sm in report["submarkets"][:max_rows]:
        price_range = f"${sm['min_price']:,.0f} - ${sm['max_price']:,.0f}"
        table.add_row(
            sm["submarket_id"],
            str(sm["property_count"]),
            f"${sm['median_price']:,.0f}",
            f"${sm['median_price_per_sqft']:,.0f}",
            price_range,
            f"{sm['price_volatility_pct']:.1f}%",
        )
    return table
