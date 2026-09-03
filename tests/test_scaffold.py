"""Scaffold verification tests for real-estate-spatial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import real_estate_spatial
from real_estate_spatial import (
    CompAdjustmentCalculator,
    CompCandidate,
    CompsEngine,
    Coordinates,
    DataCleaner,
    DataValidator,
    EnsembleValuationModel,
    GridIndex,
    HedonicSpatialRegression,
    KDTreeIndex,
    ModelMetrics,
    Property,
    PropertyDataLoader,
    SimilarityScorer,
    SpatialFeatureExtractor,
    SpatialGradientBoostingModel,
    SpatialIndex,
    SpatialIndexSerializer,
    SpatialQueryCache,
    SyntheticRealEstateGenerator,
    ValuationModel,
    ValuationOrchestrator,
    ValuationResult,
    __version__,
    bounding_box,
    haversine_distance,
    point_in_bbox,
)
from real_estate_spatial.cli.main import cli


def test_package_exports() -> None:
    """Verify package imports, version, and required top-level symbols."""
    assert __version__ == "0.1.0"
    assert real_estate_spatial.__version__ == "0.1.0"
    assert Property is not None
    assert Coordinates is not None
    assert SpatialIndex is not None
    assert CompsEngine is not None
    assert ValuationModel is not None
    assert SpatialIndexSerializer is not None
    assert SpatialQueryCache is not None
    assert DataValidator is not None
    assert DataCleaner is not None
    assert EnsembleValuationModel is not None
    assert ValuationOrchestrator is not None
    assert real_estate_spatial.cli is not None


def test_coordinates_instantiation_and_validation() -> None:
    """Verify Coordinates validation and distance calculation."""
    c1 = Coordinates(lat=37.7749, lon=-122.4194)
    assert c1.lat == 37.7749
    assert c1.lon == -122.4194

    c2 = Coordinates(lat=37.7750, lon=-122.4195)
    dist = c1.distance_to(c2)
    assert isinstance(dist, float)
    assert dist > 0.0

    with pytest.raises(ValueError):
        Coordinates(lat=95.0, lon=0.0)

    with pytest.raises(ValueError):
        Coordinates(lat=0.0, lon=-185.0)

    # Dictionary roundtrip
    d = c1.to_dict()
    assert d == {"lat": 37.7749, "lon": -122.4194}
    c_back = Coordinates.from_dict(d)
    assert c_back == c1


def test_property_instantiation_and_serialization(sample_property: Property) -> None:
    """Verify Property serialization and deserialization roundtrip."""
    prop = sample_property
    assert prop.id == "PROP-TEST-001"
    assert prop.sqft == 1850.0

    prop_dict = prop.to_dict()
    assert prop_dict["id"] == "PROP-TEST-001"
    assert prop_dict["coordinates"]["lat"] == 37.7749

    reconstructed = Property.from_dict(prop_dict)
    assert reconstructed.id == prop.id
    assert reconstructed.coordinates.lat == prop.coordinates.lat
    assert reconstructed.coordinates.lon == prop.coordinates.lon
    assert reconstructed.price == prop.price
    assert reconstructed.sqft == prop.sqft


def test_spatial_utilities() -> None:
    """Verify haversine distance, bounding box, and point containment."""
    d = haversine_distance(37.7749, -122.4194, 37.7749, -122.4194)
    assert d == pytest.approx(0.0, abs=1e-5)

    bbox = bounding_box(37.7749, -122.4194, radius_km=5.0)
    assert len(bbox) == 4
    min_lat, min_lon, max_lat, max_lon = bbox
    assert min_lat < 37.7749 < max_lat
    assert min_lon < -122.4194 < max_lon
    assert point_in_bbox(37.7749, -122.4194, bbox) is True
    assert point_in_bbox(38.5, -122.4194, bbox) is False


def test_spatial_indices(sample_properties: list[Property]) -> None:
    """Verify KDTreeIndex and GridIndex building, radius, and kNN queries."""
    target = Coordinates(lat=37.7750, lon=-122.4190)

    # 1. KDTreeIndex
    kdtree = KDTreeIndex(sample_properties)
    assert kdtree.size() == len(sample_properties)
    assert len(kdtree) == len(sample_properties)

    radius_props = kdtree.query_radius(target, radius_km=1.0)
    assert len(radius_props) >= 2

    knn_res = kdtree.query_knn(target, k=3)
    assert len(knn_res) == 3
    assert knn_res[0][1] <= knn_res[1][1] <= knn_res[2][1]

    # 2. GridIndex
    grid = GridIndex(sample_properties, cell_size_deg=0.01)
    assert grid.size() == len(sample_properties)
    grid_radius_props = grid.query_radius(target, radius_km=1.0)
    assert len(grid_radius_props) >= 2

    grid_knn = grid.query_knn(target, k=3)
    assert len(grid_knn) == 3


def test_spatial_features(sample_properties: list[Property]) -> None:
    """Verify spatial feature extraction."""
    extractor = SpatialFeatureExtractor()
    subject = sample_properties[0]
    feats = extractor.extract_all(subject, sample_properties)

    assert "lat" in feats
    assert "lon" in feats
    assert "spatial_lag_price_k5" in feats
    assert "density_1km" in feats
    assert feats["density_1km"] >= 1.0


def test_comps_engine_and_adjustments(sample_properties: list[Property]) -> None:
    """Verify CompsEngine, scorer, adjustments, and valuation output."""
    scorer = SimilarityScorer()
    adjuster = CompAdjustmentCalculator()
    engine = CompsEngine(scorer=scorer, adjuster=adjuster).fit(sample_properties)

    subject = Property(
        id="SUBJECT-01",
        address="110 Market St",
        coordinates=Coordinates(lat=37.7750, lon=-122.4190),
        sqft=1850.0,
        bedrooms=3,
        bathrooms=2.0,
        year_built=1996,
    )

    comps = engine.find_comps(subject, max_comps=3, max_radius_km=5.0)
    assert len(comps) > 0
    assert isinstance(comps[0], CompCandidate)
    assert comps[0].similarity_score > 0.0
    assert comps[0].adjusted_price > 0.0

    result = engine.value(subject, max_comps=3)
    assert isinstance(result, ValuationResult)
    assert result.estimated_value > 0.0
    assert result.lower_bound <= result.estimated_value <= result.upper_bound
    assert result.confidence_score > 0.0


def test_hedonic_model(sample_dataset: list[Property]) -> None:
    """Verify HedonicSpatialRegression training, prediction, and evaluation."""
    train_set = sample_dataset[:40]
    test_set = sample_dataset[40:]

    model = HedonicSpatialRegression(alpha=1.0)
    model.fit(train_set)
    assert model.is_fitted is True
    assert len(model.coefficients) > 0

    preds = model.predict(test_set)
    assert len(preds) == len(test_set)
    assert np.all(preds > 0)

    val_res = model.predict_one(test_set[0])
    assert isinstance(val_res, ValuationResult)
    assert val_res.estimated_value > 0.0

    metrics = model.evaluate(test_set)
    assert isinstance(metrics, ModelMetrics)
    assert metrics.mape >= 0.0
    assert metrics.rmse > 0.0


def test_spatial_gbm_model(sample_dataset: list[Property]) -> None:
    """Verify SpatialGradientBoostingModel training and prediction."""
    train_set = sample_dataset[:40]
    test_set = sample_dataset[40:]

    model = SpatialGradientBoostingModel(n_estimators=10, max_depth=3)
    model.fit(train_set)
    assert model.is_fitted is True

    preds = model.predict(test_set)
    assert len(preds) == len(test_set)
    assert np.all(preds > 0)

    val_res = model.predict_one(test_set[0])
    assert isinstance(val_res, ValuationResult)
    assert val_res.estimated_value > 0.0

    metrics = model.evaluate(test_set)
    assert isinstance(metrics, ModelMetrics)
    assert metrics.mae > 0.0


def test_data_generator_and_loader(tmp_path: Path) -> None:
    """Verify SyntheticRealEstateGenerator and PropertyDataLoader file formats."""
    gen = SyntheticRealEstateGenerator(seed=42)
    props = gen.generate(n_samples=25)
    assert len(props) == 25
    assert all(p.price is not None and p.price > 0 for p in props)

    # JSON roundtrip
    json_file = tmp_path / "props.json"
    PropertyDataLoader.save_json(props, json_file)
    loaded_json = PropertyDataLoader.load_json(json_file)
    assert len(loaded_json) == 25
    assert loaded_json[0].id == props[0].id

    # CSV roundtrip
    csv_file = tmp_path / "props.csv"
    PropertyDataLoader.save_csv(props, csv_file)
    loaded_csv = PropertyDataLoader.load_csv(csv_file)
    assert len(loaded_csv) == 25
    assert loaded_csv[0].coordinates.lat == pytest.approx(props[0].coordinates.lat, abs=1e-4)

    # GeoJSON roundtrip
    geojson_file = tmp_path / "props.geojson"
    PropertyDataLoader.save_geojson(props, geojson_file)
    loaded_geojson = PropertyDataLoader.load_geojson(geojson_file)
    assert len(loaded_geojson) == 25
    assert loaded_geojson[0].coordinates.lat == pytest.approx(props[0].coordinates.lat, abs=1e-4)


def test_cli_commands(tmp_path: Path) -> None:
    """Verify CLI commands execute cleanly."""
    runner = CliRunner()

    # 1. version
    v_res = runner.invoke(cli, ["version"])
    assert v_res.exit_code == 0
    assert "0.1.0" in v_res.output

    # 2. info
    info_res = runner.invoke(cli, ["info"])
    assert info_res.exit_code == 0
    assert "Spatial Indexing" in info_res.output

    # 3. generate
    data_file = tmp_path / "cli_test_data.json"
    gen_res = runner.invoke(cli, ["generate", "-n", "30", "-o", str(data_file)])
    assert gen_res.exit_code == 0
    assert data_file.exists()

    # 4. comps
    comps_res = runner.invoke(
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
            "1800",
        ],
    )
    assert comps_res.exit_code == 0
    assert "Comparable Properties" in comps_res.output

    # 5. value (comps, hedonic, gbm)
    for method in ["comps", "hedonic", "gbm"]:
        val_res = runner.invoke(
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
                method,
            ],
        )
        assert val_res.exit_code == 0
        assert "Estimated Value" in val_res.output

    # 6. train
    metrics_file = tmp_path / "metrics.json"
    train_res = runner.invoke(
        cli, ["train", "-d", str(data_file), "-m", "hedonic", "-o", str(metrics_file)]
    )
    assert train_res.exit_code == 0
    assert metrics_file.exists()

    # 7. benchmark
    bench_res = runner.invoke(cli, ["benchmark", "-d", str(data_file)])
    assert bench_res.exit_code == 0
    assert "Model Benchmark Comparison" in bench_res.output
