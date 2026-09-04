# Python API & CLI Reference Manual

Comprehensive technical reference for **`real-estate-spatial`** modules, classes, methods, CLI commands, and data interchange formats.

---

## Table of Contents
1. [`real_estate_spatial.types`](#1-real_estate_spatialtypes)
2. [`real_estate_spatial.spatial`](#2-real_estate_spatialspatial)
3. [`real_estate_spatial.comps`](#3-real_estate_spatialcomps)
4. [`real_estate_spatial.models`](#4-real_estate_spatialmodels)
5. [`real_estate_spatial.pipeline`](#5-real_estate_spatialpipeline)
6. [`real_estate_spatial.data`](#6-real_estate_spatialdata)
7. [`real_estate_spatial.analytics`](#7-real_estate_spatialanalytics)
8. [Command Line Interface (CLI) Reference](#8-command-line-interface-cli-reference)
9. [Data Interchange Formats](#9-data-interchange-formats)

---

## 1. `real_estate_spatial.types`

### `PropertyType(str, Enum)`
Supported property classification categories:
- `SINGLE_FAMILY = "single_family"`
- `CONDO = "condo"`
- `TOWNHOUSE = "townhouse"`
- `MULTI_FAMILY = "multi_family"`
- `COMMERCIAL = "commercial"`
- `LAND = "land"`

### `ValuationMethodology(str, Enum)`
Valuation model execution methodology flags:
- `COMPS = "comps"`
- `HEDONIC = "hedonic"`
- `GBM = "gbm"`
- `ENSEMBLE = "ensemble"`
- `FALLBACK = "fallback"`

### `Coordinates`
```python
@dataclass(frozen=True)
class Coordinates:
    lat: float
    lon: float
```
- **Methods**:
  - `distance_to(other: Coordinates, radius_km: float = 6371.0) -> float`: Returns Haversine great-circle distance in kilometers.
  - `to_tuple() -> Tuple[float, float]`: Returns `(lat, lon)`.
  - `to_dict() -> Dict[str, float]`: Returns `{"lat": float, "lon": float}`.
  - `from_dict(data: Dict[str, Any]) -> Coordinates`: Parses dictionary with `lat`/`latitude` and `lon`/`longitude`.
  - `from_tuple(data: Sequence[float]) -> Coordinates`: Parses sequence `[lat, lon]`.

### `BoundingBox`
```python
@dataclass(frozen=True)
class BoundingBox:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
```
- **Methods**:
  - `contains(point: Coordinates | Sequence[float]) -> bool`: Evaluates if coordinate point is inside bounding box.
  - `to_tuple() -> Tuple[float, float, float, float]`: Returns `(min_lat, min_lon, max_lat, max_lon)`.
  - `to_dict() -> Dict[str, float]`: Returns dictionary format.
  - `from_dict(data: Dict[str, Any]) -> BoundingBox`: Deserializes dictionary.
  - `from_tuple(bbox: Sequence[float]) -> BoundingBox`: Deserializes 4-element sequence.

### `Property`
```python
@dataclass
class Property:
    id: str
    address: str
    coordinates: Coordinates
    price: Optional[float] = None
    sqft: float = 0.0
    bedrooms: int = 0
    bathrooms: float = 0.0
    lot_size_sqft: Optional[float] = None
    year_built: Optional[int] = None
    property_type: str = "single_family"
    sale_date: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
```
- **Properties & Methods**:
  - `price_per_sqft -> Optional[float]`: Computes price / sqft if both are positive.
  - `age(ref_year: int = 2026) -> float`: Calculates property age (default 25.0 if unknown).
  - `effective_lot_size() -> float`: Returns lot size or estimates as `3.0 * sqft`.
  - `to_dict() -> Dict[str, Any]`: Returns complete dictionary representation.
  - `from_dict(data: Dict[str, Any]) -> Property`: Deserializes dictionary.
  - `to_geojson_feature() -> Dict[str, Any]`: Formats into a GeoJSON Feature dictionary.

### `CompCandidate`
```python
@dataclass
class CompCandidate:
    property: Property
    similarity_score: float
    distance_km: float
    time_diff_days: Optional[int]
    adjusted_price: float
    adjustments: Dict[str, float] = field(default_factory=dict)
```
- **Properties & Methods**:
  - `net_adjustment -> float`: Sum of all adjustments.
  - `gross_adjustment -> float`: Sum of absolute adjustments.
  - `net_adjustment_pct -> float`: Ratio of net adjustment to original comp price.
  - `to_dict() -> Dict[str, Any]`: Serializes comp candidate details.
  - `from_dict(data: Dict[str, Any]) -> CompCandidate`: Deserializes comp candidate.

### `ValuationResult`
```python
@dataclass
class ValuationResult:
    property_id: str
    estimated_value: float
    confidence_score: float
    lower_bound: float
    upper_bound: float
    comps_used: List[Any] = field(default_factory=list)
    methodology: str = "valuation"
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=...)
```
- **Properties & Methods**:
  - `spread -> float`: Difference between upper and lower bounds.
  - `spread_pct -> float`: Spread as a percentage of estimated value.
  - `to_dict() -> Dict[str, Any]`: Serializes valuation result.
  - `from_dict(data: Dict[str, Any]) -> ValuationResult`: Deserializes valuation result.

### `ModelMetrics`
```python
@dataclass
class ModelMetrics:
    mape: float
    mdape: float
    rmse: float
    mae: float
    r2: float
    within_5_pct: float
    within_10_pct: float
    within_20_pct: float
```
- **Methods**:
  - `to_dict() -> Dict[str, float]`: Converts metrics to formatted dict.
  - `from_dict(data: Dict[str, Any]) -> ModelMetrics`: Deserializes metrics.
  - `summary() -> str`: Returns multi-line human-readable summary string.

---

## 2. `real_estate_spatial.spatial`

### Coordinate & Index Utilities (`real_estate_spatial.spatial`)
- `haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius_km: float = 6371.0) -> float`: Scalar Haversine distance in km.
- `haversine_distance_vectorized(lat1: float | np.ndarray, lon1: float | np.ndarray, lats: np.ndarray, lons: np.ndarray, radius_km: float = 6371.0) -> np.ndarray`: High-speed vectorized Haversine distance array.
- `bounding_box(center: Coordinates, radius_km: float) -> BoundingBox`: Calculates geographic bounding box for given radius.
- `point_in_bbox(point: Coordinates, bbox: BoundingBox) -> bool`: Tests if coordinate lies within bounding box.
- `chord_distance_to_great_circle_km(chord_dist: float) -> float`: Converts 3D Euclidean chord distance on the unit sphere to great-circle distance in kilometers.
- `great_circle_km_to_chord_distance(distance_km: float) -> float`: Converts great-circle distance in kilometers to 3D Euclidean chord distance on the unit sphere.
- `unpack_bbox(bbox: Union[BoundingBox, Tuple[float, float, float, float], Sequence[float]]) -> Tuple[float, float, float, float]`: Unpacks any supported bounding box representation into `(min_lat, min_lon, max_lat, max_lon)`.

### `KDTreeIndex(SpatialIndex)`
Spatial $k$-d tree with 3D unit-sphere projection.
- `__init__()`: Initializes empty tree.
- `build(properties: Sequence[Property]) -> None`: Builds unit-sphere $k$-d tree.
- `query_knn(point: Coordinates, k: int = 5) -> List[Tuple[Property, float]]`: Returns $k$ nearest properties and distances in km.
- `query_radius(point: Coordinates, radius_km: float) -> List[Tuple[Property, float]]`: Returns all properties within radius in km.
- `query_bbox(bbox: BoundingBox) -> List[Property]`: Returns all properties contained in bounding box.

### `GridIndex(SpatialIndex)`
Hash-based spatial grid indexing.
- `__init__(cell_size_deg: float = 0.05)`: Sets cell size in degrees.
- `build(properties: Sequence[Property]) -> None`: Populates spatial grid buckets.
- `query_knn(point: Coordinates, k: int = 5) -> List[Tuple[Property, float]]`: Finds $k$ nearest neighbors via expanding grid rings.
- `query_radius(point: Coordinates, radius_km: float) -> List[Tuple[Property, float]]`: Finds properties within spherical radius.
- `query_bbox(bbox: BoundingBox) -> List[Property]`: Finds properties in bounding box.

### `SpatialFeatureExtractor`
Extracts spatial lag, density, and landmark proximity features.
- `__init__(spatial_index: Optional[SpatialIndex] = None, landmarks: Optional[Dict[str, Coordinates]] = None)`
- `calculate_spatial_lag(point: Coordinates, properties: Sequence[Property], attribute: str = "price_per_sqft", k: int = 5, power: float = 1.0, exclude_id: Optional[str] = None) -> float`: Distance-weighted average of neighboring attribute.
- `calculate_local_density(point: Coordinates, properties: Sequence[Property], radius_km: float = 2.0, exclude_id: Optional[str] = None) -> float`: Property count within radius.
- `distance_to_landmarks(point: Coordinates) -> Dict[str, float]`: Distances to configured landmarks.
- `extract_features(prop: Property, dataset: Sequence[Property]) -> Dict[str, float]`: Full spatial feature dictionary for single property.
- `extract_features_batch(properties: Sequence[Property], dataset: Sequence[Property]) -> List[Dict[str, float]]`: Batch feature extraction.

### `SpatialIndexSerializer` & `SpatialQueryCache` (`real_estate_spatial.spatial.persistence`)
- `SpatialIndexSerializer.save_index(index: SpatialIndex, filepath: Union[str, Path], format: Optional[str] = None) -> str`: Persists spatial index to JSON or binary.
- `SpatialIndexSerializer.load_index(filepath: Union[str, Path], format: Optional[str] = None) -> SpatialIndex`: Deserializes spatial index.
- `SpatialQueryCache(maxsize: int = 1000)`: LRU cache for spatial $k$-NN, radius, and lag queries.

---

## 3. `real_estate_spatial.comps`

### `SimilarityScorer`
```python
class SimilarityScorer:
    def __init__(
        self,
        distance_decay_km: float = 5.0,
        sqft_tolerance: float = 500.0,
        bed_penalty: float = 0.1,
        bath_penalty: float = 0.05,
        age_tolerance_years: float = 15.0,
        lot_size_tolerance: float = 2000.0,
        weights: Optional[Dict[str, float]] = None,
    )
```
- `calculate_similarity(subject: Property, candidate: Property) -> Tuple[float, float, Optional[int]]`: Returns `(similarity_score, distance_km, time_diff_days)`.

### `CompAdjustmentCalculator`
```python
class CompAdjustmentCalculator:
    def __init__(
        self,
        default_price_per_sqft: float = 350.0,
        sqft_marginal_factor: float = 0.5,
        bedroom_value: float = 15000.0,
        bathroom_value: float = 10000.0,
        lot_sqft_marginal_factor: float = 5.0,
        annual_appreciation_rate: float = 0.04,
    )
```
- `calculate_adjustments(subject: Property, comp: Property, comp_price: Optional[float] = None, reference_date: Optional[str] = None) -> Tuple[float, Dict[str, float]]`: Computes adjusted price and dictionary of adjustments.

### `CompsEngine`
```python
class CompsEngine:
    def __init__(
        self,
        spatial_index: Optional[SpatialIndex] = None,
        scorer: Optional[SimilarityScorer] = None,
        adjustment_calculator: Optional[CompAdjustmentCalculator] = None,
        default_max_comps: int = 5,
        default_max_radius_km: float = 10.0,
    )
```
- `fit(properties: Sequence[Property]) -> CompsEngine`: Builds spatial index and stores comparable pool.
- `find_comps(subject: Property, max_comps: int = 5, max_radius_km: float = 10.0, min_similarity: float = 0.0, property_type_match: bool = True) -> List[CompCandidate]`: Retrieves ranked adjusted comparable sales.
- `value(subject: Property, max_comps: int = 5, max_radius_km: float = 10.0) -> ValuationResult`: Computes valuation from top comparables.

### Date Parsing Helper (`real_estate_spatial.comps.adjustments`)
- `parse_date_string(d_str: Optional[str]) -> Optional[date]`: Robust parser for ISO-8601 (with `Z` or UTC offset) and date strings (`YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY-MM`, `YYYY`). Returns `datetime.date` or `None`.

---

## 4. `real_estate_spatial.models`

### `BaseValuationModel` (alias `ValuationModel`)
Abstract base class defining the valuation model protocol:
- `fit(properties: Sequence[Property], validation_split: float = 0.0) -> BaseValuationModel`: Abstract method to train model on property records.
- `predict(properties: Sequence[Property]) -> np.ndarray`: Abstract method returning an array of predicted prices.
- `predict_one(prop: Property) -> ValuationResult`: Valuation result for a single property including confidence score, upper/lower bounds, and methodology.
- `evaluate(test_properties: Sequence[Property]) -> ModelMetrics`: Evaluates predictions on test set and computes standardized metrics (MAPE, MdAPE, RMSE, MAE, R², accuracy within 5%/10%/20%).
- `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics`: Static method computing metrics between ground truth and predictions.

### `HedonicSpatialRegression(BaseValuationModel)`
Closed-form regularized Ridge regression with coordinate polynomial interactions.
- `__init__(alpha: float = 1.0)`: Sets L2 regularization parameter.
- `fit(properties: Sequence[Property], validation_split: float = 0.0) -> HedonicSpatialRegression`: Fits regression model.
- `predict(properties: Sequence[Property]) -> np.ndarray`: Batch valuation predictions.
- `predict_one(prop: Property) -> ValuationResult`: Single property valuation with confidence and bounds.
- `feature_importances() -> Dict[str, float]`: Returns normalized feature weights.

### `SpatialGradientBoostingModel(ValuationModel)`
Gradient Boosting regressor with spatial lag and density integration (XGBoost with NumPy tree fallback).
- `__init__(n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1, random_state: int = 42)`
- `fit(properties: Sequence[Property], validation_split: float = 0.0) -> SpatialGradientBoostingModel`: Trains model.
- `predict(properties: Sequence[Property]) -> np.ndarray`: Array of predicted values.
- `predict_one(prop: Property) -> ValuationResult`: Single property valuation.

### `EnsembleValuationModel(ValuationModel)`
Weighted multi-model ensemble combining Hedonic, Spatial GBM, and Comps.
- `__init__(models: Optional[Sequence[ValuationModel]] = None, weights_strategy: str = "performance", custom_weights: Optional[Sequence[float] | Dict[str, float]] = None)`
- `fit(properties: Sequence[Property], validation_split: float = 0.2) -> EnsembleValuationModel`: Trains base models and solves ensemble weights.
- `predict(properties: Sequence[Property]) -> np.ndarray`: Blended predictions.
- `predict_one(prop: Property) -> ValuationResult`: Valuation with confidence and bounds.
- `evaluate(test_properties: Sequence[Property]) -> ModelMetrics`: Computes MAPE, MdAPE, RMSE, MAE, R², and accuracy buckets.

---

## 5. `real_estate_spatial.pipeline`

### `ValuationOrchestrator`
Unified automated valuation pipeline.
```python
class ValuationOrchestrator:
    def __init__(
        self,
        cleaner: Optional[DataCleaner] = None,
        validator: Optional[DataValidator] = None,
        feature_extractor: Optional[SpatialFeatureExtractor] = None,
        comps_engine: Optional[CompsEngine] = None,
        ensemble_model: Optional[EnsembleValuationModel] = None,
        spatial_cache: Optional[SpatialQueryCache] = None,
    )
```
- `fit_pipeline(training_data: Sequence[Property], auto_clean: bool = True, validation_split: float = 0.2) -> ValuationOrchestrator`: Cleans data, indexes spatial domain, and fits models.
- `value_property(prop: Property, strategy: str = "auto", max_comps: int = 5, max_radius_km: float = 10.0) -> ValuationResult`: Computes valuation using strategy (`auto`, `comps`, `ensemble`, `hedonic`, `gbm`, `fallback`).
- `batch_value(properties: Sequence[Property], strategy: str = "auto", max_comps: int = 5, max_radius_km: float = 10.0) -> List[ValuationResult]`: Batch valuations.
- `explain_valuation(result: ValuationResult) -> Dict[str, Any]`: Computes valuation breakdown, comp summary, key feature drivers, and narrative summary for a given `ValuationResult`.

---

## 6. `real_estate_spatial.data`

### `PropertyDataLoader`
- `load_json(filepath: Union[str, Path]) -> List[Property]`
- `save_json(properties: Sequence[Property], filepath: Union[str, Path]) -> str`
- `load_csv(filepath: Union[str, Path]) -> List[Property]`
- `save_csv(properties: Sequence[Property], filepath: Union[str, Path]) -> str`
- `load_geojson(filepath: Union[str, Path]) -> List[Property]`
- `save_geojson(properties: Sequence[Property], filepath: Union[str, Path]) -> str`

### `ValidationIssue`
```python
@dataclass
class ValidationIssue:
    property_id: str
    field: str
    rule: str
    value: Any
    message: str
    severity: str = "error"  # "error" | "warning"
```
- `to_dict() -> Dict[str, Any]`: Serializes validation issue.

### `ValidationResult`
```python
@dataclass
class ValidationResult:
    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    issues: List[ValidationIssue]
    summary: Dict[str, int]
```
- `to_dict() -> Dict[str, Any]`: Serializes validation result.

### `CleaningReport`
```python
@dataclass
class CleaningReport:
    initial_count: int
    cleaned_count: int
    dropped_count: int
    imputed_fields: Dict[str, int]
    dropped_reasons: Dict[str, int]
```
- `to_dict() -> Dict[str, Any]`: Serializes cleaning report.

### `DataValidator`
- `validate_property(prop: Property) -> List[ValidationIssue]`: Validates single property schema and value ranges.
- `validate_dataset(properties: Sequence[Property], raise_on_error: bool = False) -> ValidationResult`: Validates entire dataset and summarizes issues.
- `detect_spatial_outliers(properties: Sequence[Property], k_neighbors: int = 15, z_threshold: float = 3.0, method: str = "zscore") -> List[Tuple[Property, float, str]]`: Identifies local price per sqft outliers relative to $k$-nearest neighbors.

### `DataCleaner`
- `clean_dataset(properties: Sequence[Property], drop_invalid: bool = True, drop_outliers: bool = True, impute_missing: bool = True, deduplicate: bool = True) -> Tuple[List[Property], CleaningReport]`

### `SyntheticRealEstateGenerator`
- `__init__(center_lat: float = 37.7749, center_lon: float = -122.4194, seed: int = 42)`
- `generate(n_samples: int = 100, radius_km: float = 10.0) -> List[Property]`: Generates realistic property dataset.

---

## 7. `real_estate_spatial.analytics`

### `SubmarketMetrics`
```python
@dataclass
class SubmarketMetrics:
    submarket_id: str
    property_count: int
    center_lat: float
    center_lon: float
    median_price: float
    mean_price: float
    median_price_per_sqft: float
    mean_price_per_sqft: float
    min_price: float
    max_price: float
    price_std_dev: float
    price_volatility_pct: float
    median_sqft: float
    property_types: Dict[str, int]
```
- `to_dict() -> Dict[str, Any]`: Serializes submarket metrics.

### `SubmarketAnalyzer`
- `__init__(grid_cell_size_deg: float = 0.05, min_cluster_size: int = 2)`
- `identify_submarkets(properties: Sequence[Property], method: str = "grid") -> List[SubmarketMetrics]`: Aggregates spatial micro-markets.
- `generate_market_report(properties: Sequence[Property]) -> Dict[str, Any]`: Generates overall market summary and submarket breakdown.

### `PortfolioRiskSummary`
```python
@dataclass
class PortfolioRiskSummary:
    total_properties: int
    total_estimated_value: float
    mean_estimated_value: float
    median_estimated_value: float
    min_estimated_value: float
    max_estimated_value: float
    average_confidence: float
    high_risk_count: int
    high_risk_property_ids: List[str]
    total_lower_bound: float
    total_upper_bound: float
    portfolio_spread_pct: float
    property_type_breakdown: Dict[str, Dict[str, float]]
```
- `to_dict() -> Dict[str, Any]`: Serializes portfolio risk summary.

### `PortfolioAssessor`
- `__init__(orchestrator: Optional[ValuationOrchestrator] = None)`
- `assess_portfolio(properties: Sequence[Property], valuation_results: Optional[Sequence[ValuationResult]] = None, risk_confidence_threshold: float = 0.6, risk_spread_threshold_pct: float = 25.0) -> Tuple[List[ValuationResult], PortfolioRiskSummary]`
- `stress_test_portfolio(properties: Sequence[Property], valuation_results: Optional[Sequence[ValuationResult]] = None, shock_pct: float = -10.0, spatial_hotspots: Optional[List[Tuple[Coordinates, float, float]]] = None) -> Dict[str, Any]`

---

## 8. Command Line Interface (CLI) Reference

The CLI entrypoint is `real-estate-spatial`. All commands support `--json` for machine parsing.

### Global Commands
- `real-estate-spatial version [--json]`
- `real-estate-spatial info [--json]`

### Data Commands
- `real-estate-spatial generate -n <count> -o <path> [--lat <float>] [--lon <float>] [--radius-km <float>] [--seed <int>] [--json]`
- `real-estate-spatial validate [-d <dataset>] [--outliers/--no-outliers] [--k-neighbors <int>] [--z-threshold <float>] [--json]`
- `real-estate-spatial clean [-d <dataset>] [-o <path>] [--drop-invalid/--keep-invalid] [--drop-outliers/--keep-outliers] [--impute/--no-impute] [--json]`

### Valuation & Modeling Commands
- `real-estate-spatial comps [-d <dataset>] --lat <float> --lon <float> --sqft <float> [--beds <int>] [--baths <float>] [--lot-size <float>] [--year-built <int>] [-t <single_family|condo|townhouse|multi_family|land>] [-k <int>] [-r <float>] [--auto-clean] [--json]`
- `real-estate-spatial value [-d <dataset>] --lat <float> --lon <float> --sqft <float> [--beds <int>] [--baths <float>] [--lot-size <float>] [--year-built <int>] [-t <single_family|condo|townhouse|multi_family|land>] [-m <auto|ensemble|comps|hedonic|gbm>] [--auto-clean] [--explain] [--json]`
- `real-estate-spatial batch-value [-d <dataset>] -i <target_props> [-o <out_valuations>] [-s <auto|ensemble|comps|hedonic|gbm>] [--auto-clean] [--json]`
- `real-estate-spatial train [-d <dataset>] [-m <hedonic|gbm|ensemble>] [-o <output_metrics_json>] [--split <float>] [--auto-clean] [--json]`
- `real-estate-spatial benchmark [-d <dataset>] [--split <float>] [--auto-clean] [--json]`

### Analytics Commands
- `real-estate-spatial submarket [-d <dataset>] [-o <output_report_json>] [-m <grid|spatial_clustering>] [--cell-size <float>] [--min-size <int>] [--auto-clean] [--json]`
- `real-estate-spatial portfolio [-d <dataset>] -p <portfolio_file> [-o <output_report_json>] [--stress-shock <float>] [--auto-clean] [--json]`

---

## 9. Data Interchange Formats

### 9.1 JSON Schema
```json
[
  {
    "id": "PROP-001",
    "address": "123 Main St, San Francisco, CA",
    "coordinates": { "lat": 37.7749, "lon": -122.4194 },
    "price": 850000.0,
    "sqft": 1850.0,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "lot_size_sqft": 4500.0,
    "year_built": 1995,
    "property_type": "single_family",
    "sale_date": "2026-05-15",
    "features": { "pool": true }
  }
]
```

### 9.2 GeoJSON Schema
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "PROP-001",
      "geometry": {
        "type": "Point",
        "coordinates": [-122.4194, 37.7749]
      },
      "properties": {
        "id": "PROP-001",
        "address": "123 Main St, San Francisco, CA",
        "price": 850000.0,
        "sqft": 1850.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "lot_size_sqft": 4500.0,
        "year_built": 1995,
        "property_type": "single_family",
        "sale_date": "2026-05-15"
      }
    }
  ]
}
```

### 9.3 CSV Schema
```csv
id,address,latitude,longitude,price,sqft,bedrooms,bathrooms,lot_size_sqft,year_built,property_type,sale_date
PROP-001,123 Main St,37.7749,-122.4194,850000.0,1850.0,3,2.0,4500.0,1995,single_family,2026-05-15
```
