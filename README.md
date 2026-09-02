# real-estate-spatial

[![CI](https://github.com/zein/real-estate-spatial/actions/workflows/ci.yml/badge.svg)](https://github.com/zein/real-estate-spatial/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Real Estate Spatial Valuation & Market Analytics Engine** is a high-performance Python framework for automated real estate valuation (AVM), appraisal-grade comparable sales selection, spatial feature engineering, and portfolio-level macroeconomic stress testing.

---

## Key Features

- **Geospatial Core & High-Speed Indexing**:
  - `KDTreeIndex` with 3D unit-sphere Euclidean mapping for true spherical distance calculation without map distortion.
  - `GridIndex` with dynamic spatial partitioning and O(1) cell hashing.
  - Feature extraction: spatial lag smoothing, local density estimation, landmark proximity, and spatial coordinate interactions.
- **Appraisal-Grade Comps Engine**:
  - Multi-dimensional similarity scoring (Haversine distance decay, GLA, bed/bath mismatch, age, lot size, property type).
  - Standardized monetary adjustments (GLA marginal utility, bed/bath premiums, lot differential, age deprecation, and compounding time-trend appreciation).
- **Hedonic Ridge & Spatial Gradient Boosting**:
  - `HedonicSpatialRegression`: Closed-form L2 regularized Ridge solver with coordinate polynomial expansions.
  - `SpatialGradientBoostingModel`: Native XGBoost engine with a built-in pure NumPy decision tree booster fallback.
  - `EnsembleValuationModel`: Performance-weighted (Inverse-MAE), equal-weighted, or custom-weighted blending.
- **End-to-End Valuation Pipeline**:
  - `ValuationOrchestrator` handling data cleaning, spatial indexing, feature extraction, multi-model execution, automated fallback strategies, and confidence interval estimation.
  - Explainability: automated feature contribution breakdowns and valuation drivers.
- **Data Engine & Validation**:
  - Automated detection of spatial coordinate outliers via Z-score and Interquartile Range (IQR).
  - Outlier truncation and median imputation for missing property features.
  - Multi-format I/O support: JSON arrays/wrapped objects, GeoJSON FeatureCollections, and CSVs.
  - `SyntheticRealEstateGenerator` for realistic urban real estate market simulation.
- **Submarket & Portfolio Analytics**:
  - Spatial micro-market clustering, pricing distributions, and price volatility analysis.
  - Portfolio aggregate valuation, high-risk property identification, and macroeconomic + localized spatial shock stress testing.

---

## Installation

### Basic Installation (Core dependencies only)
```bash
pip install -e .
```

### With Machine Learning Accelerators (XGBoost, Scikit-Learn, SciPy)
```bash
pip install -e ".[ml]"
```

### Full Development & ML Suite
```bash
pip install -e ".[all]"
```

---

## Quickstart

### Python API

```python
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.types import Coordinates, Property

# 1. Generate synthetic market data around San Francisco
generator = SyntheticRealEstateGenerator(center_lat=37.7749, center_lon=-122.4194, seed=42)
dataset = generator.generate(n_samples=500)

# 2. Instantiate and train the Valuation Orchestrator
orchestrator = ValuationOrchestrator()
orchestrator.fit_pipeline(dataset, auto_clean=True, validation_split=0.2)

# 3. Create a target subject property for valuation
subject = Property(
    id="SUBJ-001",
    address="450 Market St, San Francisco, CA",
    coordinates=Coordinates(lat=37.7760, lon=-122.4180),
    sqft=1850.0,
    bedrooms=3,
    bathrooms=2.0,
    lot_size_sqft=4200.0,
    year_built=1995,
    property_type="single_family",
)

# 4. Run automated valuation
result = orchestrator.value_property(subject, strategy="auto", max_comps=5)

print(f"Property:         {result.property_id}")
print(f"Estimated Value:  ${result.estimated_value:,.2f}")
print(f"Confidence:       {result.confidence_score * 100:.1f}%")
print(f"Valuation Range:  ${result.lower_bound:,.2f} - ${result.upper_bound:,.2f}")
print(f"Methodology:      {result.methodology}")

# 5. Inspect valuation drivers and comparable sales
explanation = orchestrator.explain_valuation(result)
print(f"\nNarrative: {explanation['narrative_summary']}")
print("\nKey Valuation Drivers:")
for feature, impact in explanation["key_drivers"].items():
    print(f"  - {feature}: ${impact:,.2f}")

print(f"\nComps Selected: {len(result.comps_used)}")
for comp in result.comps_used[:3]:
    print(f"  Comp {comp.property.id}: Base=${comp.property.price:,.0f} -> Adj=${comp.adjusted_price:,.0f} (Sim: {comp.similarity_score:.3f})")
```

---

### Command Line Interface (CLI)

The CLI executable `real-estate-spatial` provides full access to the toolkit:

```bash
# Display package capabilities and version
real-estate-spatial info
real-estate-spatial version

# 1. Generate synthetic property dataset (JSON / CSV / GeoJSON)
real-estate-spatial generate -n 500 -o data/properties.json --lat 37.7749 --lon -122.4194

# 2. Validate dataset for missing fields, schema errors, and spatial outliers
real-estate-spatial validate -i data/properties.json

# 3. Clean dataset (deduplication, median imputation, outlier removal)
real-estate-spatial clean -i data/properties.json -o data/properties_cleaned.json

# 4. Search appraisal comparables for a target property
real-estate-spatial comps -i data/properties_cleaned.json \
  --lat 37.7760 --lon -122.4180 --sqft 1850 --beds 3 --baths 2.0 -k 5

# 5. Run end-to-end automated property valuation with explainability
real-estate-spatial value -i data/properties_cleaned.json \
  --lat 37.7760 --lon -122.4180 --sqft 1850 --beds 3 --baths 2.0 --strategy auto

# 6. Batch value an entire portfolio of target properties
real-estate-spatial batch-value -d data/properties_cleaned.json -i target_props.json -o valuations.json

# 7. Train and evaluate spatial valuation models (Hedonic, GBM, Ensemble)
real-estate-spatial train -i data/properties_cleaned.json --model ensemble --split 0.2

# 8. Benchmark all model architectures head-to-head
real-estate-spatial benchmark -i data/properties_cleaned.json --split 0.2

# 9. Analyze spatial submarkets and pricing volatility
real-estate-spatial submarket -i data/properties_cleaned.json --cell-size 0.05

# 10. Portfolio risk assessment and macroeconomic stress testing
real-estate-spatial portfolio -i data/properties_cleaned.json \
  --shock-pct -12.5 --hotspot "37.7749,-122.4194,5.0,-5.0"
```

---

## Project Structure

```
real-estate-spatial/
├── docs/
│   ├── ARCHITECTURE.md          # Complete architectural deep dive & mathematical formulas
│   └── API_REFERENCE.md         # Full Python API and CLI reference
├── src/
│   └── real_estate_spatial/
│       ├── __init__.py          # Top-level exports and version
│       ├── types.py             # Domain models (Property, Coordinates, ValuationResult, etc.)
│       ├── spatial/
│       │   ├── coordinates.py   # Spherical geodesics and bounding box routines
│       │   ├── features.py      # Spatial lag, density, and landmark feature extractor
│       │   ├── index.py         # KDTree and Grid spatial indices
│       │   └── persistence.py   # Serializers and LRU spatial query caching
│       ├── comps/
│       │   ├── adjustments.py   # Appraisal-standard price adjustment engine
│       │   ├── engine.py        # Comparable property search & weighting
│       │   └── scorer.py        # Multi-attribute similarity scoring
│       ├── models/
│       │   ├── base.py          # Abstract ValuationModel interface
│       │   ├── hedonic.py       # Hedonic Spatial Ridge regression solver
│       │   ├── spatial_gbm.py   # Spatial Gradient Boosting (XGBoost + NumPy fallback)
│       │   └── ensemble.py      # Weighted multi-model ensemble
│       ├── pipeline/
│       │   └── orchestrator.py  # End-to-end ValuationOrchestrator pipeline
│       ├── data/
│       │   ├── cleaner.py       # Imputation and outlier cleaning
│       │   ├── generator.py     # Synthetic spatial property data generator
│       │   ├── loader.py        # Multi-format data loader and serializer
│       │   └── validator.py     # Schema validation and spatial outlier detector
│       ├── analytics/
│       │   ├── portfolio.py     # Portfolio risk analysis and stress testing
│       │   └── submarket.py     # Micro-market spatial clustering and volatility
│       └── cli/
│           ├── formatters.py    # Rich CLI UI table and panel formatters
│           └── main.py          # Click CLI subcommands
├── tests/                       # Complete unit and integration test suite
├── pyproject.toml               # Package build configuration and dependencies
└── README.md                    # Project overview and quickstart
```

---

## Testing & Quality Assurance

Run the test suite with test coverage reporting:

```bash
# Run pytest with coverage
python -m pytest

# Run linting and code style checks via Ruff
python -m ruff check .
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
