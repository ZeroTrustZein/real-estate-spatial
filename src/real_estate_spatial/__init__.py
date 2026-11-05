"""real-estate-spatial: Spatial property valuation and automated comparables engine."""

__version__ = "0.1.0"

from real_estate_spatial.analytics.portfolio import (
    PortfolioAssessor,
    PortfolioRiskSummary,
)
from real_estate_spatial.analytics.submarket import SubmarketAnalyzer, SubmarketMetrics
from real_estate_spatial.cli.main import cli
from real_estate_spatial.comps.adjustments import CompAdjustmentCalculator
from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.comps.scorer import SimilarityScorer
from real_estate_spatial.data.cleaner import CleaningReport, DataCleaner
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader
from real_estate_spatial.data.validator import (
    DataValidator,
    ValidationIssue,
    ValidationResult,
)
from real_estate_spatial.models.base import BaseValuationModel, ValuationModel
from real_estate_spatial.models.ensemble import EnsembleValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.pipeline.orchestrator import ValuationOrchestrator
from real_estate_spatial.spatial.coordinates import (
    EARTH_RADIUS_KM,
    bounding_box,
    haversine_distance,
    haversine_distance_vectorized,
    point_in_bbox,
)
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.spatial.index import GridIndex, KDTreeIndex, SpatialIndex
from real_estate_spatial.spatial.persistence import (
    SpatialIndexSerializer,
    SpatialQueryCache,
)
from real_estate_spatial.types import (
    BoundingBox,
    CompCandidate,
    Coordinates,
    ModelMetrics,
    Property,
    PropertyType,
    ValuationMethodology,
    ValuationResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Coordinates",
    "BoundingBox",
    "Property",
    "PropertyType",
    "ValuationMethodology",
    "ValuationResult",
    "CompCandidate",
    "ModelMetrics",
    "SpatialIndex",
    "KDTreeIndex",
    "GridIndex",
    "SpatialFeatureExtractor",
    "SpatialIndexSerializer",
    "SpatialQueryCache",
    "EARTH_RADIUS_KM",
    "bounding_box",
    "haversine_distance",
    "haversine_distance_vectorized",
    "point_in_bbox",
    "SimilarityScorer",
    "CompAdjustmentCalculator",
    "CompsEngine",
    "BaseValuationModel",
    "ValuationModel",
    "HedonicSpatialRegression",
    "SpatialGradientBoostingModel",
    "EnsembleValuationModel",
    "ValuationOrchestrator",
    "SyntheticRealEstateGenerator",
    "PropertyDataLoader",
    "DataValidator",
    "ValidationResult",
    "ValidationIssue",
    "DataCleaner",
    "CleaningReport",
    "PortfolioAssessor",
    "PortfolioRiskSummary",
    "SubmarketAnalyzer",
    "SubmarketMetrics",
    "cli",
]
