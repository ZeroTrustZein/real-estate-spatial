"""Machine learning and statistical valuation models."""

from real_estate_spatial.models.base import (
    BaseValuationModel,
    extract_structural_features,
)
from real_estate_spatial.models.ensemble import EnsembleValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel

ValuationModel = BaseValuationModel

__all__ = [
    "BaseValuationModel",
    "ValuationModel",
    "HedonicSpatialRegression",
    "SpatialGradientBoostingModel",
    "EnsembleValuationModel",
    "extract_structural_features",
]
