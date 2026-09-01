"""Base valuation model interface and metric evaluation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

from real_estate_spatial.types import ModelMetrics, Property, ValuationResult


class BaseValuationModel(ABC):
    """Abstract base class for real estate spatial valuation models."""

    def __init__(self, name: str = "BaseValuationModel") -> None:
        self.name = name
        self.is_fitted = False
        self._training_properties: List[Property] = []

    @abstractmethod
    def fit(self, properties: Sequence[Property]) -> BaseValuationModel:
        """Fit valuation model on training properties."""
        ...

    @abstractmethod
    def predict(self, properties: Sequence[Property]) -> np.ndarray:
        """Predict prices for multiple properties."""
        ...

    def predict_one(self, prop: Property) -> ValuationResult:
        """Predict valuation for a single property with confidence intervals."""
        preds = self.predict([prop])
        estimated_val = float(preds[0])
        estimated_val = max(1000.0, estimated_val)

        confidence_score = 0.85 if self.is_fitted else 0.2
        margin = max(0.05, (1.0 - confidence_score) * 0.3)

        return ValuationResult(
            property_id=prop.id,
            estimated_value=estimated_val,
            confidence_score=confidence_score,
            lower_bound=estimated_val * (1.0 - margin),
            upper_bound=estimated_val * (1.0 + margin),
            comps_used=[],
            methodology=self.name,
            feature_contributions={},
        )

    def evaluate(self, properties: Sequence[Property]) -> ModelMetrics:
        """Evaluate model performance on test properties against ground truth prices."""
        valid_props = [p for p in properties if p.price is not None and p.price > 0]
        if not valid_props:
            raise ValueError("No properties with valid positive price provided for evaluation")

        y_true = np.array([p.price for p in valid_props], dtype=np.float64)
        y_pred = self.predict(valid_props)
        return self.compute_metrics(y_true, y_pred)

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate MAPE, MdAPE, RMSE, MAE, R^2, and accuracy bands (5%, 10%, 20%)."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if len(y_true) == 0:
            return ModelMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Avoid zero division in percentage calculations
        safe_true = np.where(y_true == 0, 1.0, y_true)
        abs_errors = np.abs(y_true - y_pred)
        pct_errors = abs_errors / np.abs(safe_true)

        mape = float(np.mean(pct_errors) * 100.0)
        mdape = float(np.median(pct_errors) * 100.0)
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        within_5_pct = float(np.mean(pct_errors <= 0.05) * 100.0)
        within_10_pct = float(np.mean(pct_errors <= 0.10) * 100.0)
        within_20_pct = float(np.mean(pct_errors <= 0.20) * 100.0)

        return ModelMetrics(
            mape=mape,
            mdape=mdape,
            rmse=rmse,
            mae=mae,
            r2=r2,
            within_5_pct=within_5_pct,
            within_10_pct=within_10_pct,
            within_20_pct=within_20_pct,
        )


ValuationModel = BaseValuationModel
