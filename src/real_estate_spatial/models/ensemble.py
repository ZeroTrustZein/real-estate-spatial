"""Ensemble spatial valuation model combining Hedonic regression and Gradient Boosting."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from real_estate_spatial.models.base import BaseValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.types import ModelMetrics, Property, ValuationResult


class EnsembleValuationModel(BaseValuationModel):
    """Ensemble valuation model combining multiple spatial sub-models."""

    VALID_STRATEGIES = ("performance", "equal", "custom")

    def __init__(
        self,
        models: Optional[Sequence[BaseValuationModel]] = None,
        weights: Optional[Union[Dict[str, float], Sequence[float]]] = None,
        strategy: str = "performance",
        cv_folds: int = 3,
    ) -> None:
        super().__init__(name="EnsembleValuationModel")
        strat = strategy.lower()
        if strat not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Valid strategies: {self.VALID_STRATEGIES}"
            )

        if models is None:
            self.models: List[BaseValuationModel] = [
                HedonicSpatialRegression(),
                SpatialGradientBoostingModel(),
            ]
        else:
            self.models = list(models)
            if len(self.models) == 0:
                raise ValueError("EnsembleValuationModel requires at least one sub-model")

        self.strategy = strat
        self.weights_input = weights
        self.weights: Dict[str, float] = {}
        self.cv_folds = cv_folds
        self._submodel_metrics: Dict[str, ModelMetrics] = {}

    def fit(
        self, properties: Sequence[Property], validation_split: float = 0.2
    ) -> EnsembleValuationModel:
        """Fit each sub-model and compute ensemble weights."""
        valid_props = [p for p in properties if p.price is not None and p.price > 0]
        if len(valid_props) < 5:
            raise ValueError("Need at least 5 valid property sales to fit ensemble model")

        self._training_properties = list(valid_props)

        if self.strategy == "performance":
            n_total = len(valid_props)
            n_val = max(1, int(n_total * validation_split)) if 0.0 < validation_split < 1.0 else 0
            n_train = n_total - n_val

            if n_val > 0 and n_train >= 5:
                train_props = valid_props[:n_train]
                val_props = valid_props[n_train:]
            else:
                train_props = valid_props
                val_props = valid_props

            inv_scores: Dict[str, float] = {}
            for m in self.models:
                m.fit(train_props)
                metrics = m.evaluate(val_props)
                self._submodel_metrics[m.name] = metrics
                err = metrics.rmse if metrics.rmse > 0 else metrics.mape
                inv_scores[m.name] = 1.0 / (err + 1e-6)

            total_score = sum(inv_scores.values())
            if total_score > 0:
                self.weights = {k: v / total_score for k, v in inv_scores.items()}
            else:
                self.weights = {m.name: 1.0 / len(self.models) for m in self.models}

            # Refit all submodels on the full training set
            if train_props is not valid_props:
                for m in self.models:
                    m.fit(valid_props)

        elif self.strategy == "equal":
            for m in self.models:
                m.fit(valid_props)
                self._submodel_metrics[m.name] = m.evaluate(valid_props)
            self.weights = {m.name: 1.0 / len(self.models) for m in self.models}

        elif self.strategy == "custom":
            for m in self.models:
                m.fit(valid_props)
                self._submodel_metrics[m.name] = m.evaluate(valid_props)

            if isinstance(self.weights_input, dict):
                raw_w = {m.name: float(self.weights_input.get(m.name, 1.0)) for m in self.models}
                tot = sum(raw_w.values())
                self.weights = (
                    {k: v / tot for k, v in raw_w.items()}
                    if tot > 0
                    else {m.name: 1.0 / len(self.models) for m in self.models}
                )
            elif isinstance(self.weights_input, (list, tuple, np.ndarray)):
                if len(self.weights_input) != len(self.models):
                    raise ValueError(
                        f"Custom weights count ({len(self.weights_input)}) must match models count ({len(self.models)})"
                    )
                tot = float(sum(self.weights_input))
                self.weights = (
                    {m.name: float(w) / tot for m, w in zip(self.models, self.weights_input)}
                    if tot > 0
                    else {m.name: 1.0 / len(self.models) for m in self.models}
                )
            else:
                self.weights = {m.name: 1.0 / len(self.models) for m in self.models}

        self.is_fitted = True
        return self

    def predict(self, properties: Sequence[Property]) -> np.ndarray:
        """Combine sub-model predictions according to learned weights."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if not properties:
            return np.array([], dtype=np.float64)

        combined = np.zeros(len(properties), dtype=np.float64)
        for m in self.models:
            w = self.weights.get(m.name, 1.0 / len(self.models))
            preds = m.predict(properties)
            combined += w * preds

        return np.maximum(1000.0, combined)

    def predict_one(self, prop: Property) -> ValuationResult:
        """Predict valuation for a single property with confidence intervals and contributions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        preds = self.predict([prop])
        estimated_val = float(preds[0])

        sub_preds: List[float] = []
        feature_conts: Dict[str, float] = {}

        for m in self.models:
            sub_p = float(m.predict([prop])[0])
            sub_preds.append(sub_p)
            w = self.weights.get(m.name, 1.0 / len(self.models))
            feature_conts[f"submodel_{m.name}"] = sub_p

            if hasattr(m, "get_feature_contributions"):
                try:
                    m_conts = m.get_feature_contributions(prop)
                    for k, v in m_conts.items():
                        feature_conts[k] = feature_conts.get(k, 0.0) + w * v
                except Exception:
                    pass

        if len(sub_preds) > 1 and estimated_val > 0:
            rel_std = float(np.std(sub_preds) / estimated_val)
            confidence_score = float(np.clip(0.92 - rel_std * 0.5, 0.5, 0.98))
        else:
            confidence_score = 0.88

        margin = max(0.04, (1.0 - confidence_score) * 0.25)
        lower_bound = max(1000.0, estimated_val * (1.0 - margin))
        upper_bound = estimated_val * (1.0 + margin)

        return ValuationResult(
            property_id=prop.id,
            estimated_value=estimated_val,
            confidence_score=confidence_score,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            comps_used=[],
            methodology="ensemble_valuation",
            feature_contributions=feature_conts,
        )

    def get_model_weights(self) -> Dict[str, float]:
        """Return learned or assigned model weights."""
        return dict(self.weights)

    def get_submodel_metrics(self) -> Dict[str, ModelMetrics]:
        """Return evaluation metrics for each sub-model."""
        return dict(self._submodel_metrics)
