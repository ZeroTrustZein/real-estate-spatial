"""End-to-end spatial valuation pipeline orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.data.cleaner import CleaningReport, DataCleaner
from real_estate_spatial.data.validator import DataValidator
from real_estate_spatial.models.ensemble import EnsembleValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.spatial.persistence import SpatialQueryCache
from real_estate_spatial.types import CompCandidate, Property, ValuationResult


class ValuationOrchestrator:
    """Orchestrates property validation, cleaning, spatial feature extraction, comp search, and ML ensemble valuation."""

    def __init__(
        self,
        cleaner: Optional[DataCleaner] = None,
        validator: Optional[DataValidator] = None,
        feature_extractor: Optional[SpatialFeatureExtractor] = None,
        comps_engine: Optional[CompsEngine] = None,
        ensemble_model: Optional[EnsembleValuationModel] = None,
        spatial_cache: Optional[SpatialQueryCache] = None,
    ) -> None:
        self.validator = validator or DataValidator()
        self.cleaner = cleaner or DataCleaner(validator=self.validator)
        self.feature_extractor = feature_extractor or SpatialFeatureExtractor()
        self.comps_engine = comps_engine or CompsEngine()
        self.ensemble_model = ensemble_model or EnsembleValuationModel()
        self.spatial_cache = spatial_cache or SpatialQueryCache()
        self.cleaning_report: Optional[CleaningReport] = None
        self._properties: List[Property] = []
        self.is_fitted: bool = False

    def fit_pipeline(
        self,
        training_data: Sequence[Property],
        auto_clean: bool = True,
        validation_split: float = 0.2,
    ) -> ValuationOrchestrator:
        """Clean training dataset, build spatial indices, and train valuation models."""
        if auto_clean:
            cleaned_props, report = self.cleaner.clean_dataset(training_data)
            self.cleaning_report = report
        else:
            cleaned_props = list(training_data)

        self._properties = cleaned_props
        self.spatial_cache.clear()

        # Fit Comps Engine
        self.comps_engine.fit(self._properties)

        # Fit ML Ensemble (Hedonic + Spatial GBM)
        if len(self._properties) >= 5:
            self.ensemble_model.fit(
                self._properties, validation_split=validation_split
            )

        self.is_fitted = True
        return self

    def value_property(
        self,
        prop: Property,
        strategy: str = "auto",
        max_comps: int = 5,
        max_radius_km: float = 10.0,
    ) -> ValuationResult:
        """Estimate property value using auto-selection or specified strategy."""
        strat = strategy.lower()

        # Cache key lookup for KNN/Radius if applicable
        cached_comps = self.spatial_cache.get_radius(prop.coordinates, max_radius_km)
        if cached_comps is None and self.is_fitted:
            comps = self.comps_engine.find_comps(
                prop, max_comps=max_comps, max_radius_km=max_radius_km
            )
        else:
            comps = self.comps_engine.find_comps(
                prop, max_comps=max_comps, max_radius_km=max_radius_km
            )

        if strat == "auto":
            # Auto Strategy: Comps -> Blended -> Ensemble -> Heuristic Fallback
            if len(comps) >= 3:
                avg_sim = float(np.mean([c.similarity_score for c in comps]))
                if avg_sim >= 0.65:
                    comps_res = self.comps_engine.value(
                        prop, max_comps=max_comps, max_radius_km=max_radius_km
                    )
                    if self.ensemble_model.is_fitted:
                        ens_res = self.ensemble_model.predict_one(prop)
                        w_comps = max(0.1, comps_res.confidence_score)
                        w_ens = max(0.1, ens_res.confidence_score)
                        tot = w_comps + w_ens
                        blended_val = (
                            (w_comps / tot) * comps_res.estimated_value
                            + (w_ens / tot) * ens_res.estimated_value
                        )
                        blended_conf = min(0.98, max(comps_res.confidence_score, ens_res.confidence_score))
                        margin = max(0.04, (1.0 - blended_conf) * 0.22)

                        merged_contribs = dict(ens_res.feature_contributions)
                        for k, v in comps_res.feature_contributions.items():
                            merged_contribs[f"comp_{k}"] = v

                        return ValuationResult(
                            property_id=prop.id,
                            estimated_value=blended_val,
                            confidence_score=blended_conf,
                            lower_bound=blended_val * (1.0 - margin),
                            upper_bound=blended_val * (1.0 + margin),
                            comps_used=comps,
                            methodology="auto_blended_comps_ensemble",
                            feature_contributions=merged_contribs,
                        )
                    return comps_res

            if self.ensemble_model.is_fitted:
                res = self.ensemble_model.predict_one(prop)
                if comps:
                    res.comps_used = comps
                return res

            if len(self._properties) > 0:
                return self.comps_engine.value(
                    prop, max_comps=max_comps, max_radius_km=max_radius_km
                )

            return self._heuristic_fallback(prop)

        elif strat == "ensemble":
            if self.ensemble_model.is_fitted:
                res = self.ensemble_model.predict_one(prop)
                if comps:
                    res.comps_used = comps
                return res
            return self.value_property(prop, strategy="comps", max_comps=max_comps, max_radius_km=max_radius_km)

        elif strat == "comps":
            return self.comps_engine.value(
                prop, max_comps=max_comps, max_radius_km=max_radius_km
            )

        elif strat == "hedonic":
            hedonic = next(
                (m for m in self.ensemble_model.models if isinstance(m, HedonicSpatialRegression)),
                None,
            )
            if hedonic and hedonic.is_fitted:
                res = hedonic.predict_one(prop)
                if comps:
                    res.comps_used = comps
                return res
            return self.value_property(prop, strategy="comps", max_comps=max_comps, max_radius_km=max_radius_km)

        elif strat == "gbm":
            gbm = next(
                (m for m in self.ensemble_model.models if isinstance(m, SpatialGradientBoostingModel)),
                None,
            )
            if gbm and gbm.is_fitted:
                res = gbm.predict_one(prop)
                if comps:
                    res.comps_used = comps
                return res
            return self.value_property(prop, strategy="comps", max_comps=max_comps, max_radius_km=max_radius_km)

        elif strat == "heuristic":
            return self._heuristic_fallback(prop)

        else:
            raise ValueError(
                f"Unknown valuation strategy: '{strategy}'. Supported strategies: 'auto', 'ensemble', 'comps', 'hedonic', 'gbm', 'heuristic'"
            )

    def batch_value(
        self, properties: Sequence[Property], strategy: str = "auto"
    ) -> List[ValuationResult]:
        """Value multiple properties in batch."""
        return [self.value_property(p, strategy=strategy) for p in properties]

    def explain_valuation(self, result: ValuationResult) -> Dict[str, Any]:
        """Produce structured explainability dictionary with comps, bounds, drivers, and narrative summary."""
        comps_list = result.comps_used or []
        top_comp_addresses: List[str] = []
        distances: List[float] = []

        for c in comps_list:
            if isinstance(c, CompCandidate):
                top_comp_addresses.append(c.property.address)
                distances.append(c.distance_km)
            elif isinstance(c, dict):
                prop_dict = c.get("property", {})
                top_comp_addresses.append(str(prop_dict.get("address", "")))
                distances.append(float(c.get("distance_km", 0.0)))
            elif isinstance(c, Property):
                top_comp_addresses.append(c.address)

        avg_dist = float(np.mean(distances)) if distances else 0.0

        comps_summary: Dict[str, Any] = {
            "comp_count": len(comps_list),
            "number_of_comps": len(comps_list),
            "top_comp_addresses": top_comp_addresses,
            "distances": distances,
            "avg_distance_km": round(avg_dist, 3),
        }

        # Key drivers
        drivers = dict(
            sorted(
                result.feature_contributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:6]
        )

        conf_pct = round(result.confidence_score * 100, 1)
        val_str = f"${result.estimated_value:,.2f}"
        lower_str = f"${result.lower_bound:,.2f}"
        upper_str = f"${result.upper_bound:,.2f}"
        spread_str = f"{result.spread_pct * 100:.1f}%"

        narrative = (
            f"Property {result.property_id} is valued at {val_str} ({conf_pct}% confidence) "
            f"via {result.methodology}. Expected valuation range is {lower_str} - {upper_str} "
            f"(spread: {spread_str}) based on {len(comps_list)} comps."
        )

        return {
            "property_id": result.property_id,
            "estimated_value": result.estimated_value,
            "confidence_score": result.confidence_score,
            "methodology": result.methodology,
            "value_range": {
                "lower_bound": result.lower_bound,
                "upper_bound": result.upper_bound,
                "spread_pct": result.spread_pct,
            },
            "comps_summary": comps_summary,
            "key_drivers": drivers,
            "narrative_summary": narrative,
        }

    def _heuristic_fallback(self, prop: Property) -> ValuationResult:
        default_val = max(50000.0, prop.sqft * self.comps_engine.adjuster.default_price_per_sqft)
        return ValuationResult(
            property_id=prop.id,
            estimated_value=default_val,
            confidence_score=0.15,
            lower_bound=default_val * 0.8,
            upper_bound=default_val * 1.2,
            comps_used=[],
            methodology="fallback_sqft_heuristic",
            feature_contributions={},
        )
