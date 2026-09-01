"""Comparable property price adjustment calculations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Optional, Tuple

from real_estate_spatial.types import Property


class CompAdjustmentCalculator:
    """Calculates standardized monetary adjustments to bring a comp in line with subject property."""

    def __init__(
        self,
        default_price_per_sqft: float = 350.0,
        sqft_marginal_factor: float = 0.5,
        bedroom_value: float = 15000.0,
        bathroom_value: float = 10000.0,
        lot_sqft_marginal_factor: float = 5.0,
        annual_appreciation_rate: float = 0.04,
    ) -> None:
        self.default_price_per_sqft = default_price_per_sqft
        self.sqft_marginal_factor = sqft_marginal_factor
        self.bedroom_value = bedroom_value
        self.bathroom_value = bathroom_value
        self.lot_sqft_marginal_factor = lot_sqft_marginal_factor
        self.annual_appreciation_rate = annual_appreciation_rate

    def _parse_date(self, d_str: Optional[str]) -> Optional[date]:
        if not d_str:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(d_str, fmt).date()
            except ValueError:
                continue
        return None

    def calculate_adjustments(
        self,
        subject: Property,
        comp: Property,
        comp_price: Optional[float] = None,
        reference_date: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate line-item adjustments and adjusted comp price."""
        base_price = comp_price if comp_price is not None else (comp.price or 0.0)
        adjustments: Dict[str, float] = {}

        # 1. Gross Living Area (Sqft) adjustment
        base_ppsqft = (
            (base_price / comp.sqft)
            if comp.sqft > 0 and base_price > 0
            else self.default_price_per_sqft
        )
        marginal_ppsqft = base_ppsqft * self.sqft_marginal_factor
        sqft_diff = subject.sqft - comp.sqft
        adjustments["sqft"] = sqft_diff * marginal_ppsqft

        # 2. Bedroom count adjustment
        bed_diff = subject.bedrooms - comp.bedrooms
        adjustments["bedrooms"] = float(bed_diff * self.bedroom_value)

        # 3. Bathroom count adjustment
        bath_diff = subject.bathrooms - comp.bathrooms
        adjustments["bathrooms"] = float(bath_diff * self.bathroom_value)

        # 4. Lot size adjustment (if both have lot sizes)
        if subject.lot_size_sqft is not None and comp.lot_size_sqft is not None:
            lot_diff = subject.lot_size_sqft - comp.lot_size_sqft
            adjustments["lot_size"] = float(lot_diff * self.lot_sqft_marginal_factor)
        else:
            adjustments["lot_size"] = 0.0

        # 5. Age / Year Built adjustment
        if subject.year_built is not None and comp.year_built is not None:
            year_diff = subject.year_built - comp.year_built
            # Approx $1,000 per year of age difference
            adjustments["year_built"] = float(year_diff * 1000.0)
        else:
            adjustments["year_built"] = 0.0

        # 6. Time trend / market appreciation adjustment
        ref_d = self._parse_date(reference_date) or date.today()
        sale_d = self._parse_date(comp.sale_date)
        if sale_d is not None and sale_d < ref_d:
            days_diff = (ref_d - sale_d).days
            years_diff = days_diff / 365.25
            time_adj = base_price * (
                (1.0 + self.annual_appreciation_rate) ** years_diff - 1.0
            )
            adjustments["market_time"] = float(time_adj)
        else:
            adjustments["market_time"] = 0.0

        total_adjustment = sum(adjustments.values())
        adjusted_price = max(1000.0, base_price + total_adjustment)

        return (adjusted_price, adjustments)
