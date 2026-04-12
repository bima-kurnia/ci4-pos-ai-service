"""
services/product_forecast.py
==============================
Product demand forecasting using:
  1. Weighted Moving Average on each product's daily sales history
  2. Trend detection via linear slope
  3. Demand score = WMA × trend_multiplier
  4. Products ranked by predicted demand for next N days
"""

import numpy as np
import logging
from typing import List
from schemas.models import (
    ProductSalesEntry, ProductRanking, ProductForecastResponse
)

logger = logging.getLogger(__name__)


class ProductForecastService:
    """
    Ranks products by predicted demand for the next forecast period.

    Algorithm:
    - Compute exponential WMA on each product's daily quantity sold
    - Detect trend direction via linear regression slope
    - Score = WMA × trend_multiplier
    - Multiply by forecast_days to get total predicted qty
    """

    # Decay factor for WMA — higher = more weight on recent sales
    ALPHA = 0.35

    def forecast(
        self,
        product_sales: List[ProductSalesEntry],
        forecast_days: int = 30,
    ) -> ProductForecastResponse:

        logger.info(f"Starting product forecast: {len(product_sales)} products, {forecast_days} days")

        rankings: List[ProductRanking] = []

        for product in product_sales:
            if not product.daily_qty:
                continue

            # Extract time-series quantities
            sorted_entries = sorted(product.daily_qty, key=lambda x: x.get("date", ""))
            qty_series     = np.array([e.get("qty", 0) for e in sorted_entries], dtype=float)

            if len(qty_series) == 0:
                continue

            # ── Exponential WMA ──────────────────────────────────
            wma = self._exp_wma(qty_series)

            # ── Linear trend ─────────────────────────────────────
            slope, trend_str, trend_pct = self._compute_trend(qty_series)

            # ── Trend multiplier ─────────────────────────────────
            # Amplify growing products, discount falling ones
            if slope > 0:
                multiplier = 1.0 + min(slope / (wma + 1e-9) * 0.5, 0.3)
            elif slope < 0:
                multiplier = max(0.7, 1.0 + slope / (wma + 1e-9) * 0.3)
            else:
                multiplier = 1.0

            # ── Predicted total qty over forecast_days ───────────
            daily_predicted = max(0.0, wma * multiplier)
            total_predicted = int(round(daily_predicted * forecast_days))

            # ── Predicted revenue ─────────────────────────────────
            # Derive unit price from total_revenue / total_qty
            avg_price = (
                product.total_revenue / product.total_qty
                if product.total_qty > 0
                else 0.0
            )
            predicted_revenue = round(total_predicted * avg_price, 2)

            # ── Confidence ────────────────────────────────────────
            # More historical data = higher confidence
            data_points = len(qty_series)
            confidence  = min(0.95, 0.5 + data_points * 0.01)

            rankings.append(ProductRanking(
                rank              = 0,  # assigned after sort
                product_id        = product.product_id,
                name              = product.name,
                predicted_qty     = total_predicted,
                predicted_revenue = predicted_revenue,
                trend             = trend_str,
                trend_pct         = round(trend_pct, 2),
                confidence        = round(confidence, 3),
            ))

        # ── Sort by predicted quantity descending ─────────────────
        rankings.sort(key=lambda x: x.predicted_qty, reverse=True)

        # Assign rank numbers
        for i, r in enumerate(rankings):
            r.rank = i + 1

        logger.info(f"Product forecast complete — top product: {rankings[0].name if rankings else 'N/A'}")

        return ProductForecastResponse(
            status        = "success",
            forecast_days = forecast_days,
            rankings      = rankings,
            model_used    = "ExponentialWMA + LinearTrendAmplifier",
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _exp_wma(self, series: np.ndarray) -> float:
        """Exponential Weighted Moving Average — returns smoothed last value."""
        if len(series) == 0:
            return 0.0
        smoothed = series[0]
        for val in series[1:]:
            smoothed = self.ALPHA * val + (1 - self.ALPHA) * smoothed
        return float(smoothed)

    def _compute_trend(self, series: np.ndarray):
        """
        Returns (slope, trend_str, trend_pct).
        Fits linear regression and computes % change over the window.
        """
        n = len(series)
        if n < 3:
            return 0.0, "stable", 0.0

        X = np.arange(n, dtype=float)
        x_mean = X.mean()
        y_mean = series.mean()

        num = np.sum((X - x_mean) * (series - y_mean))
        den = np.sum((X - x_mean) ** 2)
        slope = float(num / den) if den != 0 else 0.0

        # Percentage change: slope×n / mean
        trend_pct = (slope * n / y_mean * 100) if y_mean > 0 else 0.0

        if trend_pct > 10:
            trend_str = "rising"
        elif trend_pct < -10:
            trend_str = "falling"
        else:
            trend_str = "stable"

        return slope, trend_str, trend_pct