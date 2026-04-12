"""
services/sales_forecast.py
===========================
Sales forecasting using:
  1. Linear Regression on time-indexed revenue (trend line)
  2. Weighted Moving Average for smoothing
  3. Seasonality detection (weekday effects)

No Prophet dependency — pure numpy + scikit-learn.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Tuple
from schemas.models import DailySale, DayPrediction, SalesForecastResponse

logger = logging.getLogger(__name__)


class SalesForecastService:
    """
    Forecasts future daily revenue and transaction count using
    a combination of linear trend + weighted moving average.
    """

    # Confidence interval multiplier (±1.5 × std dev)
    CI_MULTIPLIER = 1.5

    # Weights for WMA: more recent days get higher weight
    # Last 7 days get exponentially more weight than older data
    WMA_ALPHA = 0.3  # exponential smoothing factor

    def forecast(
        self,
        daily_sales: List[DailySale],
        forecast_days: int = 30,
    ) -> SalesForecastResponse:
        """
        Main entry point. Takes historical daily sales and returns
        a forecast for the next `forecast_days` days.
        """
        logger.info(f"Starting sales forecast: {len(daily_sales)} historical days, {forecast_days} forecast days")

        # ── 1. Parse and sort historical data ──────────────────────
        records = sorted(daily_sales, key=lambda x: x.date)
        revenues = np.array([r.revenue for r in records], dtype=float)
        counts   = np.array([r.count   for r in records], dtype=int)
        dates    = [r.date for r in records]

        n = len(revenues)

        # ── 2. Fill gaps in data (0 revenue days might be missing) ─
        revenues = self._fill_zeros(revenues)

        # ── 3. Detect weekday seasonality factors ──────────────────
        weekday_factors = self._compute_weekday_factors(dates, revenues)

        # ── 4. Fit linear regression on revenue trend ──────────────
        X = np.arange(n).reshape(-1, 1)
        slope, intercept = self._fit_linear(X.flatten(), revenues)
        logger.info(f"Linear trend — slope: {slope:.2f}, intercept: {intercept:.2f}")

        # ── 5. Exponential smoothing on counts ─────────────────────
        smoothed_count = self._exp_smooth(counts.astype(float))

        # ── 6. Compute residual std dev for confidence bands ───────
        trend_values = slope * X.flatten() + intercept
        residuals    = revenues - trend_values
        std_dev      = float(np.std(residuals)) if len(residuals) > 1 else revenues.mean() * 0.1

        # ── 7. Generate last date for sequence ─────────────────────
        last_date = datetime.strptime(dates[-1], "%Y-%m-%d")

        # ── 8. Build predictions ───────────────────────────────────
        predictions: List[DayPrediction] = []

        for i in range(1, forecast_days + 1):
            future_idx  = n + i - 1
            future_date = last_date + timedelta(days=i)
            weekday     = future_date.weekday()  # 0=Mon, 6=Sun

            # Trend projection
            trend_rev = slope * future_idx + intercept

            # Apply weekday seasonality
            seasonality = weekday_factors.get(weekday, 1.0)
            predicted_rev = max(0.0, trend_rev * seasonality)

            # Count prediction: scale by revenue ratio
            avg_rev   = revenues[-7:].mean() if n >= 7 else revenues.mean()
            count_ratio = predicted_rev / avg_rev if avg_rev > 0 else 1.0
            predicted_count = max(0, int(round(smoothed_count * count_ratio)))

            # Confidence bands widen as we forecast further out
            uncertainty = std_dev * self.CI_MULTIPLIER * (1 + i * 0.02)
            lower = max(0.0, predicted_rev - uncertainty)
            upper = predicted_rev + uncertainty

            # Confidence score: decays with forecast horizon
            confidence = max(0.5, 1.0 - (i / forecast_days) * 0.4)

            predictions.append(DayPrediction(
                date               = future_date.strftime("%Y-%m-%d"),
                predicted_revenue  = round(predicted_rev, 2),
                predicted_count    = predicted_count,
                lower_bound        = round(lower, 2),
                upper_bound        = round(upper, 2),
                confidence         = round(confidence, 3),
            ))

        # ── 9. Summarise trend ─────────────────────────────────────
        recent_avg  = float(revenues[-7:].mean()) if n >= 7 else float(revenues.mean())
        forecast_avg = float(np.mean([p.predicted_revenue for p in predictions[:7]]))
        trend_pct    = ((forecast_avg - recent_avg) / recent_avg * 100) if recent_avg > 0 else 0.0

        if trend_pct > 5:
            trend = "growing"
        elif trend_pct < -5:
            trend = "declining"
        else:
            trend = "stable"

        logger.info(f"Forecast complete — trend: {trend} ({trend_pct:.1f}%)")

        return SalesForecastResponse(
            status             = "success",
            forecast_days      = forecast_days,
            predictions        = predictions,
            trend              = trend,
            trend_pct          = round(trend_pct, 2),
            avg_daily_revenue  = round(recent_avg, 2),
            model_used         = "LinearRegression + WeightedMovingAverage + WeekdaySeasonality",
        )

    # ── Private helpers ────────────────────────────────────────────

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Ordinary least squares: y = slope*X + intercept."""
        n = len(X)
        if n < 2:
            return 0.0, float(y.mean())
        x_mean = X.mean()
        y_mean = y.mean()
        numerator   = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        slope     = float(numerator / denominator) if denominator != 0 else 0.0
        intercept = float(y_mean - slope * x_mean)
        return slope, intercept

    def _exp_smooth(self, series: np.ndarray) -> float:
        """Exponential smoothing — returns the smoothed last value."""
        smoothed = series[0]
        for val in series[1:]:
            smoothed = self.WMA_ALPHA * val + (1 - self.WMA_ALPHA) * smoothed
        return float(smoothed)

    def _fill_zeros(self, arr: np.ndarray) -> np.ndarray:
        """Replace 0-revenue days with local median to avoid skew."""
        median = np.median(arr[arr > 0]) if np.any(arr > 0) else 1.0
        arr = arr.copy()
        arr[arr == 0] = median * 0.3  # low but not zero
        return arr

    def _compute_weekday_factors(
        self,
        dates: List[str],
        revenues: np.ndarray,
    ) -> dict:
        """
        Compute average revenue per weekday relative to overall mean.
        Returns dict {weekday_int: multiplier}.
        Weekend days typically earn more for retail.
        """
        weekday_totals = {i: [] for i in range(7)}
        overall_mean   = revenues.mean()

        for date_str, rev in zip(dates, revenues):
            wd = datetime.strptime(date_str, "%Y-%m-%d").weekday()
            weekday_totals[wd].append(rev)

        factors = {}
        for wd, vals in weekday_totals.items():
            if vals and overall_mean > 0:
                factors[wd] = float(np.mean(vals)) / overall_mean
            else:
                factors[wd] = 1.0

        return factors