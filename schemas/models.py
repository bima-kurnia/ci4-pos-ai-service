"""
schemas/models.py
=================
Pydantic schemas for request validation and response serialization.
These define the exact JSON shape CI4 sends and receives.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


# ─────────────────────────────────────────────
# SALES FORECAST
# ─────────────────────────────────────────────

class DailySale(BaseModel):
    """One day of historical sales data."""
    date:    str   = Field(..., example="2024-01-15")
    revenue: float = Field(..., ge=0, example=450000.0)
    count:   int   = Field(..., ge=0, example=12)


class SalesForecastRequest(BaseModel):
    """
    Payload CI4 sends to /predict-sales.
    Contains the last N days of daily sales history.
    """
    daily_sales:    List[DailySale] = Field(..., min_items=7)
    forecast_days:  int             = Field(default=30, ge=7, le=90)

    class Config:
        schema_extra = {
            "example": {
                "daily_sales": [
                    {"date": "2024-01-01", "revenue": 350000, "count": 10},
                    {"date": "2024-01-02", "revenue": 420000, "count": 13},
                ],
                "forecast_days": 30
            }
        }


class DayPrediction(BaseModel):
    date:               str
    predicted_revenue:  float
    predicted_count:    int
    lower_bound:        float
    upper_bound:        float
    confidence:         float   # 0.0 – 1.0


class SalesForecastResponse(BaseModel):
    status:             str
    forecast_days:      int
    predictions:        List[DayPrediction]
    trend:              str     # "growing" | "stable" | "declining"
    trend_pct:          float   # percentage change vs recent average
    avg_daily_revenue:  float   # historical average
    model_used:         str


# ─────────────────────────────────────────────
# PRODUCT FORECAST
# ─────────────────────────────────────────────

class ProductSalesEntry(BaseModel):
    """Aggregated sales data for one product."""
    product_id:   int
    name:         str
    total_qty:    int
    total_revenue: float
    # List of (date_str, qty_sold) tuples for trend analysis
    daily_qty:    List[dict]  # [{date, qty}]


class ProductForecastRequest(BaseModel):
    product_sales:  List[ProductSalesEntry]
    forecast_days:  int = Field(default=30, ge=7, le=90)


class ProductRanking(BaseModel):
    rank:               int
    product_id:         int
    name:               str
    predicted_qty:      int
    predicted_revenue:  float
    trend:              str     # "rising" | "stable" | "falling"
    trend_pct:          float
    confidence:         float


class ProductForecastResponse(BaseModel):
    status:         str
    forecast_days:  int
    rankings:       List[ProductRanking]
    model_used:     str


# ─────────────────────────────────────────────
# CUSTOMER INSIGHTS
# ─────────────────────────────────────────────

class CustomerTransaction(BaseModel):
    date:   str
    amount: float


class CustomerEntry(BaseModel):
    customer_id:  int
    name:         str
    transactions: List[CustomerTransaction]


class CustomerInsightsRequest(BaseModel):
    customers:      List[CustomerEntry]
    analysis_date:  Optional[str] = None   # defaults to today


class RFMScore(BaseModel):
    recency_score:   int   # 1-5
    frequency_score: int   # 1-5
    monetary_score:  int   # 1-5
    total_rfm:       int   # 3-15


class CustomerInsight(BaseModel):
    customer_id:        int
    name:               str
    segment:            str     # "champion" | "loyal" | "at_risk" | "lost" | "new"
    segment_label:      str     # human-readable label
    rfm:                RFMScore
    recency_days:       int     # days since last purchase
    frequency:          int     # total transactions
    monetary:           float   # total spend
    avg_order_value:    float
    purchase_interval:  float   # avg days between purchases
    recommendation:     str     # actionable insight


class CustomerInsightsResponse(BaseModel):
    status:             str
    total_customers:    int
    segments_summary:   dict    # {segment: count}
    customers:          List[CustomerInsight]
    model_used:         str