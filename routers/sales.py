"""
routers/sales.py
================
POST /predict-sales
"""

from fastapi import APIRouter, HTTPException
from schemas.models import SalesForecastRequest, SalesForecastResponse
from services.sales_forecast import SalesForecastService
import logging

logger = router = APIRouter()
router = APIRouter()
service = SalesForecastService()


@router.post("", response_model=SalesForecastResponse)
def predict_sales(payload: SalesForecastRequest):
    """
    Forecast daily revenue and transaction count for the next N days.

    **Request body:**
    - `daily_sales`: list of historical daily sales (min 7 entries)
    - `forecast_days`: number of days to forecast (7–90, default 30)

    **Returns:**
    - Day-by-day predictions with confidence intervals
    - Overall trend direction and percentage
    """
    try:
        if len(payload.daily_sales) < 7:
            raise HTTPException(
                status_code=422,
                detail="At least 7 days of historical data required."
            )
        result = service.forecast(
            daily_sales   = payload.daily_sales,
            forecast_days = payload.forecast_days,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"/predict-sales error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")