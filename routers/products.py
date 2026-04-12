"""
routers/products.py
====================
POST /predict-products
"""

from fastapi import APIRouter, HTTPException
from schemas.models import ProductForecastRequest, ProductForecastResponse
from services.product_forecast import ProductForecastService
import logging

router = APIRouter()
service = ProductForecastService()


@router.post("", response_model=ProductForecastResponse)
def predict_products(payload: ProductForecastRequest):
    """
    Rank products by predicted demand for the next N days.

    **Request body:**
    - `product_sales`: list of products with their daily sales history
    - `forecast_days`: forecast window (default 30)

    **Returns:**
    - Products ranked by predicted quantity sold
    - Trend direction per product (rising/stable/falling)
    """
    try:
        if not payload.product_sales:
            raise HTTPException(status_code=422, detail="No product data provided.")

        result = service.forecast(
            product_sales = payload.product_sales,
            forecast_days = payload.forecast_days,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"/predict-products error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Product forecast failed: {str(e)}")