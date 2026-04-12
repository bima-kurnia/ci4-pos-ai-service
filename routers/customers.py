"""
routers/customers.py
=====================
POST /customer-insights
"""

from fastapi import APIRouter, HTTPException
from schemas.models import CustomerInsightsRequest, CustomerInsightsResponse
from services.customer_insights import CustomerInsightService
import logging

router = APIRouter()
service = CustomerInsightService()


@router.post("", response_model=CustomerInsightsResponse)
def customer_insights(payload: CustomerInsightsRequest):
    """
    Segment customers using RFM (Recency, Frequency, Monetary) analysis.

    **Request body:**
    - `customers`: list of customers with their transaction history
    - `analysis_date`: reference date for recency calculation (optional, default today)

    **Returns:**
    - Each customer's RFM scores and segment
    - Actionable recommendation per customer
    - Segment summary counts
    """
    try:
        if not payload.customers:
            raise HTTPException(status_code=422, detail="No customer data provided.")

        result = service.analyse(
            customers     = payload.customers,
            analysis_date = payload.analysis_date,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"/customer-insights error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Customer insights failed: {str(e)}")