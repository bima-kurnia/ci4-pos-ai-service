"""
SwiftPOS AI Service
===================
FastAPI microservice providing ML-powered analytics for the POS system.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Endpoints:
    POST /predict-sales       → 7-day and 30-day revenue forecast
    POST /predict-products    → Product demand ranking
    POST /customer-insights   → RFM customer segmentation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import sales, products, customers
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="SwiftPOS AI Service",
    description="AI-powered sales analytics microservice for SwiftPOS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow requests from CI4 backend (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(sales.router,     prefix="/predict-sales",      tags=["Sales Forecast"])
app.include_router(products.router,  prefix="/predict-products",   tags=["Product Forecast"])
app.include_router(customers.router, prefix="/customer-insights",  tags=["Customer Insights"])


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "SwiftPOS AI Service",
        "status":  "running",
        "version": "1.0.0",
        "endpoints": [
            "POST /predict-sales",
            "POST /predict-products",
            "POST /customer-insights",
        ],
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}