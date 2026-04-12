# AI-Powered Sales Analytics — System Architecture

## A. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Admin)                          │
│                   /analytics/* (CI4 Views)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              CodeIgniter 4 (PHP) — POS Backend                  │
│                                                                 │
│  AnalyticsController                                            │
│  ├── index()        → dashboard view                           │
│  ├── salesForecast()→ calls AI /predict-sales                  │
│  ├── productForecast()→ calls AI /predict-products             │
│  └── customerInsights()→ calls AI /customer-insights           │
│                                                                 │
│  AiService (CI4 Service class)                                  │
│  └── uses CI4 HTTP Client → POST to FastAPI                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST (JSON)
                           │ Internal network / localhost
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python FastAPI — AI Microservice                    │
│              http://localhost:8001                               │
│                                                                 │
│  Routers                                                        │
│  ├── POST /predict-sales       → SalesForecastService           │
│  ├── POST /predict-products    → ProductForecastService         │
│  └── POST /customer-insights   → CustomerInsightService         │
│                                                                 │
│  ML Models (scikit-learn + numpy)                               │
│  ├── Linear Regression (sales trend)                            │
│  ├── Weighted Moving Average (product demand)                   │
│  └── RFM Scoring (customer value)                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Read
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MySQL Database                              │
│   transactions, transaction_items, products, customers          │
│   + ai_predictions (cache table)                                │
└─────────────────────────────────────────────────────────────────┘

## B. API FLOW

1. Admin visits /analytics
2. CI4 AnalyticsController queries MySQL for historical data
3. CI4 sends JSON payload to FastAPI endpoint
4. FastAPI preprocesses data → runs ML model → returns predictions
5. CI4 caches result in ai_predictions table
6. CI4 renders view with chart data

## C. DATA FLOW (per feature)

### Sales Prediction
CI4 → { daily_sales: [{date, revenue, count}] } → FastAPI
FastAPI → { predictions: [{date, predicted_revenue, confidence}] } → CI4

### Product Forecast  
CI4 → { product_sales: [{product_id, name, daily_qty: [...]}] } → FastAPI
FastAPI → { rankings: [{product_id, name, predicted_qty, trend}] } → CI4

### Customer Insights
CI4 → { customers: [{customer_id, transactions: [{date, amount}]}] } → FastAPI
FastAPI → { segments: [{customer_id, rfm_score, segment, insights}] } → CI4
```