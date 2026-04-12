"""
services/customer_insights.py
================================
Customer segmentation using simplified RFM (Recency, Frequency, Monetary) analysis.

RFM Scoring (each dimension 1-5):
  - Recency:   5 = purchased recently, 1 = long time ago
  - Frequency: 5 = buys very often,    1 = rarely
  - Monetary:  5 = high spender,       1 = low spender

Segments mapped from RFM total score (3-15):
  ┌────────────┬───────────────┬────────────────────────────────────────┐
  │ Segment    │ Score Range   │ Description                            │
  ├────────────┼───────────────┼────────────────────────────────────────┤
  │ champion   │ 13-15         │ Bought recently, often, spends most    │
  │ loyal      │ 10-12         │ Buys regularly, good value             │
  │ potential  │  7-9          │ Recent but low frequency               │
  │ at_risk    │  4-6          │ Was good, hasn't bought in a while     │
  │ lost       │  3            │ Low on all dimensions                  │
  └────────────┴───────────────┴────────────────────────────────────────┘
"""

import numpy as np
import logging
from datetime import datetime, date
from typing import List, Optional
from schemas.models import (
    CustomerEntry, CustomerInsight, RFMScore, CustomerInsightsResponse
)

logger = logging.getLogger(__name__)

# Segment metadata
SEGMENT_META = {
    "champion": {
        "label":          "🏆 Champion",
        "recommendation": "Reward them. They are your best customers — offer loyalty perks or exclusive deals.",
    },
    "loyal": {
        "label":          "⭐ Loyal Customer",
        "recommendation": "Upsell higher-value products. Send them personalised offers to maintain loyalty.",
    },
    "potential": {
        "label":          "🌱 Potential Loyalist",
        "recommendation": "Engage with targeted promotions. Nudge them to buy more frequently.",
    },
    "at_risk": {
        "label":          "⚠️ At Risk",
        "recommendation": "Send a win-back campaign with a discount to re-engage before they churn.",
    },
    "lost": {
        "label":          "💤 Lost Customer",
        "recommendation": "Attempt a re-engagement campaign with a strong offer. Low priority if unresponsive.",
    },
}


class CustomerInsightService:

    def analyse(
        self,
        customers: List[CustomerEntry],
        analysis_date: Optional[str] = None,
    ) -> CustomerInsightsResponse:

        ref_date = (
            datetime.strptime(analysis_date, "%Y-%m-%d").date()
            if analysis_date
            else date.today()
        )

        logger.info(f"Running RFM analysis for {len(customers)} customers, ref date: {ref_date}")

        # ── Step 1: Compute raw RFM metrics per customer ──────────
        raw_metrics = []
        for customer in customers:
            if not customer.transactions:
                continue
            metrics = self._compute_raw_metrics(customer, ref_date)
            if metrics:
                raw_metrics.append((customer, metrics))

        if not raw_metrics:
            return CustomerInsightsResponse(
                status="success",
                total_customers=0,
                segments_summary={},
                customers=[],
                model_used="RFM Segmentation",
            )

        # ── Step 2: Score each dimension (1-5) using percentiles ──
        recencies   = np.array([m["recency_days"]  for _, m in raw_metrics])
        frequencies = np.array([m["frequency"]     for _, m in raw_metrics])
        monetaries  = np.array([m["monetary"]      for _, m in raw_metrics])

        r_scores = self._score_recency(recencies)
        f_scores = self._score_quantile(frequencies, ascending=True)
        m_scores = self._score_quantile(monetaries,  ascending=True)

        # ── Step 3: Build CustomerInsight objects ─────────────────
        insights: List[CustomerInsight] = []
        segment_counts: dict = {}

        for idx, (customer, metrics) in enumerate(raw_metrics):
            r = int(r_scores[idx])
            f = int(f_scores[idx])
            m = int(m_scores[idx])
            total_rfm = r + f + m

            segment = self._map_segment(total_rfm)
            meta    = SEGMENT_META[segment]

            segment_counts[segment] = segment_counts.get(segment, 0) + 1

            insights.append(CustomerInsight(
                customer_id       = customer.customer_id,
                name              = customer.name,
                segment           = segment,
                segment_label     = meta["label"],
                rfm               = RFMScore(
                    recency_score   = r,
                    frequency_score = f,
                    monetary_score  = m,
                    total_rfm       = total_rfm,
                ),
                recency_days      = int(metrics["recency_days"]),
                frequency         = int(metrics["frequency"]),
                monetary          = round(float(metrics["monetary"]), 2),
                avg_order_value   = round(float(metrics["avg_order_value"]), 2),
                purchase_interval = round(float(metrics["purchase_interval"]), 1),
                recommendation    = meta["recommendation"],
            ))

        # Sort by RFM total descending (champions first)
        insights.sort(key=lambda x: x.rfm.total_rfm, reverse=True)

        logger.info(f"RFM complete. Segments: {segment_counts}")

        return CustomerInsightsResponse(
            status           = "success",
            total_customers  = len(insights),
            segments_summary = segment_counts,
            customers        = insights,
            model_used       = "RFM Segmentation (Percentile Scoring)",
        )

    # ── Private helpers ────────────────────────────────────────────

    def _compute_raw_metrics(self, customer: CustomerEntry, ref_date: date) -> Optional[dict]:
        """Compute Recency, Frequency, Monetary raw values."""
        if not customer.transactions:
            return None

        amounts = [t.amount for t in customer.transactions]
        dates   = []
        for t in customer.transactions:
            try:
                dates.append(datetime.strptime(t.date, "%Y-%m-%d").date())
            except ValueError:
                continue

        if not dates:
            return None

        last_date     = max(dates)
        recency_days  = (ref_date - last_date).days
        frequency     = len(customer.transactions)
        monetary      = sum(amounts)
        avg_order     = monetary / frequency if frequency > 0 else 0.0

        # Average days between purchases
        if len(dates) >= 2:
            sorted_dates = sorted(dates)
            gaps = [(sorted_dates[i+1] - sorted_dates[i]).days
                    for i in range(len(sorted_dates) - 1)]
            purchase_interval = float(np.mean(gaps))
        else:
            purchase_interval = float(recency_days)

        return {
            "recency_days":    recency_days,
            "frequency":       frequency,
            "monetary":        monetary,
            "avg_order_value": avg_order,
            "purchase_interval": purchase_interval,
        }

    def _score_recency(self, recency_days: np.ndarray) -> np.ndarray:
        """Lower recency (more recent) = higher score."""
        return self._score_quantile(recency_days, ascending=False)

    def _score_quantile(self, values: np.ndarray, ascending: bool = True) -> np.ndarray:
        """
        Assign 1-5 scores based on percentile rank.
        ascending=True  → higher value = higher score (frequency, monetary)
        ascending=False → lower value  = higher score (recency)
        """
        n = len(values)
        if n == 0:
            return np.array([])

        scores = np.zeros(n, dtype=int)
        percentiles = np.percentile(values, [20, 40, 60, 80])

        for i, v in enumerate(values):
            if ascending:
                if   v <= percentiles[0]: scores[i] = 1
                elif v <= percentiles[1]: scores[i] = 2
                elif v <= percentiles[2]: scores[i] = 3
                elif v <= percentiles[3]: scores[i] = 4
                else:                     scores[i] = 5
            else:
                # Inverted: lower value = better score
                if   v >= percentiles[3]: scores[i] = 1
                elif v >= percentiles[2]: scores[i] = 2
                elif v >= percentiles[1]: scores[i] = 3
                elif v >= percentiles[0]: scores[i] = 4
                else:                     scores[i] = 5

        return scores

    def _map_segment(self, total_rfm: int) -> str:
        if total_rfm >= 13: return "champion"
        if total_rfm >= 10: return "loyal"
        if total_rfm >= 7:  return "potential"
        if total_rfm >= 4:  return "at_risk"
        return "lost"