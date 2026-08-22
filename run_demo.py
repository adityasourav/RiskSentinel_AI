"""
AI Risk Manager — Interactive Quickstart Demo Runner
Executes quick inference across all defense modules and prints key risk decisions.
"""

import os
import json
import pandas as pd
from riskguard.models.classifier import RiskClassifier
from riskguard.models.cost_optimizer import CostOptimizer, CostMatrix
from riskguard.graph.abuse_ring import generate_mock_syndicate_graph
from riskguard.chargeback.responder import ChargebackEvidenceResponder


def main():
    print("=" * 70)
    print("      🛡️ RISKGUARD AI — AI RISK MANAGER DEMO")
    print("=" * 70)

    model_path = "output/risk_classifier.pkl"
    if not os.path.exists(model_path):
        print("Model artifact not found. Please run: python evaluate_all.py first.")
        return

    classifier = RiskClassifier.load(model_path)
    print("\n[1] Scoring Incoming Checkout Transaction:")

    sample_order = {
        "order_amount": 450.0,
        "item_count": 4,
        "distinct_categories": 1,
        "high_resale_ratio": 0.80,
        "wardrobing_flag": 1,
        "discount_percentage": 0.30,
        "account_age_days": 14.0,
        "historical_orders": 2,
        "historical_return_rate": 0.60,
        "historical_chargebacks": 1,
        "delivery_address_changes_30d": 2,
        "billing_shipping_mismatch": 1,
        "ip_distance_to_billing_km": 150.0,
        "device_trust_score": 0.30,
        "checkout_duration_sec": 5.0,
        "payment_velocity_24h": 4,
        "card_is_prepaid": 1,
        "card_country_mismatch": 1
    }

    res = classifier.explain_order(sample_order)
    print(f"  - Risk Probability:    {res['risk_probability'] * 100:.1f}%")
    print(f"  - Classification Tier: {res['risk_tier']}")
    print(f"  - Recommended Action:  {res['recommended_action']}")
    print("  - Risk Attribution Factors:")
    for f in res["risk_factors"]:
        print(f"    * {f}")

    print("\n[2] Abuse-Ring Graph Sentinel Detection:")
    sentinel = generate_mock_syndicate_graph()
    ring_df = sentinel.detect_rings(min_cluster_accounts=2)
    for _, r in ring_df.head(2).iterrows():
        print(
            f"  - {r['ring_id']} [{r['syndicate_tier']}] Score: {r['syndicate_risk_score']}/100 | {r['num_accounts']} Accounts sharing device: {r['shared_devices']}")

    print("\n[3] Chargeback Evidence Responder:")
    with open("data/mock_disputes.json", "r") as f:
        disputes = json.load(f)
    cb_responder = ChargebackEvidenceResponder()
    score_res = cb_responder.score_dispute_evidence(disputes[0])
    print(
        f"  - Case {disputes[0]['dispute_id']}: Evidence Completeness {score_res['evidence_score_pct']}% | Win Likelihood {score_res['win_probability_band']}")
    print(f"  - Action: {score_res['recommended_action']}")

    print("\n" + "=" * 70)
    print("To launch the full interactive web dashboard, run:")
    print("  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()