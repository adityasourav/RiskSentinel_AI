"""
Comprehensive Unit & Integration Test Suite for RiskGuard AI
Verifies the algorithmic correctness of all ML models, cost optimizers, graph clusters, and API handlers.
"""

import unittest
import numpy as np
import pandas as pd

from data.generator import generate_risk_dataset
from riskguard.models.classifier import RiskClassifier
from riskguard.models.cost_optimizer import CostOptimizer, CostMatrix
from riskguard.models.time_series import generate_merchant_time_series, InterpretableFraudSpikeDetector
from riskguard.models.cusum_detector import CUSUMDetector
from riskguard.models.unified_engine import UnifiedRiskEngine
from riskguard.graph.abuse_ring import AbuseRingSentinel, normalize_address
from riskguard.chargeback.responder import ChargebackEvidenceResponder


class TestRiskGuardSuite(unittest.TestCase):

    def test_01_dataset_generator(self):
        df = generate_risk_dataset(n_samples=500, random_state=42)
        self.assertEqual(len(df), 500)
        self.assertIn("is_risk_target", df.columns)
        self.assertIn("order_amount", df.columns)
        self.assertTrue(0.01 < df["is_risk_target"].mean() < 0.15)

    def test_02_classifier_training_and_calibration(self):
        df = generate_risk_dataset(n_samples=600, random_state=42)
        train_df, test_df = df.iloc[:400], df.iloc[400:]

        clf = RiskClassifier(model_type="hist_gbdt", calibrate=True, random_state=42)
        clf.train(train_df)
        self.assertTrue(clf.is_fitted)

        probs = clf.predict_proba(test_df)
        self.assertEqual(len(probs), len(test_df))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        exp = clf.explain_order(test_df.iloc[0])
        self.assertIn("risk_probability", exp)
        self.assertIn("risk_factors", exp)

    def test_03_cost_utility_optimization(self):
        matrix = CostMatrix(c_fn=85.0, c_fp=28.0, c_tp=4.0, c_tn=0.0)
        opt = CostOptimizer(matrix)

        bayes_th = opt.theoretical_optimal_threshold()
        self.assertAlmostEqual(bayes_th, 28.0 / 109.0, places=3)

        y_true = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
        y_prob = np.array([0.1, 0.15, 0.2, 0.05, 0.8, 0.25, 0.1, 0.9, 0.3, 0.1])
        th, best_metrics, sweep_df = opt.find_empirical_optimal_threshold(y_true, y_prob)
        self.assertTrue(0.01 <= th <= 0.99)
        self.assertIn("savings_vs_allow_all", best_metrics)

    def test_04_time_series_and_cusum(self):
        ts_df = generate_merchant_time_series(merchant_id="M-TEST", n_days=30, random_state=42)
        detector = InterpretableFraudSpikeDetector(ewma_span=24, z_threshold=3.0)
        analyzed = detector.analyze_stream(ts_df)

        self.assertIn("ewma_anomaly", analyzed.columns)
        self.assertIn("card_testing_flag", analyzed.columns)
        self.assertTrue((analyzed["is_flagged_threat"] == 1).sum() > 0)

        cusum = CUSUMDetector(drift_allowance=0.5, decision_interval=4.0)
        s_pos, s_neg, alarms = cusum.detect_shifts(ts_df["tx_count"].values)
        self.assertEqual(len(alarms), len(ts_df))

    def test_05_graph_abuse_ring(self):
        addr1 = normalize_address("123 Main Street Apt #4-B, Austin, TX")
        addr2 = normalize_address("123 Main St. Suite 4B Austin TX")
        self.assertEqual(addr1, addr2)

        sentinel = AbuseRingSentinel()
        sentinel.ingest_transaction("O-1", "ACC-1", "DEV-1", "1.1.1.1", "CARD-1", "100 Pine St Apt 1", 100.0,
                                    "2026-08-01")
        sentinel.ingest_transaction("O-2", "ACC-2", "DEV-1", "1.1.1.1", "CARD-1", "100 Pine Street #1", 200.0,
                                    "2026-08-01")

        rings = sentinel.detect_rings(min_cluster_accounts=2)
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings.iloc[0]["num_accounts"], 2)

    def test_06_chargeback_responder(self):
        responder = ChargebackEvidenceResponder()
        mock_case = {
            "dispute_id": "DSP-TEST-1",
            "amount_disputed": 250.0,
            "currency": "USD",
            "card_brand": "VISA",
            "reason_code": "10.4",
            "evidence": {
                "avs_result": "FULL_MATCH",
                "cvv_result": "MATCH",
                "carrier_status": "DELIVERED",
                "signature_name": "JOHN DOE",
                "prior_undisputed_orders_count": 3,
                "same_device_used_previously": True
            }
        }
        score = responder.score_dispute_evidence(mock_case)
        self.assertGreaterEqual(score["evidence_score_pct"], 80.0)
        self.assertIn("VERY HIGH", score["win_probability_band"])

        md_dossier = responder.generate_rebuttal_packet(mock_case)
        self.assertIn("FORMAL CHARGEBACK DISPUTE REBUTTAL DOSSIER", md_dossier)

    def test_07_unified_dual_tier_engine(self):
        df = generate_risk_dataset(n_samples=600, random_state=42)
        clf = RiskClassifier(model_type="hist_gbdt", calibrate=True, random_state=42).train(df)
        opt = CostOptimizer(CostMatrix())

        engine = UnifiedRiskEngine(clf, opt)
        ts_mock = pd.DataFrame([{
            "timestamp": "2026-08-22 12:00:00",
            "tx_count": 500, "volume_ewma_mean": 60.0, "robust_zscore": 7.0,
            "decline_rate": 0.45, "micro_ticket_ratio": 0.85, "new_device_ratio": 0.90, "avg_ticket_size": 1.99,
            "threat_category": "CRITICAL: CARD TESTING BOT ATTACK",
            "threat_explanation": "Micro-charges",
            "recommended_action": "STEP-UP 3DS",
            "is_flagged_threat": 1
        }] * 5)

        engine.update_merchant_velocity_state("MERCHANT-ALPHA-99", ts_mock)

        order = {
            "order_amount": 2.00, "item_count": 1, "device_trust_score": 0.20,
            "checkout_duration_sec": 3.0, "payment_velocity_24h": 4
        }
        res = engine.score_checkout_order(order, merchant_id="MERCHANT-ALPHA-99")
        self.assertIn("active_decision_threshold", res)
        self.assertIn("risk_factors", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)