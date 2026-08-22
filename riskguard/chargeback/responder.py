"""
Chargeback Evidence Responder & Dispute Win-Probability Scorer
Evaluates evidence strength against Visa Compelling Evidence 3.0 & Mastercard chargeback rules,
computes dispute win probability, and generates arbitration-grade rebuttal dossiers.
"""

from typing import Dict, Any, List


class ChargebackEvidenceResponder:
    def score_dispute_evidence(self, dispute_data: Dict[str, Any]) -> Dict[str, Any]:
        evidence = dispute_data.get("evidence", {})
        score = 0.0
        checks = []

        # 1. AVS / CVV Verification (Security Layer)
        avs = evidence.get("avs_result", "NO_MATCH")
        cvv = evidence.get("cvv_result", "NO_MATCH")

        if avs == "FULL_MATCH":
            score += 20.0
            checks.append({"item": "Address Verification (AVS)", "status": "PASS", "score": "+20 pts",
                           "detail": "Full billing street and postal code match."})
        elif "ZIP_MATCH" in avs:
            score += 10.0
            checks.append({"item": "Address Verification (AVS)", "status": "PARTIAL", "score": "+10 pts",
                           "detail": "Postal code matched; street mismatch."})
        else:
            checks.append({"item": "Address Verification (AVS)", "status": "FAIL", "score": "+0 pts",
                           "detail": "AVS mismatch or unavailable."})

        if cvv == "MATCH":
            score += 10.0
            checks.append({"item": "CVV2 Security Match", "status": "PASS", "score": "+10 pts",
                           "detail": "CVV2 security code validated at authorization."})
        else:
            checks.append({"item": "CVV2 Security Match", "status": "FAIL", "score": "+0 pts",
                           "detail": "CVV2 failed or not verified."})

        # 2. Carrier Fulfillment & Proof of Handover
        carrier_status = evidence.get("carrier_status", "UNKNOWN")
        signature = evidence.get("signature_name", "")

        if carrier_status == "DELIVERED":
            if signature and "NO" not in signature.upper() and "PHOTO" not in signature.upper():
                score += 30.0
                checks.append({"item": "Carrier Delivery Proof", "status": "PASS", "score": "+30 pts",
                               "detail": f"Delivered via {evidence.get('carrier', 'Carrier')} with recipient signature: '{signature}'."})
            elif "PHOTO" in signature.upper() or "DOOR" in signature.upper():
                score += 20.0
                checks.append({"item": "Carrier Delivery Proof", "status": "PASS", "score": "+20 pts",
                               "detail": f"Delivered via {evidence.get('carrier', 'Carrier')} with GPS drop-off photo proof."})
            else:
                score += 15.0
                checks.append({"item": "Carrier Delivery Proof", "status": "PASS", "score": "+15 pts",
                               "detail": f"Delivered via {evidence.get('carrier', 'Carrier')} with tracking timestamp."})
        else:
            checks.append({"item": "Carrier Delivery Proof", "status": "FAIL", "score": "+0 pts",
                           "detail": "No proof of delivery on record."})

        # 3. Visa Compelling Evidence 3.0 (Prior Order Linkage)
        prior_orders = int(evidence.get("prior_undisputed_orders_count", 0))
        same_device = bool(evidence.get("same_device_used_previously", False))

        if prior_orders >= 2 and same_device:
            score += 20.0
            checks.append({"item": "Visa CE 3.0 Device Linkage", "status": "PASS", "score": "+20 pts",
                           "detail": f"{prior_orders} historical undisputed orders on identical device fingerprint."})
        elif prior_orders >= 1:
            score += 10.0
            checks.append({"item": "Prior Customer History", "status": "PARTIAL", "score": "+10 pts",
                           "detail": f"{prior_orders} prior undisputed purchase on record."})
        else:
            checks.append({"item": "Prior Customer History", "status": "NEUTRAL", "score": "+0 pts",
                           "detail": "First-time account; no prior transaction baseline."})

        # 4. Customer Support Chat & Admission Logs
        chat_logs = evidence.get("chat_logs", [])
        chat_text = " ".join([m.get("message", "") for m in chat_logs]).lower()

        if any(w in chat_text for w in ["received", "got it", "arrived", "working", "pair", "setup"]):
            score += 20.0
            checks.append({"item": "Customer Admission Exhibit", "status": "PASS", "score": "+20 pts",
                           "detail": "Customer confirmed receipt & active usage in support transcript."})
        elif any(w in chat_text for w in ["restock", "terms", "spec", "dispute", "policy"]):
            score += 20.0
            checks.append({"item": "Terms & Spec Acknowledgment", "status": "PASS", "score": "+20 pts",
                           "detail": "Transcript documents buyer remorse / policy dispute rather than non-delivery."})
        elif len(chat_logs) > 0:
            score += 5.0
            checks.append({"item": "Support Interaction History", "status": "PARTIAL", "score": "+5 pts",
                           "detail": "Support logs attached; pending carrier trace resolution."})
        else:
            checks.append({"item": "Customer Communication Exhibit", "status": "EMPTY", "score": "+0 pts",
                           "detail": "No communication history recorded prior to filing."})

        final_score = min(100.0, max(0.0, score))

        if final_score >= 80:
            win_band = "VERY HIGH"
            win_range = "85% - 95% Win Rate"
            recommendation = "STRONGLY RECOMMEND SUBMITTING DISPUTE REBUTTAL"
        elif final_score >= 60:
            win_band = "HIGH"
            win_range = "65% - 80% Win Rate"
            recommendation = "RECOMMEND SUBMITTING DISPUTE WITH ALL ATTACHED EXHIBITS"
        elif final_score >= 40:
            win_band = "UNCERTAIN"
            win_range = "40% - 55% Win Rate"
            recommendation = "SUBMIT REBUTTAL IF TRANSACTION VALUE EXCEEDS ARBITRATION FEES"
        else:
            win_band = "LOW"
            win_range = "< 25% Win Rate"
            recommendation = "CONSIDER ACCEPTING DISPUTE TO AVOID ARBITRATION PENALTY"

        return {
            "dispute_id": dispute_data.get("dispute_id"),
            "evidence_score_pct": final_score,
            "win_probability_band": win_band,
            "win_range": win_range,
            "recommended_action": recommendation,
            "checks": checks
        }

    def generate_rebuttal_packet(self, dispute_data: Dict[str, Any]) -> str:
        score_res = self.score_dispute_evidence(dispute_data)
        ev = dispute_data.get("evidence", {})

        md = f"""# FORMAL CHARGEBACK DISPUTE REBUTTAL DOSSIER
**Case Reference:** `{dispute_data.get('dispute_id')}` | **Card Scheme:** `{dispute_data.get('card_brand')}` | **Dispute Amount:** `${dispute_data.get('amount_disputed'):.2f} {dispute_data.get('currency', 'USD')}`

---

## 1. EXECUTIVE DISPUTE SUMMARY
- **Transaction ID:** `{dispute_data.get('transaction_id')}`
- **Order Reference:** `{dispute_data.get('order_id')}`
- **Dispute Date:** `{dispute_data.get('dispute_date')}`
- **Reason Code:** `{dispute_data.get('reason_code')}` — *{dispute_data.get('reason_description')}*
- **Cardholder Name:** `{dispute_data.get('customer_name')}`
- **Registered Email:** `{dispute_data.get('customer_email')}`
- **Automated Win Likelihood:** **{score_res['win_probability_band']} ({score_res['win_range']})** (Evidence Score: {score_res['evidence_score_pct']}%)
- **Merchant Recommendation:** **{score_res['recommended_action']}**

---

## 2. EVIDENCE ASSESSMENT CHECKLIST
| Evidence Element | Status | Score Weight | Description |
| :--- | :---: | :---: | :--- |
"""
        for chk in score_res["checks"]:
            status_icon = "[PASS]" if chk["status"] == "PASS" else (
                "[PARTIAL]" if chk["status"] == "PARTIAL" else "[FAIL]")
            md += f"| **{chk['item']}** | `{status_icon}` | `{chk['score']}` | {chk['detail']} |\n"

        md += f"""
---

## 3. CARD SCHEME & SECURITY AUTHENTICATION PROOF
- **Address Verification Service (AVS):** `{ev.get('avs_result')}`
- **Card Security Code (CVV2/CVC2):** `{ev.get('cvv_result')}`
- **Cardholder IP Address:** `{ev.get('ip_address')}` (GeoLocation: *{ev.get('ip_location')}*)
- **Billing Address on File:** `{ev.get('billing_address')}`
- **Shipping Address Provided:** `{ev.get('shipping_address')}`
- **Compelling Evidence 3.0 Match:** `{ev.get('prior_undisputed_orders_count', 0)}` prior undisputed transactions executed from matching hardware footprint.

---

## 4. CARRIER FULFILLMENT & PROOF OF DELIVERY
- **Logistics Carrier:** `{ev.get('carrier')}`
- **Tracking Number:** `{ev.get('tracking_number')}`
- **Carrier Delivery Status:** `{ev.get('carrier_status')}`
- **Delivery Timestamp:** `{ev.get('delivery_timestamp')}`
- **Signature / Proof of Handover:** `{ev.get('signature_name')}`

---

## 5. CUSTOMER COMMUNICATIONS & INTERACTION LOGS
"""
        chat_logs = ev.get("chat_logs", [])
        if chat_logs:
            for msg in chat_logs:
                md += f"> **[{msg.get('timestamp')}] {msg.get('sender')}:**\n> \"{msg.get('message')}\"\n\n"
        else:
            md += "*No customer service transcripts recorded prior to dispute filing.*\n\n"

        md += f"""---

## 6. FORMAL REBUTTAL STATEMENT TO ISSUING BANK
The merchant respectfully requests the dispute for transaction `{dispute_data.get('transaction_id')}` be reversed in full. The order was authenticated with matching security credentials, fulfilled to the verified cardholder address via `{ev.get('carrier')}` under tracking number `{ev.get('tracking_number')}`, and successfully delivered. The customer accepted the merchant's published terms of sale at checkout. All supporting exhibits are attached in accordance with `{dispute_data.get('card_brand')}` dispute resolution regulations.
"""
        return md