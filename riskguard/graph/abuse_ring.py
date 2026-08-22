"""
Graph-Based Abuse-Ring & Syndicate Sentinel
Detects coordinated fraud and return-abuse rings sharing hardware fingerprints,
fuzzy delivery addresses, IPs, and payment instruments.
"""

import re
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd


def normalize_address(address_str: str) -> str:
    if not address_str or not isinstance(address_str, str):
        return "unknown_addr"

    text = address_str.lower().strip()

    replacements = {
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bboulevard\b": "blvd",
        r"\broad\b": "rd",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bapartment\b": "apt",
        r"\bsuite\b": "apt",
        r"\bste\b": "apt",
        r"\bunit\b": "apt",
        r"#\s*": "apt "
    }
    for pat, rep in replacements.items():
        text = re.sub(pat, rep, text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(\d+)\s+([a-z])\b", r"\1\2", text)
    text = re.sub(r"\b(apt)\s+(apt)\b", r"\1", text)
    tokens = [t for t in text.split() if t]
    return "_".join(tokens)


class DisjointSetUnion:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = defaultdict(int)

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


class AbuseRingSentinel:
    def __init__(self):
        self.dsu = DisjointSetUnion()
        self.entity_types: Dict[str, str] = {}
        self.graph_edges: List[Tuple[str, str, str]] = []
        self.transactions: List[Dict[str, Any]] = []

    def ingest_transaction(
            self,
            order_id: str,
            account_id: str,
            device_id: str,
            ip_address: str,
            card_hash: str,
            shipping_address: str,
            order_amount: float,
            timestamp: str,
            is_chargeback: int = 0,
            is_return_abuse: int = 0
    ) -> None:
        norm_addr = "ADDR:" + normalize_address(shipping_address)
        acc_node = "ACC:" + account_id
        dev_node = "DEV:" + device_id
        ip_node = "IP:" + ip_address
        card_node = "CARD:" + card_hash

        self.entity_types[acc_node] = "ACCOUNT"
        self.entity_types[dev_node] = "DEVICE"
        self.entity_types[ip_node] = "IP"
        self.entity_types[card_node] = "CARD"
        self.entity_types[norm_addr] = "ADDRESS"

        for linked_node, edge_type in [
            (dev_node, "USED_DEVICE"),
            (ip_node, "LOGGED_IP"),
            (card_node, "PAID_WITH_CARD"),
            (norm_addr, "SHIPPED_TO_ADDRESS")
        ]:
            self.dsu.union(acc_node, linked_node)
            self.graph_edges.append((acc_node, linked_node, edge_type))

        self.transactions.append({
            "order_id": order_id,
            "account_node": acc_node,
            "device_node": dev_node,
            "ip_node": ip_node,
            "card_node": card_node,
            "address_node": norm_addr,
            "amount": order_amount,
            "timestamp": timestamp,
            "is_chargeback": is_chargeback,
            "is_return_abuse": is_return_abuse
        })

    def detect_rings(self, min_cluster_accounts: int = 2) -> pd.DataFrame:
        clusters: Dict[str, Set[str]] = defaultdict(set)
        for node in self.entity_types:
            root = self.dsu.find(node)
            clusters[root].add(node)

        ring_records = []
        ring_counter = 1

        for root, members in clusters.items():
            accounts = [m for m in members if self.entity_types[m] == "ACCOUNT"]
            devices = [m for m in members if self.entity_types[m] == "DEVICE"]
            ips = [m for m in members if self.entity_types[m] == "IP"]
            cards = [m for m in members if self.entity_types[m] == "CARD"]
            addresses = [m for m in members if self.entity_types[m] == "ADDRESS"]

            num_accounts = len(accounts)
            if num_accounts < min_cluster_accounts:
                continue

            member_set = set(accounts)
            ring_txs = [t for t in self.transactions if t["account_node"] in member_set]
            total_orders = len(ring_txs)
            total_spend = sum(t["amount"] for t in ring_txs)
            total_chargebacks = sum(t["is_chargeback"] for t in ring_txs)
            total_returns = sum(t["is_return_abuse"] for t in ring_txs)

            risk_score = 30.0
            risk_score += min(35.0, (num_accounts - 1) * 12.0)
            if len(devices) < num_accounts:
                risk_score += 15.0
            if len(cards) < num_accounts:
                risk_score += 15.0
            if total_chargebacks > 0:
                risk_score += 20.0
            if total_returns > 1:
                risk_score += 15.0
            risk_score = min(100.0, risk_score)

            tier = "CONFIRMED ABUSE RING" if risk_score >= 80 else (
                "SUSPECTED SYNDICATE" if risk_score >= 50 else "BENIGN CLUSTER (HOUSEHOLD)")

            ring_records.append({
                "ring_id": f"RING-{ring_counter:03d}",
                "syndicate_tier": tier,
                "syndicate_risk_score": round(risk_score, 1),
                "num_accounts": num_accounts,
                "num_devices": len(devices),
                "num_cards": len(cards),
                "num_ips": len(ips),
                "num_addresses": len(addresses),
                "total_orders": total_orders,
                "total_spend": round(total_spend, 2),
                "chargebacks": total_chargebacks,
                "return_abuses": total_returns,
                "sample_accounts": ", ".join([a.replace("ACC:", "") for a in accounts[:4]]),
                "shared_devices": ", ".join([d.replace("DEV:", "") for d in devices[:2]]),
                "shared_address": addresses[0].replace("ADDR:", "").replace("_", " ").title() if addresses else "N/A"
            })
            ring_counter += 1

        df = pd.DataFrame(ring_records)
        if not df.empty:
            df = df.sort_values("syndicate_risk_score", ascending=False).reset_index(drop=True)
        return df


def generate_mock_syndicate_graph() -> AbuseRingSentinel:
    sentinel = AbuseRingSentinel()
    syn1_accounts = ["CUST-9011", "CUST-9012", "CUST-9013", "CUST-9014", "CUST-9015"]
    for i, acc in enumerate(syn1_accounts):
        sentinel.ingest_transaction(
            order_id=f"ORD-SYN1-{i + 1}",
            account_id=acc,
            device_id="DEV-EMULATOR-8819",
            ip_address="194.26.29.11",
            card_hash="CARD-HASH-7721-VISA",
            shipping_address=f"742 Evergreen Terrace Apt #{i + 1}B, Springfield, OR 97477",
            order_amount=699.99,
            timestamp=f"2026-08-1{i} 14:00:00",
            is_chargeback=1 if i == 0 else 0,
            is_return_abuse=1 if i > 0 else 0
        )

    syn2_accounts = ["CUST-8401", "CUST-8402", "CUST-8403"]
    for i, acc in enumerate(syn2_accounts):
        sentinel.ingest_transaction(
            order_id=f"ORD-SYN2-{i + 1}",
            account_id=acc,
            device_id=f"DEV-MOBILE-{i + 100}",
            ip_address="198.51.100.99",
            card_hash="CARD-HASH-3391-MC",
            shipping_address="500 West Madison St Suite 2100 Chicago IL 60661",
            order_amount=340.50,
            timestamp=f"2026-08-1{i + 2} 11:30:00",
            is_chargeback=0,
            is_return_abuse=1
        )

    sentinel.ingest_transaction(
        order_id="ORD-BENIGN-1",
        account_id="CUST-3001",
        device_id="DEV-MACBOOK-001",
        ip_address="73.189.44.12",
        card_hash="CARD-HASH-1100-AMEX",
        shipping_address="104 Oak Ridge Way, Austin, TX 78701",
        order_amount=120.00,
        timestamp="2026-08-10 09:00:00",
        is_chargeback=0,
        is_return_abuse=0
    )
    sentinel.ingest_transaction(
        order_id="ORD-BENIGN-2",
        account_id="CUST-3002",
        device_id="DEV-IPHONE-002",
        ip_address="73.189.44.12",
        card_hash="CARD-HASH-1101-VISA",
        shipping_address="104 Oak Ridge Way, Austin, TX 78701",
        order_amount=85.00,
        timestamp="2026-08-12 18:00:00",
        is_chargeback=0,
        is_return_abuse=0
    )
    return sentinel