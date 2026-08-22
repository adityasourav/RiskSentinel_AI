"""
Synthetic E-Commerce Order & Risk Dataset Generator
Generates realistic transactional records with complex risk signals, realistic correlations,
and class imbalance for rigorous ML model training and evaluation.
"""

import numpy as np
import pandas as pd


def generate_risk_dataset(
        n_samples: int = 15000,
        random_state: int = 42
) -> pd.DataFrame:
    np.random.seed(random_state)

    # 1. Base Customer Demographics & History
    account_age_days = np.random.exponential(scale=250, size=n_samples) + 1
    historical_orders = np.random.poisson(lam=np.clip(account_age_days / 25, 0.5, 30))

    historical_return_rate = np.random.beta(a=1.2, b=7.0, size=n_samples)
    historical_chargebacks = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.965, 0.025, 0.007, 0.003])

    # 2. Order Specifics
    order_amount = np.random.lognormal(mean=4.3, sigma=0.85, size=n_samples)
    order_amount = np.clip(order_amount, 10.0, 3500.0).round(2)

    item_count = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n_samples,
                                  p=[0.45, 0.25, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01])
    distinct_categories = np.clip(np.random.poisson(lam=1.4, size=n_samples) + 1, 1, item_count)

    high_resale_ratio = np.random.beta(a=1.0, b=3.5, size=n_samples)
    wardrobing_flag = np.random.binomial(n=1, p=np.where(item_count > 1, 0.16, 0.0))
    discount_percentage = np.random.choice([0.0, 0.10, 0.15, 0.20, 0.30, 0.50], size=n_samples,
                                           p=[0.42, 0.24, 0.15, 0.10, 0.06, 0.03])

    # 3. Behavioral & Device Signals
    checkout_duration_sec = np.random.lognormal(mean=3.9, sigma=0.75, size=n_samples)
    checkout_duration_sec = np.clip(checkout_duration_sec, 2.5, 1200.0).round(1)

    device_trust_score = np.random.beta(a=7.0, b=1.5, size=n_samples)
    ip_distance_to_billing_km = np.random.exponential(scale=40, size=n_samples)
    delivery_address_changes_30d = np.random.poisson(lam=0.20, size=n_samples)

    billing_shipping_mismatch = np.random.binomial(n=1, p=0.07, size=n_samples)
    card_is_prepaid = np.random.binomial(n=1, p=0.08, size=n_samples)
    card_country_mismatch = np.random.binomial(n=1, p=0.035, size=n_samples)
    payment_velocity_24h = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples,
                                            p=[0.82, 0.11, 0.04, 0.015, 0.01, 0.005])

    # 4. Latent Risk Formulation (~6.5% prevalence)
    latent_score = (
            -3.6
            + 2.8 * (historical_chargebacks > 0)
            + 2.6 * (wardrobing_flag == 1) * (historical_return_rate > 0.25)
            + 2.5 * (order_amount > 280) * (high_resale_ratio > 0.55) * (account_age_days < 45)
            + 2.2 * (payment_velocity_24h >= 3)
            + 2.0 * (device_trust_score < 0.35)
            + 1.8 * card_is_prepaid * (ip_distance_to_billing_km > 80)
            + 1.6 * card_country_mismatch
            + 1.4 * billing_shipping_mismatch * (delivery_address_changes_30d >= 1)
            + 1.3 * (checkout_duration_sec < 6.0) * (item_count > 2)
            + 1.5 * (historical_return_rate > 0.40)
            + 1.1 * (discount_percentage >= 0.30) * (account_age_days < 20)
            + np.random.normal(0, 0.3, size=n_samples)
    )

    prob_risk = 1.0 / (1.0 + np.exp(-latent_score))
    is_risk_target = np.random.binomial(n=1, p=prob_risk)

    is_chargeback = np.zeros(n_samples, dtype=int)
    is_return_abuse = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        if is_risk_target[i] == 1:
            chargeback_weight = (
                    0.3 +
                    0.3 * (card_is_prepaid[i] or card_country_mismatch[i]) +
                    0.2 * (device_trust_score[i] < 0.4)
            )
            if np.random.rand() < min(0.75, chargeback_weight):
                is_chargeback[i] = 1
            else:
                is_return_abuse[i] = 1

    gross_margin_rate = 0.35
    margin_amount = (order_amount * gross_margin_rate).round(2)

    loss_amount = np.where(
        is_chargeback == 1,
        order_amount + 25.0,
        np.where(
            is_return_abuse == 1,
            12.0 + (order_amount * 0.30) + 8.0,
            0.0
        )
    ).round(2)

    base_time = pd.Timestamp("2026-06-01 00:00:00")
    random_seconds = np.random.randint(0, 90 * 86400, size=n_samples)
    timestamps = [base_time + pd.Timedelta(seconds=int(s)) for s in random_seconds]

    df = pd.DataFrame({
        "order_id": [f"ORD-{i + 100001}" for i in range(n_samples)],
        "customer_id": [f"CUST-{np.random.randint(1000, 7000):05d}" for _ in range(n_samples)],
        "timestamp": timestamps,
        "order_amount": order_amount,
        "item_count": item_count,
        "distinct_categories": distinct_categories,
        "high_resale_ratio": high_resale_ratio.round(3),
        "wardrobing_flag": wardrobing_flag,
        "discount_percentage": discount_percentage,
        "account_age_days": account_age_days.round(1),
        "historical_orders": historical_orders,
        "historical_return_rate": historical_return_rate.round(3),
        "historical_chargebacks": historical_chargebacks,
        "delivery_address_changes_30d": delivery_address_changes_30d,
        "billing_shipping_mismatch": billing_shipping_mismatch,
        "ip_distance_to_billing_km": ip_distance_to_billing_km.round(1),
        "device_trust_score": device_trust_score.round(3),
        "checkout_duration_sec": checkout_duration_sec,
        "payment_velocity_24h": payment_velocity_24h,
        "card_is_prepaid": card_is_prepaid,
        "card_country_mismatch": card_country_mismatch,
        "margin_amount": margin_amount,
        "loss_amount": loss_amount,
        "is_return_abuse": is_return_abuse,
        "is_chargeback": is_chargeback,
        "is_risk_target": is_risk_target
    })

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_risk_dataset(n_samples=15000)
    out_path = "data/orders_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated dataset with {len(df)} orders saved to {out_path}.")
    print(f"Risk target prevalence: {df['is_risk_target'].mean() * 100:.2f}%")