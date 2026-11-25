import time
import numpy as np
import polars as pl
from pathlib import Path


def generate_ab_dataset(
    n_users: int = 1_000_000,
    seed: int = 42,
    output_path: str = "data/ab_experiment_1m.parquet",
) -> pl.DataFrame:
    """
    Generates a 1M-row realistic A/B testing dataset with:
    - Pre-experiment covariates (for CUPED variance reduction)
    - User segments (device, city_tier, tenure_days)
    - Heterogeneous Treatment Effects (HTE) across segments
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    user_id = np.arange(1, n_users + 1, dtype=np.int64)
    device = rng.choice(["Android", "iOS", "Web"], size=n_users, p=[0.65, 0.20, 0.15])
    city_tier = rng.choice(["Tier-1", "Tier-2", "Tier-3"], size=n_users, p=[0.30, 0.45, 0.25])
    tenure_days = rng.integers(1, 1095, size=n_users)

    # Randomized 50/50 split
    variant = rng.integers(0, 2, size=n_users, dtype=np.int8)

    # Baseline pre-experiment metric (correlated with device, tier, and tenure)
    device_base = np.where(device == "iOS", 45.0, np.where(device == "Web", 35.0, 25.0))
    tier_base = np.where(city_tier == "Tier-1", 15.0, np.where(city_tier == "Tier-2", 8.0, 3.0))
    noise_pre = rng.lognormal(mean=2.0, sigma=0.6, size=n_users)
    pre_metric = np.round(device_base + tier_base + 0.02 * tenure_days + noise_pre, 2)

    # True Heterogeneous Treatment Effect (HTE):
    # iOS + Tier-1 users get +6.5 lift; Web gets +2.5 lift; Android gets +0.5 lift
    true_lift = np.where(
        (device == "iOS") & (city_tier == "Tier-1"),
        6.5,
        np.where(device == "Web", 2.5, 0.5),
    )

    # Post-experiment outcome strongly correlated with pre_metric
    noise_post = rng.normal(loc=0.0, scale=12.0, size=n_users)
    post_metric = np.maximum(
        0.0,
        np.round(0.85 * pre_metric + (variant * true_lift) + noise_post, 2),
    )

    df = pl.DataFrame(
        {
            "user_id": user_id,
            "device": device,
            "city_tier": city_tier,
            "tenure_days": tenure_days,
            "variant": variant,
            "pre_metric": pre_metric,
            "post_metric": post_metric,
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")

    elapsed = time.perf_counter() - start_time
    print(f"[SUCCESS] Generated {n_users:,} rows in {elapsed:.2f}s -> {output_path}")
    return df


if __name__ == "__main__":
    df = generate_ab_dataset()
    print(f"\nDimensions: {df.shape}")
    print("\nSchema & Data Types:")
    print(df.schema)
    print("\nFirst 5 Rows:")
    print(df.head(5))
    print("\nNull Count per Column:")
    print(df.null_count())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nVariant Balance Check:")
    print(
        df.group_by("variant").agg(
            pl.len().alias("users"),
            pl.col("pre_metric").mean().round(4).alias("avg_pre"),
            pl.col("post_metric").mean().round(4).alias("avg_post"),
            pl.col("post_metric").std().round(4).alias("std_post"),
        )
    )
