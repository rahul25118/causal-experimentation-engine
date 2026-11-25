import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def estimate_cate_dml(
    df: pl.DataFrame, n_splits: int = 2, seed: int = 42
) -> tuple[pl.DataFrame, dict]:
    """
    Implements Cross-Fitted Double Machine Learning (R-Learner) to estimate
    Conditional Average Treatment Effects (CATE) without overfitting bias.
    """
    # Prepare feature matrix X (purely pre-experiment covariates)
    X_df = df.select(["pre_metric", "tenure_days", "device", "city_tier"]).to_dummies(
        columns=["device", "city_tier"], drop_first=False
    )
    feature_names = X_df.columns
    X = X_df.to_numpy()
    T = df["variant"].to_numpy().astype(np.float64)
    Y = df["post_metric"].to_numpy().astype(np.float64)

    n = len(df)
    m_hat = np.zeros(n, dtype=np.float64)
    e_hat = np.zeros(n, dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Step 1: Out-of-fold nuisance estimation
    for train_idx, val_idx in kf.split(X):
        # Outcome nuisance model E[Y | X]
        outcome_model = HistGradientBoostingRegressor(
            max_depth=5, max_iter=80, learning_rate=0.08, random_state=seed
        )
        outcome_model.fit(X[train_idx], Y[train_idx])
        m_hat[val_idx] = outcome_model.predict(X[val_idx])

        # Propensity nuisance model P(T=1 | X)
        prop_model = LogisticRegression(max_iter=200)
        prop_model.fit(X[train_idx], T[train_idx])
        e_hat[val_idx] = np.clip(prop_model.predict_proba(X[val_idx])[:, 1], 0.05, 0.95)

    # Step 2: Orthogonalized R-Learner residuals
    y_res = Y - m_hat
    t_res = T - e_hat
    weights = t_res ** 2
    pseudo_outcome = y_res / t_res

    # Step 3: Cross-fitted final CATE stage model tau(X)
    cate_pred = np.zeros(n, dtype=np.float64)
    final_model = None
    for train_idx, val_idx in kf.split(X):
        cate_model = HistGradientBoostingRegressor(
            max_depth=4, max_iter=100, learning_rate=0.05, min_samples_leaf=500, random_state=seed
        )
        cate_model.fit(X[train_idx], pseudo_outcome[train_idx], sample_weight=weights[train_idx])
        cate_pred[val_idx] = cate_model.predict(X[val_idx])
        final_model = cate_model

    # Ground truth CATE for benchmark validation
    device_arr = df["device"].to_numpy()
    tier_arr = df["city_tier"].to_numpy()
    true_cate = np.where(
        (device_arr == "iOS") & (tier_arr == "Tier-1"),
        6.5,
        np.where(device_arr == "Web", 2.5, 0.5),
    )

    rmse = float(np.sqrt(mean_squared_error(true_cate, cate_pred)))
    r2 = float(r2_score(true_cate, cate_pred))

    out_df = df.with_columns(
        [
            pl.Series("true_cate", true_cate),
            pl.Series("predicted_cate", np.round(cate_pred, 4)),
        ]
    )

    metrics = {
        "cate_rmse": round(rmse, 4),
        "cate_r2": round(r2, 4),
        "mean_predicted_ate": round(float(np.mean(cate_pred)), 4),
        "feature_names": feature_names,
    }
    return out_df, metrics


def run_hte_pipeline(parquet_path: str = "data/ab_experiment_1m.parquet") -> None:
    start_time = time.perf_counter()
    df = pl.read_parquet(parquet_path)

    res_df, metrics = estimate_cate_dml(df)

    # Segment-level comparison: True vs DML Predicted CATE
    seg_summary = (
        res_df.group_by(["device", "city_tier"])
        .agg(
            pl.len().alias("users"),
            pl.col("true_cate").mean().round(3).alias("true_lift"),
            pl.col("predicted_cate").mean().round(3).alias("dml_predicted_lift"),
            pl.col("predicted_cate").std().round(3).alias("pred_std"),
        )
        .sort("dml_predicted_lift", descending=True)
    )

    # Policy Simulation: Blanket Rollout vs Targeted Rollout (CATE > 1.5)
    total_users = len(res_df)
    blanket_total_gain = float(res_df["true_cate"].sum())
    targeted_df = res_df.filter(pl.col("predicted_cate") > 1.5)
    targeted_users = len(targeted_df)
    targeted_total_gain = float(targeted_df["true_cate"].sum())

    elapsed = time.perf_counter() - start_time
    print(f"[SUCCESS] Completed Cross-Fitted Double ML (R-Learner) on {total_users:,} rows in {elapsed:.2f}s\n")
    print("=== 1. CATE MODEL RECOVERY METRICS ===")
    print(f"  CATE RMSE vs Ground Truth: {metrics['cate_rmse']}")
    print(f"  CATE R^2 Score           : {metrics['cate_r2']}")
    print(f"  Overall Mean ATE         : {metrics['mean_predicted_ate']}")

    print("\n=== 2. SEGMENT-WISE CAUSAL UPLIFT RECOVERY ===")
    print(seg_summary)

    print("\n=== 3. TARGETED ROLLOUT POLICY SIMULATION (CATE > 1.5) ===")
    print(f"  Blanket Rollout (100% users)  -> Total Lift: {blanket_total_gain:,.0f} units across {total_users:,} users")
    print(
        f"  Targeted Rollout ({targeted_users/total_users*100:.1f}% users) -> Total Lift: {targeted_total_gain:,.0f} units "
        f"(Captures {targeted_total_gain/blanket_total_gain*100:.1f}% of total value at {targeted_users/total_users*100:.1f}% rollout cost!)"
    )

    # Visualization
    Path("reports").mkdir(exist_ok=True)
    pdf = seg_summary.to_pandas()
    pdf["segment"] = pdf["device"] + " | " + pdf["city_tier"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot A: True vs Predicted CATE by Segment
    x = np.arange(len(pdf))
    width = 0.35
    axes[0].bar(x - width / 2, pdf["true_lift"], width, label="True Causal Lift", color="#2b5c8f")
    axes[0].bar(x + width / 2, pdf["dml_predicted_lift"], width, label="DML Estimated CATE", color="#e66101")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pdf["segment"], rotation=35, ha="right")
    axes[0].set_ylabel("Treatment Effect (Lift)")
    axes[0].set_title("Double ML (R-Learner) Segment Uplift Recovery")
    axes[0].legend()

    # Plot B: Cumulative Uplift Curve (Qini-style targeting curve)
    sorted_df = res_df.sort("predicted_cate", descending=True)
    cum_pct = np.linspace(0, 100, 100)
    idx_steps = np.linspace(0, total_users - 1, 100, dtype=int)
    cum_gain_dml = np.cumsum(sorted_df["true_cate"].to_numpy())[idx_steps] / blanket_total_gain * 100
    cum_gain_random = cum_pct

    axes[1].plot(cum_pct, cum_gain_dml, color="#1b7837", linewidth=2.5, label="DML Uplift Targeting Policy")
    axes[1].plot(cum_pct, cum_gain_random, color="gray", linestyle="--", label="Random / Blanket Rollout")
    axes[1].set_xlabel("% of Population Targeted (Sorted by Predicted CATE)")
    axes[1].set_ylabel("% of Cumulative Incremental Gain Captured")
    axes[1].set_title("Cumulative Causal Uplift Curve (Policy Optimization)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/hte_uplift_analysis.png", dpi=150)
    print("\n[SUCCESS] Saved HTE Uplift figure -> reports/hte_uplift_analysis.png")


if __name__ == "__main__":
    run_hte_pipeline()
