import duckdb
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def run_srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> dict:
    """Performs Pearson Chi-Square test for Sample Ratio Mismatch (SRM)."""
    total = n_control + n_treatment
    observed = np.array([n_control, n_treatment])
    expected = np.array([total * (1 - expected_ratio), total * expected_ratio])
    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "total_users": int(total),
        "control_pct": round(n_control / total * 100, 4),
        "treatment_pct": round(n_treatment / total * 100, 4),
        "chi2_stat": round(float(chi2_stat), 4),
        "p_value": round(float(p_value), 4),
        "srm_detected": bool(p_value < 0.001),
    }


def run_eda_pipeline(parquet_path: str = "data/ab_experiment_1m.parquet") -> None:
    con = duckdb.connect()

    # 1. SQL-based fast aggregation & SRM check via DuckDB
    counts = con.execute(
        f"""
        SELECT
            SUM(CASE WHEN variant = 0 THEN 1 ELSE 0 END) AS n_control,
            SUM(CASE WHEN variant = 1 THEN 1 ELSE 0 END) AS n_treatment,
            CORR(pre_metric, post_metric) AS pearson_rho,
            AVG(pre_metric) AS mean_pre,
            AVG(post_metric) AS mean_post
        FROM '{parquet_path}'
        """
    ).fetchone()

    n_ctrl, n_treat, rho, mean_pre, mean_post = counts
    srm_res = run_srm_check(n_ctrl, n_treat)
    theoretical_cuped_var_red = (rho ** 2) * 100

    print("=== 1. SAMPLE RATIO MISMATCH (SRM) DIAGNOSTIC ===")
    for k, v in srm_res.items():
        print(f"  {k}: {v}")

    print("\n=== 2. COVARIATE CORRELATION & THEORETICAL CUPED GAIN ===")
    print(f"  Pearson Corr(pre_metric, post_metric): {rho:.4f}")
    print(f"  Theoretical Variance Reduction (rho^2): {theoretical_cuped_var_red:.2f}%")

    # 2. Segment-wise Raw Lift Breakdown using DuckDB SQL
    segment_df = con.execute(
        f"""
        SELECT
            device,
            city_tier,
            COUNT(*) AS users,
            ROUND(AVG(CASE WHEN variant = 0 THEN post_metric END), 3) AS ctrl_post,
            ROUND(AVG(CASE WHEN variant = 1 THEN post_metric END), 3) AS treat_post,
            ROUND(
                AVG(CASE WHEN variant = 1 THEN post_metric END) -
                AVG(CASE WHEN variant = 0 THEN post_metric END), 3
            ) AS raw_lift
        FROM '{parquet_path}'
        GROUP BY device, city_tier
        ORDER BY raw_lift DESC
        """
    ).pl()

    print("\n=== 3. SEGMENT-LEVEL OBSERVED LIFT (DUCKDB SQL) ===")
    print(segment_df)

    # 3. Shape & Outlier Diagnostics on sample / full Polars frame
    df = pl.read_parquet(parquet_path)
    pre_vals = df["pre_metric"].to_numpy()
    post_vals = df["post_metric"].to_numpy()

    print("\n=== 4. DISTRIBUTION SHAPE & ZERO-INFLATION CHECK ===")
    print(f"  pre_metric  -> Skewness: {stats.skew(pre_vals):.3f}, Kurtosis: {stats.kurtosis(pre_vals):.3f}")
    print(f"  post_metric -> Skewness: {stats.skew(post_vals):.3f}, Kurtosis: {stats.kurtosis(post_vals):.3f}")
    print(f"  post_metric == 0 count (clipped floor): {(post_vals == 0).sum():,} ({(post_vals == 0).mean()*100:.3f}%)")

    # 4. Publication-Quality Visuals (using 50k stratified sample for crisp plotting)
    Path("reports").mkdir(exist_ok=True)
    sample_pd = df.sample(n=50_000, seed=42).to_pandas()

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot A: Post-Metric Distribution by Variant
    sns.kdeplot(
        data=sample_pd, x="post_metric", hue="variant",
        fill=True, common_norm=False, palette=["#4C72B0", "#DD8452"], ax=axes[0]
    )
    axes[0].set_title("Post-Experiment Metric Distribution by Variant")
    axes[0].set_xlabel("Post-Experiment Metric")

    # Plot B: Pre vs Post Correlation (CUPED Foundation)
    sns.regplot(
        data=sample_pd.sample(5000, random_state=42),
        x="pre_metric", y="post_metric",
        scatter_kws={"alpha": 0.15, "s": 10, "color": "#2b5c8f"},
        line_kws={"color": "#d95f02", "linewidth": 2},
        ax=axes[1]
    )
    axes[1].set_title(f"Pre vs. Post Metric Correlation (rho = {rho:.3f})")
    axes[1].set_xlabel("Pre-Experiment Metric (Covariate)")
    axes[1].set_ylabel("Post-Experiment Metric (Outcome)")

    # Plot C: Segment-Level Raw Lift Heatmap
    pivot_lift = segment_df.to_pandas().pivot(index="device", columns="city_tier", values="raw_lift")
    sns.heatmap(pivot_lift, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[2])
    axes[2].set_title("Observed Raw Lift Across Segments (Treatment - Control)")

    plt.tight_layout()
    plt.savefig("reports/eda_diagnostics.png", dpi=150)
    print("\n[SUCCESS] Saved EDA diagnostic figure -> reports/eda_diagnostics.png")


if __name__ == "__main__":
    run_eda_pipeline()
