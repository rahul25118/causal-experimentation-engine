import time
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import Ridge


def compute_welch_ttest(y_ctrl: np.ndarray, y_treat: np.ndarray) -> dict:
    """Computes two-sample Welch's t-test, ATE, Standard Error, and 95% CI."""
    ate = float(np.mean(y_treat) - np.mean(y_ctrl))
    se = float(
        np.sqrt(
            np.var(y_treat, ddof=1) / len(y_treat)
            + np.var(y_ctrl, ddof=1) / len(y_ctrl)
        )
    )
    t_stat, p_val = stats.ttest_ind(y_treat, y_ctrl, equal_var=False)
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se
    return {
        "ate": round(ate, 5),
        "std_error": round(se, 5),
        "ci_95_low": round(ci_low, 5),
        "ci_95_high": round(ci_high, 5),
        "ci_width": round(ci_high - ci_low, 5),
        "t_stat": round(float(t_stat), 4),
        "p_value": float(p_val),
    }


def apply_univariate_cuped(
    y_post: np.ndarray, x_pre: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Applies classical CUPED adjustment using pooled pre-experiment covariate:
    Y_cuped = Y_post - theta * (X_pre - mean(X_pre))
    """
    x_mean = np.mean(x_pre)
    cov_xy = np.cov(y_post, x_pre, ddof=1)[0, 1]
    var_x = np.var(x_pre, ddof=1)
    theta = float(cov_xy / var_x)
    y_cuped = y_post - theta * (x_pre - x_mean)
    return y_cuped, theta


def apply_multivariate_cupac(df: pl.DataFrame) -> np.ndarray:
    """
    Applies Multivariate Regression-Adjusted CUPED (CUPAC) using Ridge regression
    trained purely on pre-experiment covariates (no treatment leakage).
    """
    # One-hot encode categorical pre-experiment features cleanly
    features_df = df.select(["pre_metric", "tenure_days", "device", "city_tier"]).to_dummies(
        columns=["device", "city_tier"], drop_first=True
    )
    X = features_df.to_numpy()
    y = df["post_metric"].to_numpy()

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    y_hat = model.predict(X)

    # Center predictions to preserve unbiased ATE
    y_cupac = y - (y_hat - np.mean(y_hat))
    return y_cupac


def run_variance_reduction_benchmark(
    parquet_path: str = "data/ab_experiment_1m.parquet",
) -> pl.DataFrame:
    start_time = time.perf_counter()
    df = pl.read_parquet(parquet_path)

    variant = df["variant"].to_numpy()
    y_post = df["post_metric"].to_numpy()
    x_pre = df["pre_metric"].to_numpy()

    ctrl_mask = variant == 0
    treat_mask = variant == 1

    # 1. Naive Unadjusted Estimator
    naive_res = compute_welch_ttest(y_post[ctrl_mask], y_post[treat_mask])
    naive_var = float(np.var(y_post, ddof=1))

    # 2. Univariate CUPED Estimator
    y_cuped, theta = apply_univariate_cuped(y_post, x_pre)
    cuped_res = compute_welch_ttest(y_cuped[ctrl_mask], y_cuped[treat_mask])
    cuped_var = float(np.var(y_cuped, ddof=1))
    cuped_var_red = (1.0 - cuped_var / naive_var) * 100

    # 3. Multivariate CUPAC Estimator
    y_cupac = apply_multivariate_cupac(df)
    cupac_res = compute_welch_ttest(y_cupac[ctrl_mask], y_cupac[treat_mask])
    cupac_var = float(np.var(y_cupac, ddof=1))
    cupac_var_red = (1.0 - cupac_var / naive_var) * 100

    summary = pl.DataFrame(
        [
            {
                "method": "1. Naive Welch t-test",
                "ate": naive_res["ate"],
                "std_error": naive_res["std_error"],
                "ci_95_width": naive_res["ci_width"],
                "t_stat": naive_res["t_stat"],
                "variance_reduction_pct": 0.0,
            },
            {
                "method": f"2. Univariate CUPED (theta={theta:.3f})",
                "ate": cuped_res["ate"],
                "std_error": cuped_res["std_error"],
                "ci_95_width": cuped_res["ci_width"],
                "t_stat": cuped_res["t_stat"],
                "variance_reduction_pct": round(cuped_var_red, 2),
            },
            {
                "method": "3. Multivariate CUPAC (Ridge)",
                "ate": cupac_res["ate"],
                "std_error": cupac_res["std_error"],
                "ci_95_width": cupac_res["ci_width"],
                "t_stat": cupac_res["t_stat"],
                "variance_reduction_pct": round(cupac_var_red, 2),
            },
        ]
    )

    elapsed = time.perf_counter() - start_time
    print(f"[SUCCESS] Completed Variance Reduction Benchmark on {len(df):,} rows in {elapsed:.2f}s\n")
    print(summary)
    return summary


if __name__ == "__main__":
    run_variance_reduction_benchmark()
