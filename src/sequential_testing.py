import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from src.variance_reduction import apply_univariate_cuped


def compute_msprt_pvalue(
    delta_hat: float, var_delta: float, tau_sq: float = 0.5
) -> tuple[float, float]:
    """
    Computes Mixture Sequential Probability Ratio Test (mSPRT) statistic
    and Always-Valid p-value under normal mixing prior N(0, tau^2).
    """
    ratio = var_delta / (var_delta + tau_sq)
    exponent = (tau_sq * (delta_hat ** 2)) / (2.0 * var_delta * (var_delta + tau_sq))
    # Cap exponent to prevent float64 overflow on very large sample sizes
    exponent = min(exponent, 500.0)
    lambda_n = np.sqrt(ratio) * np.exp(exponent)
    always_valid_p = float(min(1.0, 1.0 / max(lambda_n, 1e-300)))
    return float(lambda_n), always_valid_p


def run_sequential_simulation(
    parquet_path: str = "data/ab_experiment_1m.parquet",
    n_checkpoints: int = 20,
    tau_sq: float = 0.5,
    alpha: float = 0.05,
) -> pl.DataFrame:
    start_time = time.perf_counter()
    df = pl.read_parquet(parquet_path)

    # Shuffle order to simulate realistic sequential arrival over 20 days
    df = df.sample(fraction=1.0, shuffle=True, seed=42)

    variant = df["variant"].to_numpy()
    y_post = df["post_metric"].to_numpy()
    x_pre = df["pre_metric"].to_numpy()

    # Apply CUPED adjustment
    y_cuped, _ = apply_univariate_cuped(y_post, x_pre)

    total_n = len(df)
    step_size = total_n // n_checkpoints
    records = []

    for step in range(1, n_checkpoints + 1):
        n_curr = step * step_size
        v_sub = variant[:n_curr]
        y_raw_sub = y_post[:n_curr]
        y_cup_sub = y_cuped[:n_curr]

        c_mask = v_sub == 0
        t_mask = v_sub == 1

        # Raw mSPRT
        raw_ate = float(np.mean(y_raw_sub[t_mask]) - np.mean(y_raw_sub[c_mask]))
        raw_var = float(
            np.var(y_raw_sub[t_mask], ddof=1) / t_mask.sum()
            + np.var(y_raw_sub[c_mask], ddof=1) / c_mask.sum()
        )
        _, raw_av_p = compute_msprt_pvalue(raw_ate, raw_var, tau_sq=tau_sq)

        # CUPED mSPRT
        cup_ate = float(np.mean(y_cup_sub[t_mask]) - np.mean(y_cup_sub[c_mask]))
        cup_var = float(
            np.var(y_cup_sub[t_mask], ddof=1) / t_mask.sum()
            + np.var(y_cup_sub[c_mask], ddof=1) / c_mask.sum()
        )
        _, cup_av_p = compute_msprt_pvalue(cup_ate, cup_var, tau_sq=tau_sq)

        # Always-valid 95% confidence sequence radius (Johari et al. / Howard et al.)
        cs_radius = np.sqrt(
            (cup_var * (cup_var + tau_sq) / tau_sq)
            * (2.0 * np.log(1.0 / alpha) + np.log((cup_var + tau_sq) / cup_var))
        )

        records.append(
            {
                "checkpoint_day": step,
                "cumulative_users": n_curr,
                "cuped_ate": round(cup_ate, 4),
                "cs_95_low": round(cup_ate - cs_radius, 4),
                "cs_95_high": round(cup_ate + cs_radius, 4),
                "raw_always_valid_p": raw_av_p,
                "cuped_always_valid_p": cup_av_p,
                "safe_to_ship": bool(cup_av_p < alpha),
            }
        )

    seq_df = pl.DataFrame(records)
    elapsed = time.perf_counter() - start_time
    print(f"[SUCCESS] Completed 20-Day Sequential mSPRT Simulation in {elapsed:.2f}s\n")
    print(seq_df.head(10))

    # Find earliest safe stopping day
    ship_days = seq_df.filter(pl.col("safe_to_ship"))
    if len(ship_days) > 0:
        first_day = ship_days.row(0, named=True)
        print(
            f"\n[EARLY STOPPING SIGNAL] CUPED + mSPRT crossed alpha={alpha} on Day {first_day['checkpoint_day']} "
            f"({first_day['cumulative_users']:,} users) with ATE={first_day['cuped_ate']} "
            f"(Always-Valid p={first_day['cuped_always_valid_p']:.2e})"
        )

    # Visualization
    Path("reports").mkdir(exist_ok=True)
    pdf = seq_df.to_pandas()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot A: Always-Valid Confidence Sequence over time
    axes[0].plot(pdf["checkpoint_day"], pdf["cuped_ate"], marker="o", color="#1b7837", label="CUPED ATE")
    axes[0].fill_between(
        pdf["checkpoint_day"], pdf["cs_95_low"], pdf["cs_95_high"],
        color="#1b7837", alpha=0.2, label="Always-Valid 95% Confidence Sequence"
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Always-Valid 95% Confidence Sequence (No Peeking Bias)")
    axes[0].set_xlabel("Experiment Day (50k users/day)")
    axes[0].set_ylabel("Estimated Treatment Effect")
    axes[0].legend()

    # Plot B: Raw mSPRT vs CUPED mSPRT p-value trajectory (Log Scale)
    axes[1].plot(pdf["checkpoint_day"], np.maximum(pdf["raw_always_valid_p"], 1e-16), marker="s", label="Raw mSPRT p-value")
    axes[1].plot(pdf["checkpoint_day"], np.maximum(pdf["cuped_always_valid_p"], 1e-16), marker="o", label="CUPED mSPRT p-value")
    axes[1].axhline(alpha, color="red", linestyle="--", label=f"Significance Threshold (alpha={alpha})")
    axes[1].set_yscale("log")
    axes[1].set_title("Always-Valid p-value Trajectory (Log Scale)")
    axes[1].set_xlabel("Experiment Day")
    axes[1].set_ylabel("Always-Valid p-value (log10)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/sequential_trajectory.png", dpi=150)
    print("[SUCCESS] Saved Sequential Monitoring plot -> reports/sequential_trajectory.png")
    return seq_df


if __name__ == "__main__":
    run_sequential_simulation()
