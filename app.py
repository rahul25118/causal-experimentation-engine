import duckdb
import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
from src.data_generator import generate_ab_dataset
from src.eda_diagnostics import run_srm_check
from src.variance_reduction import apply_univariate_cuped, compute_welch_ttest
from src.sequential_testing import compute_msprt_pvalue

st.set_page_config(page_title="Causal Experimentation Engine", layout="wide")
st.title("Large-Scale Causal Inference & Experimentation Engine")
st.caption("DuckDB + Polars | CUPED Variance Reduction | Always-Valid mSPRT | Double ML Uplift")

DATA_PATH = "data/ab_experiment_1m.parquet"
if not Path(DATA_PATH).exists():
    with st.spinner("Generating 1,000,000-row A/B dataset..."):
        generate_ab_dataset(n_users=1_000_000, output_path=DATA_PATH)

con = duckdb.connect()

# Sidebar Filters
st.sidebar.header("Experiment Controls")
selected_device = st.sidebar.multiselect("Device Filter", ["Android", "iOS", "Web"], default=["Android", "iOS", "Web"])
selected_tier = st.sidebar.multiselect("City Tier Filter", ["Tier-1", "Tier-2", "Tier-3"], default=["Tier-1", "Tier-2", "Tier-3"])
tau_sq = st.sidebar.slider("mSPRT Mixing Variance (tau^2)", 0.1, 2.0, 0.5, 0.1)

if not selected_device or not selected_tier:
    st.warning("Please select at least one Device and City Tier.")
    st.stop()

dev_sql = ",".join([f"'{d}'" for d in selected_device])
tier_sql = ",".join([f"'{t}'" for t in selected_tier])

filtered_df = con.execute(
    f"""
    SELECT variant, pre_metric, post_metric, device, city_tier
    FROM '{DATA_PATH}'
    WHERE device IN ({dev_sql}) AND city_tier IN ({tier_sql})
    """
).pl()

v = filtered_df["variant"].to_numpy()
y_post = filtered_df["post_metric"].to_numpy()
x_pre = filtered_df["pre_metric"].to_numpy()

c_mask, t_mask = (v == 0), (v == 1)
srm = run_srm_check(int(c_mask.sum()), int(t_mask.sum()))

naive_res = compute_welch_ttest(y_post[c_mask], y_post[t_mask])
y_cuped, theta = apply_univariate_cuped(y_post, x_pre)
cuped_res = compute_welch_ttest(y_cuped[c_mask], y_cuped[t_mask])
var_red = (1 - np.var(y_cuped) / np.var(y_post)) * 100

cup_var_delta = float(np.var(y_cuped[t_mask], ddof=1) / t_mask.sum() + np.var(y_cuped[c_mask], ddof=1) / c_mask.sum())
_, av_p = compute_msprt_pvalue(cuped_res["ate"], cup_var_delta, tau_sq=tau_sq)

# KPI Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Filtered Users", f"{len(filtered_df):,}", f"SRM p={srm['p_value']}")
k2.metric("CUPED ATE (Lift)", f"{cuped_res['ate']:.4f}", f"Naive: {naive_res['ate']:.4f}")
k3.metric("Variance Reduction", f"{var_red:.2f}%", f"theta = {theta:.3f}")
k4.metric("Always-Valid p-value", f"{av_p:.2e}", "Significant" if av_p < 0.05 else "Inconclusive")

st.subheader("Segment-Level Uplift Summary (DuckDB Aggregation)")
seg_table = con.execute(
    f"""
    SELECT
        device,
        city_tier,
        COUNT(*) AS users,
        ROUND(AVG(CASE WHEN variant = 0 THEN post_metric END), 3) AS control_mean,
        ROUND(AVG(CASE WHEN variant = 1 THEN post_metric END), 3) AS treatment_mean,
        ROUND(AVG(CASE WHEN variant = 1 THEN post_metric END) - AVG(CASE WHEN variant = 0 THEN post_metric END), 3) AS observed_lift
    FROM '{DATA_PATH}'
    WHERE device IN ({dev_sql}) AND city_tier IN ({tier_sql})
    GROUP BY device, city_tier
    ORDER BY observed_lift DESC
    """
).df()
st.dataframe(seg_table, use_container_width=True)
