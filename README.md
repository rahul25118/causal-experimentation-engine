# Large-Scale Causal Inference & Experimentation Engine

A production-grade statistical experimentation platform designed to eliminate peeking bias, reduce estimator variance, and uncover heterogeneous treatment effects across **1,000,000+ user-level events** using **Polars, DuckDB, SciPy, and Cross-Fitted Double Machine Learning**.

---

## System Architecture

```mermaid
flowchart LR
    A[(1M+ User Parquet Logs)] --> B[DuckDB & Polars Engine]
    B --> C{SRM Chi-Square Gate}
    C -- "p < 0.001 (Fail)" --> D[Halt Pipeline & Alert]
    C -- "p >= 0.001 (Pass)" --> E[CUPED / CUPAC Variance Reduction]
    E --> F[Always-Valid mSPRT Sequential Monitor]
    E --> G[Cross-Fitted Double ML R-Learner]
    F --> H[Early Stopping & Safe Launch Decision]
    G --> I[Segment Uplift & Targeted Policy ROI]
    H & I --> J[Streamlit Interactive Decision App]
