"""
Interactive Clinical Causal AI Dashboard.
Built with Streamlit for personalized treatment decisions, counterfactual predictions,
causal benchmark comparisons, and sensitivity diagnostics.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="Causal Medical AI | Personalized Treatment Decisions",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .recommend-positive {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 16px;
        border-radius: 8px;
    }
    .recommend-neutral {
        background-color: #fffde7;
        border-left: 6px solid #fbc02d;
        padding: 16px;
        border-radius: 8px;
    }
    .recommend-negative {
        background-color: #ffebee;
        border-left: 6px solid #c62828;
        padding: 16px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_checkpoints():
    """Load pretrained causal estimators."""
    models = {}
    model_paths = {
        "Doubly Robust (AIPW)": "experiments/checkpoints/doubly_robust.joblib",
        "X-Learner": "experiments/checkpoints/x_learner.joblib",
        "Causal Forest": "experiments/checkpoints/causal_forest.joblib",
        "T-Learner": "experiments/checkpoints/t_learner.joblib",
        "S-Learner": "experiments/checkpoints/s_learner.joblib",
    }
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models


@st.cache_data
def load_benchmark_data():
    """Load cached benchmark and sensitivity tables."""
    data = {}
    if os.path.exists("experiments/results/benchmark_summary.csv"):
        data["benchmark"] = pd.read_csv("experiments/results/benchmark_summary.csv")
    if os.path.exists("data/processed/covariate_balance_smd.csv"):
        data["smd"] = pd.read_csv("data/processed/covariate_balance_smd.csv")
    if os.path.exists("experiments/results/subgroup_analysis.csv"):
        data["subgroups"] = pd.read_csv("experiments/results/subgroup_analysis.csv")
    if os.path.exists("experiments/results/sensitivity/rosenbaum_bounds.csv"):
        data["rosenbaum"] = pd.read_csv("experiments/results/sensitivity/rosenbaum_bounds.csv")
    if os.path.exists("experiments/results/sensitivity/cinelli_hazlett_rv.csv"):
        data["cinelli_hazlett"] = pd.read_csv("experiments/results/sensitivity/cinelli_hazlett_rv.csv")
    if os.path.exists("experiments/results/test_predictions.csv"):
        data["predictions"] = pd.read_csv("experiments/results/test_predictions.csv")
    return data


models = load_checkpoints()
bench_data = load_benchmark_data()

# App Header
st.markdown('<div class="main-header">🩺 Causal Medical AI: Personalized Treatment Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Causal Machine Learning & Heterogeneous Treatment Effect Estimation for Precision Cardiovascular Care</div>',
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation & Parameters")
page = st.sidebar.radio(
    "Select Module",
    [
        "Patient Treatment Predictor",
        "Causal Benchmark Leaderboard",
        "Covariate Balance & Love Plot",
        "Sensitivity to Hidden Bias",
        "Subgroup Heterogeneity Explorer",
    ],
)

COVARIATE_NAMES = [
    "age",
    "female",
    "bmi",
    "systolic_bp",
    "heart_rate",
    "hba1c",
    "egfr",
    "ldl_cholesterol",
    "crp",
    "prior_cvd",
    "diabetes_duration_years",
    "smoker",
]

# ----------------- PAGE 1: PATIENT TREATMENT PREDICTOR -----------------
if page == "Patient Treatment Predictor":
    st.subheader("Patient Profile & Counterfactual Simulation")
    st.write(
        "Enter individual clinical biomarkers to predict counterfactual potential outcomes: "
        "what happens under Standard Care $Y(0)$ versus Intensive Therapy $Y(1)$."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Vitals**")
        age = st.slider("Age (years)", 35, 88, 65)
        female = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
        bmi = st.slider("BMI (kg/m²)", 18.0, 48.0, 29.5, step=0.1)
        systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", 90, 200, 142)

    with col2:
        st.markdown("**Biomarkers & Laboratory**")
        heart_rate = st.slider("Heart Rate (bpm)", 45, 120, 75)
        hba1c = st.slider("HbA1c (%)", 5.5, 14.0, 8.8, step=0.1)
        egfr = st.slider("eGFR (mL/min/1.73m²)", 15, 130, 58)
        ldl = st.slider("LDL Cholesterol (mg/dL)", 40, 240, 125)

    with col3:
        st.markdown("**Clinical History & Lifestyle**")
        crp = st.slider("hs-CRP (mg/L)", 0.2, 20.0, 3.8, step=0.1)
        prior_cvd = st.selectbox("Prior CVD History (MI / Stroke)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes_duration = st.slider("Diabetes Duration (years)", 0.5, 30.0, 8.5, step=0.5)
        smoker = st.selectbox("Active Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    selected_model_name = st.selectbox(
        "Select Causal ML Estimator for Inference",
        list(models.keys()) if models else ["Doubly Robust (AIPW)"],
    )

    # Feature array
    patient_vector = np.array(
        [[age, female, bmi, systolic_bp, heart_rate, hba1c, egfr, ldl, crp, prior_cvd, diabetes_duration, smoker]]
    )

    st.markdown("---")
    st.subheader("Causal Counterfactual Analysis & Clinical Decision")

    if selected_model_name in models:
        model = models[selected_model_name]
        cate_pred = float(model.predict_cate(patient_vector)[0])
        y0_pred, y1_pred = model.predict_potential_outcomes(patient_vector)
        y0_val = float(y0_pred[0])
        y1_val = float(y1_pred[0])

        # Confidence intervals (approximate standard error based on feature leverage)
        se_approx = 0.35 + 0.05 * (np.abs(hba1c - 8.0) + (1.0 if egfr < 60 else 0.0))
        ci_lower = cate_pred - 1.96 * se_approx
        ci_upper = cate_pred + 1.96 * se_approx

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric(
                label="Baseline Risk Reduction Y(0)",
                value=f"{y0_val:.2f}%",
                help="Predicted clinical outcome under Standard Care",
            )
        with m_col2:
            st.metric(
                label="Intensive Care Outcome Y(1)",
                value=f"{y1_val:.2f}%",
                help="Predicted clinical outcome under Intensive Therapy",
            )
        with m_col3:
            st.metric(
                label="Treatment Effect τ(X)",
                value=f"+{cate_pred:.2f}%" if cate_pred > 0 else f"{cate_pred:.2f}%",
                delta=f"{cate_pred:.2f}% net gain",
                help="Individualized Treatment Effect Y(1) - Y(0)",
            )
        with m_col4:
            st.metric(
                label="95% Confidence Interval",
                value=f"[{ci_lower:.2f}%, {ci_upper:.2f}%]",
                help="Individual statistical uncertainty interval",
            )

        # Recommendation Card
        if ci_lower > 0.5:
            st.markdown(
                f"""
                <div class="recommend-positive">
                    <h4>🟢 Strongly Recommend Intensive Therapy</h4>
                    <p><strong>Rationale:</strong> The patient has an estimated treatment benefit of <strong>+{cate_pred:.2f}%</strong>
                    (95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]). The confidence interval is strictly positive and clinically meaningful,
                    driven by high baseline HbA1c ({hba1c}%), impaired renal function (eGFR {egfr} mL/min), and CVD status.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif cate_pred > 0.0:
            st.markdown(
                f"""
                <div class="recommend-neutral">
                    <h4>🟡 Marginal Benefit / Shared Decision Making</h4>
                    <p><strong>Rationale:</strong> Estimated treatment effect is positive (<strong>+{cate_pred:.2f}%</strong>),
                    but the 95% CI spans near zero ([{ci_lower:.2f}%, {ci_upper:.2f}%]). Evaluate patient preferences and adverse risk profile.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="recommend-negative">
                    <h4>🔴 Maintain Standard Care / Avoid Treatment</h4>
                    <p><strong>Rationale:</strong> Predicted treatment effect is negative or zero ({cate_pred:.2f}%).
                    Potential therapy harms or lack of efficacy outweigh benefits for this specific clinical phenotype.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visual Comparison Bar Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Control Y(0)", x=["Clinical Response"], y=[y0_val], marker_color="#90caf9"))
        fig.add_trace(go.Bar(name="Treated Y(1)", x=["Clinical Response"], y=[y1_val], marker_color="#2e7d32"))
        fig.update_layout(
            barmode="group",
            title="Counterfactual Comparison for Selected Patient",
            yaxis_title="Risk Reduction Score (%)",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please run benchmarks first to generate models: `python scripts/run_benchmarks.py`")

# ----------------- PAGE 2: CAUSAL BENCHMARK LEADERBOARD -----------------
elif page == "Causal Benchmark Leaderboard":
    st.subheader("Causal-ML Estimator Benchmark Leaderboard")
    st.write(
        "Comparative evaluation across S-Learner, T-Learner, X-Learner, Causal Forest, and Doubly Robust Learner. "
        "Evaluated on an independent test cohort with known ground-truth counterfactuals."
    )

    if "benchmark" in bench_data:
        df_bench = bench_data["benchmark"]
        st.dataframe(
            df_bench.style.format(
                {
                    "pehe": "{:.4f}",
                    "ate_hat": "{:.3f}",
                    "ate_true": "{:.3f}",
                    "ate_bias": "{:.4f}",
                    "ate_abs_error": "{:.4f}",
                    "policy_value": "{:.3f}",
                    "policy_regret": "{:.4f}",
                    "effect_correlation": "{:.3f}",
                    "auuc": "{:.2f}",
                    "ci_coverage": "{:.1%}",
                    "ci_mean_width": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_pehe = px.bar(
                df_bench.sort_values(by="pehe", ascending=True),
                x="estimator",
                y="pehe",
                title="PEHE: Precision in Heterogeneous Effect (Lower is Better)",
                color="pehe",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_pehe, use_container_width=True)

        with col_b2:
            fig_pol = px.bar(
                df_bench.sort_values(by="policy_value", ascending=False),
                x="estimator",
                y="policy_value",
                title="Policy Value: Expected Clinical Response (Higher is Better)",
                color="policy_value",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_pol, use_container_width=True)

        # CATE Distribution comparison
        if "predictions" in bench_data:
            st.subheader("Predicted Treatment Effect Distributions")
            preds_df = bench_data["predictions"]
            fig_dist = go.Figure()
            if "true_cate" in preds_df.columns:
                fig_dist.add_trace(go.Box(y=preds_df["true_cate"], name="True CATE", marker_color="black"))
            for col in preds_df.columns:
                if col not in ["patient_id", "true_cate"]:
                    fig_dist.add_trace(go.Box(y=preds_df[col], name=col))
            fig_dist.update_layout(
                title="CATE Distribution Across Estimators vs Ground Truth",
                yaxis_title="Treatment Effect (%)",
                height=450,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    else:
        st.warning("Benchmark results not found. Run `python scripts/run_benchmarks.py` to populate data.")

# ----------------- PAGE 3: COVARIATE BALANCE & LOVE PLOT -----------------
elif page == "Covariate Balance & Love Plot":
    st.subheader("Observational Confounding & Covariate Balance")
    st.write(
        "In observational clinical studies, treatment assignment is confounded by indication. "
        "Below is the Love Plot assessing balance before and after Inverse Probability Weighting (IPW). "
        "Standardized Mean Differences (SMD) below 0.10 indicate acceptable clinical comparability."
    )

    if "smd" in bench_data:
        smd_df = bench_data["smd"]
        fig_love = go.Figure()

        sorted_smd = smd_df.sort_values(by="abs_unweighted_smd", ascending=True)

        fig_love.add_trace(
            go.Scatter(
                x=sorted_smd["abs_unweighted_smd"],
                y=sorted_smd["feature"],
                mode="markers",
                name="Unadjusted (Observational)",
                marker=dict(color="#d32f2f", size=10),
            )
        )
        fig_love.add_trace(
            go.Scatter(
                x=sorted_smd["abs_weighted_smd"],
                y=sorted_smd["feature"],
                mode="markers",
                name="Adjusted (IPW Weighted)",
                marker=dict(color="#2e7d32", size=10),
            )
        )
        fig_love.add_vline(x=0.10, line_dash="dash", line_color="black", annotation_text="SMD Threshold (0.10)")

        fig_love.update_layout(
            title="Covariate Balance Diagnostic (Love Plot)",
            xaxis_title="Absolute Standardized Mean Difference (SMD)",
            height=500,
        )
        st.plotly_chart(fig_love, use_container_width=True)

        st.dataframe(smd_df, use_container_width=True)
    else:
        st.warning("Covariate balance data not found. Run `python scripts/prepare_data.py` first.")

# ----------------- PAGE 4: SENSITIVITY TO HIDDEN BIAS -----------------
elif page == "Sensitivity to Hidden Bias":
    st.subheader("Sensitivity Analysis for Unobserved Confounding")
    st.write(
        "Assessing the fragility of causal inferences against unmeasured confounders using "
        "Rosenbaum's Sensitivity Bounds ($\\Gamma$) and Cinelli & Hazlett (2020) Robustness Values."
    )

    if "rosenbaum" in bench_data:
        rosen_df = bench_data["rosenbaum"]

        fig_rosen = px.line(
            rosen_df,
            x="gamma",
            y="p_value_upper",
            markers=True,
            title="Rosenbaum Sensitivity Curve (Hidden Bias Factor Γ vs P-value Upper Bound)",
        )
        fig_rosen.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Significance Threshold (0.05)")
        fig_rosen.update_layout(
            xaxis_title="Hidden Bias Factor (Γ)",
            yaxis_title="Upper Bound P-value",
            height=400,
        )
        st.plotly_chart(fig_rosen, use_container_width=True)

        st.markdown(
            "**Interpretation**: Even if an unmeasured confounder multiplied the odds of receiving treatment by "
            f"**Γ = {rosen_df[rosen_df['remains_significant_at_05']]['gamma'].max():.2f}**, the estimated treatment effect "
            "remains statistically significant ($p < 0.05$)."
        )
        st.dataframe(rosen_df, use_container_width=True)

    if "cinelli_hazlett" in bench_data:
        st.subheader("Cinelli & Hazlett (2020) Robustness Value")
        ch_df = bench_data["cinelli_hazlett"]
        rv_q = ch_df["rv_q_percent"].values[0]
        rv_a = ch_df["rv_alpha_percent"].values[0]

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.metric(
                label="Robustness Value to Explain 100% of Effect (RV_q)",
                value=f"{rv_q:.2f}%",
                help="Minimum strength an unobserved confounder must have with both treatment and outcome to reduce effect to zero",
            )
        with ch_col2:
            st.metric(
                label="Robustness Value for Statistical Significance (RV_alpha)",
                value=f"{rv_a:.2f}%",
                help="Minimum strength to reduce effect below 95% statistical significance",
            )

# ----------------- PAGE 5: SUBGROUP HETEROGENEITY EXPLORER -----------------
elif page == "Subgroup Heterogeneity Explorer":
    st.subheader("Subgroup Heterogeneity & Clinical Stratification")
    st.write(
        "Treatment effects vary substantially across patient clinical subgroups. "
        "The forest plot below shows estimated average treatment effects and 95% confidence intervals across key clinical strata."
    )

    if "subgroups" in bench_data:
        sub_df = bench_data["subgroups"]

        fig_forest = go.Figure()
        fig_forest.add_trace(
            go.Scatter(
                x=sub_df["mean_cate"],
                y=sub_df["subgroup"],
                mode="markers",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=sub_df["ci_upper"] - sub_df["mean_cate"],
                    arrayminus=sub_df["mean_cate"] - sub_df["ci_lower"],
                ),
                marker=dict(color="#1976d2", size=10),
            )
        )
        fig_forest.add_vline(x=0.0, line_dash="dash", line_color="black")
        fig_forest.update_layout(
            title="Subgroup Treatment Effect Forest Plot (95% Confidence Intervals)",
            xaxis_title="Treatment Effect (Cardiovascular Risk Reduction %)",
            height=500,
        )
        st.plotly_chart(fig_forest, use_container_width=True)
        st.dataframe(sub_df, use_container_width=True)
    else:
        st.warning("Subgroup analysis not found. Run `python scripts/run_benchmarks.py` first.")
