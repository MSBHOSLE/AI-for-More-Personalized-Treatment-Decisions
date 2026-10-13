# Causal Medical AI: Personalized Treatment Decisions

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/)

An end-to-end causal machine learning and statistical framework designed for **individualized treatment effect estimation (ITE / CATE)**, observational confounding adjustment, and precision clinical policy optimization in high-stakes healthcare scenarios.

---

## 🎯 Clinical Context & Motivation
Standard predictive models in healthcare estimate associative conditional risk $\mathbb{E}[Y \mid X]$, which fails to answer counterfactual questions: **"What would happen to patient $i$ under Intensive Therapy $Y(1)$ versus Standard Care $Y(0)$?"**

Because factual observational records only show $Y = T \cdot Y(1) + (1-T) \cdot Y(0)$, naïve regression models suffer from **confounding by indication** (sicker patients receive more aggressive therapy).

This repository combines machine learning and causal inference to:
1. **Estimate Heterogeneous Treatment Effects (CATE)**: $\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$.
2. **Correct for Selection Bias**: Propensity score adjustment and doubly robust Neyman orthogonal scoring.
3. **Quantify Uncertainty**: Individualized 95% bootstrap confidence intervals for clinical decision gating.
4. **Stress-Test Robustness**: Rosenbaum sensitivity bounds, Cinelli & Hazlett robustness values ($RV$), and Oster's $\delta$ to evaluate vulnerability to unmeasured confounders.
5. **Optimize Personalized Policies**: Policy value curves and Qini / Uplift curves to prioritize patients who benefit most.

---

## 📁 Repository Structure

```
causal-medical-ai/
├── configs/
│   ├── dataset.yaml               # Cohort generation parameters, features, and clinical units
│   ├── training.yaml              # Hyperparameters for meta-learners, trees, and propensity models
│   └── experiments.yaml           # Benchmark, sensitivity, sample size, and imbalance configs
├── data/
│   ├── raw/                       # Full observational cohort with true potential outcomes
│   ├── processed/                 # Train/test partitions and covariate balance SMD tables
│   └── README.md                  # Clinical data dictionary and DGP documentation
├── src/
│   ├── data/
│   │   ├── dataset.py             # CausalDataset abstraction, common support trimming, IPW
│   │   └── simulator.py           # Cardiometabolic clinical cohort generator with known CATE
│   ├── estimators/
│   │   ├── s_learner.py           # S-Learner with feature interactions
│   │   ├── t_learner.py           # T-Learner (separate factual/counterfactual response surfaces)
│   │   ├── x_learner.py           # X-Learner (Künzel et al. 2019, 2-stage propensity weighted)
│   │   ├── causal_forest.py       # Honest Causal Forest (Athey & Wager GRF adaptation)
│   │   └── doubly_robust.py       # Doubly Robust AIPW with K-fold cross-fitting
│   ├── evaluation/
│   │   ├── metrics.py             # PEHE, ATE Bias, Policy Value, Policy Regret, Qini & AUUC
│   │   ├── diagnostics.py         # Standardized Mean Differences (Love Plot), overlap checks
│   │   └── confidence_intervals.py# Nonparametric bootstrap CIs and coverage diagnostics
│   ├── experiments/
│   │   ├── sample_size.py         # Sample efficiency analysis across cohort sizes
│   │   ├── confounding.py         # Degradation benchmarks under unobserved confounding
│   │   └── treatment_imbalance.py # Stress tests across treatment prevalence (5% to 50%)
│   ├── sensitivity.py             # Rosenbaum's Gamma, Cinelli-Hazlett RV, Oster's Delta
│   ├── visualization.py           # Publication figures (Love Plot, Qini curves, Forest plot)
│   └── utils.py                   # Reproducibility, YAML loading, markdown reporting
├── scripts/
│   ├── prepare_data.py            # Generates cohort, computes propensity scores & balance
│   ├── run_benchmarks.py          # Trains 5 causal estimators, evaluates PEHE, saves checkpoints
│   ├── run_sensitivity.py         # Runs Rosenbaum, Cinelli-Hazlett, and Oster sensitivity analyses
│   └── generate_report.py         # Compiles empirical metrics into reports/research_report.md
├── tests/
│   ├── test_dataset.py            # Simulator, covariate scaling, and balance tests
│   ├── test_estimators.py         # Unit tests for all 5 causal ML estimators
│   ├── test_evaluation.py         # Metrics, PEHE, AUUC, and CI tests
│   └── test_sensitivity.py        # Rosenbaum and Cinelli-Hazlett robustness tests
├── experiments/
│   ├── checkpoints/               # Serialized model joblib binaries
│   └── results/                   # Benchmark CSVs, predictions, and sensitivity data
├── reports/
│   ├── figures/                   # High-res publication plots
│   └── research_report.md         # Comprehensive empirical research report
├── app.py                         # Interactive Streamlit clinical decision application
├── requirements.txt               # Pinned Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## 🔬 Benchmark Results

Empirical results on the held-out test cohort ($N=1,250$):

| Estimator | PEHE $\downarrow$ | ATE Hat | ATE Bias $\downarrow$ | Policy Value $\uparrow$ | AUUC $\uparrow$ | 95% CI Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **X-Learner** | **0.3599** | 3.4016 | 0.0397 | **8.3573** | **1519.35** | 88.24% |
| **S-Learner** | 0.3645 | 3.4070 | 0.0452 | 8.3573 | 1526.76 | 85.92% |
| **T-Learner** | 0.4460 | 3.4589 | 0.0971 | 8.3573 | 1502.49 | 94.08% |
| **Doubly Robust (AIPW)** | 0.4903 | **3.3874** | **0.0255** | 8.3565 | 1515.12 | **94.32%** |
| **Causal Forest** | 0.5983 | 3.7233 | 0.3614 | 8.3573 | 1513.33 | 36.16% |

- **Heterogeneity Estimation**: X-Learner achieved the lowest PEHE (0.3599), effectively leveraging propensity weighting to overcome group size differences.
- **Population ATE Estimation**: Doubly Robust (AIPW) achieved the lowest ATE bias (0.0255) and highest 95% CI coverage (94.32%), confirming double robustness and Neyman orthogonality.
- **Covariate Balance**: IPW reduced Standardized Mean Differences (SMD) across all 12 covariates from up to 0.68 down to $< 0.04$, well below the 0.10 clinical threshold.

---

## 🛡️ Sensitivity Analysis: Hidden Confounding Robustness

1. **Rosenbaum's Bounds ($\Gamma$)**:
   The treatment effect remains statistically significant ($p < 0.0001$) up to $\Gamma = 3.00$. An unobserved confounder would have to triple the odds of receiving treatment to overturn the conclusion.
2. **Cinelli & Hazlett (2020) Robustness Value ($RV$)**:
   - $RV_q = 67.37\%$: An unobserved confounder must explain over 67% of both treatment selection and outcome residual variance to eliminate the treatment effect.
   - $RV_\alpha = 65.76\%$: Robustness value to reduce the effect below 95% statistical significance.
3. **Oster's Delta ($\delta$)**:
   $\delta = 4.098$ (exceeds the recommended threshold $\delta > 1.0$), demonstrating substantial resilience to selection on unobservables.

---

## 🚀 Quickstart Guide

### 1. Setup Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests -v
```

### 3. Prepare Data & Baseline Diagnostics
```bash
python scripts/prepare_data.py
```

### 4. Execute Causal Benchmarks
```bash
python scripts/run_benchmarks.py
```

### 5. Run Sensitivity Analysis
```bash
python scripts/run_sensitivity.py
```

### 6. Generate Research Report
```bash
python scripts/generate_report.py
```

### 7. Launch Interactive Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interact with:
- **Patient Treatment Predictor**: Enter patient vitals, biomarkers, and clinical history to receive counterfactual predictions and individualized recommendations.
- **Benchmark Leaderboard**: Interactive sorting, metrics, and CATE distributions.
- **Covariate Balance**: Interactive Love plot.
- **Sensitivity Studio**: Rosenbaum curve and Cinelli-Hazlett robustness calculators.
- **Subgroup Forest Explorer**: Treatment effect across clinical subpopulations.
