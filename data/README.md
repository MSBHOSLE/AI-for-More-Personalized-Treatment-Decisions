# Causal Medical Dataset: Personalized Cardiovascular Treatment Cohort

## 1. Clinical Context & Motivation
Personalized medicine requires estimating the **Individualized Treatment Effect (ITE)**:
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$
In high-risk cardiovascular and metabolic patient care, prescribing intensive cardioprotective therapy (e.g., combined SGLT2i + GLP-1 RA / intensive lipid and BP control) yields significant benefits for some patients while causing adverse events, polypharmacy burden, or limited efficacy in others.

Because standard clinical records only observe the factual outcome:
$$Y = T \cdot Y(1) + (1 - T) \cdot Y(0)$$
direct supervision for $\tau(x)$ is impossible (the fundamental problem of causal inference). This repository provides a rigorous semi-synthetic benchmark with known ground truth counterfactuals to evaluate causal ML estimators.

## 2. Variables & Feature Dictionary

| Variable Name | Type | Units / Values | Description | Clinical Role |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | Identifier | String | Unique patient identifier | ID |
| `age` | Continuous | Years (35 - 88) | Patient chronological age | Confounder & Effect Modifier |
| `female` | Binary | 0 = Male, 1 = Female | Biological sex | Demographic covariate |
| `bmi` | Continuous | $\text{kg/m}^2$ (18 - 48) | Body Mass Index | Metabolic risk factor |
| `systolic_bp` | Continuous | mmHg (90 - 200) | Systolic blood pressure | Hemodynamic risk factor |
| `heart_rate` | Continuous | bpm (45 - 120) | Resting heart rate | Hemodynamic covariate |
| `hba1c` | Continuous | % (5.5 - 14.0) | Glycated hemoglobin | Glycemic severity & Effect Modifier |
| `egfr` | Continuous | $\text{mL/min/1.73m}^2$ | Estimated glomerular filtration rate | Renal function & Effect Modifier |
| `ldl_cholesterol` | Continuous | mg/dL (40 - 240) | Low-density lipoprotein cholesterol | Atherogenic lipid marker |
| `crp` | Continuous | mg/L (0.2 - 20.0) | C-reactive protein (systemic inflammation) | Inflammatory biomarker |
| `prior_cvd` | Binary | 0 = No, 1 = Yes | History of myocardial infarction or stroke | Confounder & Effect Modifier |
| `diabetes_duration_years` | Continuous | Years (0 - 30) | Duration since diabetes diagnosis | Disease chronicity |
| `smoker` | Binary | 0 = Non, 1 = Active | Active smoking status | Lifestyle risk factor |
| `treatment` ($T$) | Binary | 0 = Standard Care, 1 = Intensive Therapy | Treatment indicator | Intervention |
| `propensity_score` ($e(X)$) | Continuous | $(0, 1)$ | True conditional probability $P(T=1 \mid X)$ | Selection probability |
| `true_cate` ($\tau(X)$) | Continuous | Risk reduction % | True treatment effect $Y(1) - Y(0)$ | Causal ground truth |
| `y0_true` | Continuous | Risk reduction % | True unobserved potential outcome under control | Counterfactual $Y(0)$ |
| `y1_true` | Continuous | Risk reduction % | True unobserved potential outcome under treatment | Counterfactual $Y(1)$ |
| `outcome` ($Y$) | Continuous | Risk reduction % | Factual observed clinical response | Observed outcome |

## 3. Data Generation Mechanism

1. **Covariate Distribution**: Samples are drawn from realistic correlated truncated normal and Bernoulli distributions mimicking cardiovascular outpatient registries.
2. **Propensity Score (Confounding)**:
   $$\text{logit}(e(X)) = \alpha_0 + 0.04 \cdot (\text{age} - 62) + 0.35 \cdot (\text{hba1c} - 8.2) + 0.85 \cdot \text{prior\_cvd} - 0.02 \cdot (\text{egfr} - 68) + \gamma \cdot U$$
   where $U$ is an unobserved frailty confounder for sensitivity benchmarks.
3. **Potential Outcomes & CATE**:
   - Baseline response: $\mu_0(X) = 2.0 + 0.03 \cdot \text{age} + 0.25 \cdot \text{hba1c} + 0.4 \cdot \text{prior\_cvd} + 0.01 \cdot \text{crp}$
   - Heterogeneous treatment effect:
     $$\tau(X) = 1.8 + 0.8 \cdot (\text{hba1c} - 7.5)_+ + 1.2 \cdot \text{prior\_cvd} + 0.5 \cdot \mathbb{I}(\text{egfr} < 60) - 0.03 \cdot (\text{age} - 60)$$
   - Potential outcomes: $Y(0) = \mu_0(X) + \epsilon_0$, $Y(1) = \mu_0(X) + \tau(X) + \epsilon_1$.
4. **Factual Realization**:
   $$Y = T \cdot Y(1) + (1 - T) \cdot Y(0)$$

## 4. Directory Structure
- `data/raw/`: Generated raw clinical cohorts with full potential outcomes.
- `data/processed/`: Train and test splits with factual outcome and ground truth CATE for benchmarking.
