# Product Requirements Document (PRD)

## Project Title
**House Price Regression Modeling, Statistical Diagnostics & Valuation System**

---

## 1. Executive Summary & Vision

### 1.1 Problem Statement
Conventional machine learning workflows for real estate price prediction disproportionately emphasize black-box accuracy on isolated test splits while ignoring classical econometric assumptions. In real estate appraisal, asset valuation, and risk assessment, unverified models pose severe business risks:
- Multicollinear predictors lead to erratic coefficient estimates and unstable marginal price effects.
- Non-normal, heteroscedastic residuals produce invalid confidence bounds and systematically misprice properties at luxury and entry-level tiers.
- Overfitted complex tree architectures lack explainability and fail to generalize when spatial or structural market dynamics shift.

### 1.2 Vision & Objectives
This system delivers an end-to-end, statistically validated real estate pricing framework that balances **rigorous econometrics** with **modern regularized machine learning**. The core deliverables are:
1. **Clean Ames Dataset (`house_prices.csv`)**: Standardized 81-feature dataset containing structural, spatial, and temporal attributes.
2. **Diagnostic & Modeling Pipeline (`house_price_modelling.ipynb`)**: Executed diagnostic notebook validating Ordinary Least Squares (OLS) assumptions (normality, homoscedasticity, linearity, multicollinearity via VIF), performing regularized optimization (Ridge, Lasso, ElasticNet), and benchmarking against non-linear ensemble regressors.
3. **Production Model Artifacts (`models/`)**: Fully serialized `sklearn.pipeline.Pipeline` objects and comprehensive schema metadata for low-latency inference.
4. **Interactive Valuation & Diagnostics Web Application (`app.py`)**: A Streamlit dashboard offering dual-model real-time valuation, 95% prediction intervals, and interactive residual diagnostics.

---

## 2. Stakeholders & User Personas

| Persona | Role | Primary Needs | Key Features Used |
| :--- | :--- | :--- | :--- |
| **Real Estate Appraiser** | Property valuation & portfolio auditing | Transparent pricing estimates, 95% confidence intervals, and condition sensitivity | Valuation Tool, Dual-Model comparison, Prediction intervals |
| **Quantitative Risk Analyst** | Mortgage underwriting & stress testing | Statistical validation of residual variance and homoscedasticity across price tiers | Diagnostic Tab, Breusch-Pagan metrics, Residual vs Fitted plots |
| **Data Scientist / ML Engineer** | Model governance and pipeline maintenance | Reproducible preprocessing pipelines, clean schema contracts, automated serialization | Model metadata, Joblib pipelines, Jupyter diagnostic notebook |

---

## 3. System Architecture & Directory Blueprint

```text
House Pricing Regression Modelling/
├── house_prices.csv             # Structured Ames source dataset (1460 observations, 81 attributes)
├── house_price_modelling.ipynb  # Comprehensive diagnostic & modeling notebook (fully executed)
├── prd.md                       # Product Requirements Document (this specification)
├── idea.md                      # Project blueprint, statistical rationale & methodological roadmap
├── AGENTS.md                    # Operational constraints, linting rules & execution guidelines
├── app.py                       # Interactive Streamlit property valuation & diagnostics dashboard
├── README.md                    # Setup, methodology, validation metrics, and user guide
├── models/
│   ├── elasticnet_pipeline.joblib       # Regularized linear pipeline (statistically verified)
│   ├── gradient_boosting_pipeline.joblib# Tree-based non-linear benchmark pipeline
│   └── metadata.json                    # Feature schemas, target transformation parameters & diagnostics
└── venv/                        # Local Python virtual environment
```

---

## 4. Functional Requirements

### 4.1 Data Ingestion & Target Treatment
- **FR-1.1 Ingestion**: Ingest the Ames housing dataset (`1460` records, `80` features + `SalePrice`).
- **FR-1.2 Target Transformation**: Apply logarithmic transformation $y' = \log(1 + y)$ (`np.log1p`) to correct right-skewed pricing distributions and stabilize variance.
- **FR-1.3 Inverse Transformation**: Provide exact inverse transformations ($e^{y'} - 1$) at inference time to return predictions in USD ($).

### 4.2 Feature Engineering & Multicollinearity Filtering
- **FR-2.1 Domain Feature Synthesis**:
  - `TotalLivingArea`: $\text{GrLivArea} + \text{TotalBsmtSF}$
  - `HouseAge`: $\text{YrSold} - \text{YearBuilt}$
  - `RemodelAge`: $\text{YrSold} - \text{YearRemodAdd}$
  - `TotalBathrooms`: $\text{FullBath} + 0.5 \times \text{HalfBath} + \text{BsmtFullBath} + 0.5 \times \text{BsmtHalfBath}$
  - `PorchArea`: $\text{OpenPorchSF} + \text{EnclosedPorch} + \text{3SsnPorch} + \text{ScreenPorch}$
- **FR-2.2 Multicollinearity Diagnostics**:
  - Compute Variance Inflation Factor (VIF) on numeric predictors.
  - Apply automated collinearity pruning for features with $\text{VIF} > 5.0$ or pairwise Pearson correlation $|r| > 0.80$ (e.g., resolving `GarageCars` vs `GarageArea`, `TotalBsmtSF` vs `1stFlrSF`).

### 4.3 Statistical Assumption Diagnostics
Prior to model acceptance, the pipeline must compute and report:
- **FR-3.1 Normality of Residuals**: Shapiro-Wilk test, Jarque-Bera test, and Quantile-Quantile (Q-Q) residuals plot.
- **FR-3.2 Homoscedasticity**: Breusch-Pagan and White test statistics; residual vs. predicted fitted plot verifying constant error variance across all pricing tiers.
- **FR-3.3 Influence & Outliers**: Cook's Distance and Leverage analysis to identify anomalous commercial or non-arm's-length sales.

### 4.4 Model Development & Regularization
- **FR-4.1 Candidate Algorithms**:
  1. **OLS Baseline**: Benchmark model for coefficient interpretations and standard diagnostic tests.
  2. **Ridge Regression ($L_2$)**: Addressing multicollinearity through shrinkage.
  3. **Lasso Regression ($L_1$)**: Inducing feature sparsity and automated selection.
  4. **ElasticNet Regression ($L_1 + L_2$)**: Optimal balance of sparsity and grouping effects.
  5. **HistGradientBoosting Regressor**: High-capacity non-linear baseline to measure interaction effects.
- **FR-4.2 Cross-Validation Strategy**:
  - 5-Fold Repeated Cross-Validation on log-transformed targets.
  - Evaluation metrics: $R^2$, Root Mean Squared Logarithmic Error (RMSLE), Mean Absolute Percentage Error (MAPE), and Mean Absolute Error (MAE).

### 4.5 Production Pipeline Serialization
- **FR-5.1 Artifact Export**: Export self-contained `scikit-learn` pipelines using `joblib`:
  - `models/elasticnet_pipeline.joblib`
  - `models/gradient_boosting_pipeline.joblib`
- **FR-5.2 Metadata Contract**: Export `models/metadata.json` capturing:
  - Input features, data types, and default reference values.
  - Categorical column levels for form validation.
  - Target scaling parameters and cross-validated residual standard error ($\sigma_\epsilon$) for 95% prediction interval computation ($1.96 \times \sigma_\epsilon$).
  - Out-of-fold validation scores ($R^2$, RMSLE, MAE).

### 4.6 Interactive Web Dashboard (`app.py`)
- **FR-6.1 Property Valuation Simulator**:
  - Input controls for property attributes: Square footage, Quality Rating (`OverallQual` 1-10), Condition (`OverallCond` 1-9), Neighborhood, Year Built, Remodel Year, Bathrooms, Bedrooms, Garage capacity, Basement area.
- **FR-6.2 Dual-Model Inference**:
  - Real-time side-by-side display of ElasticNet and Gradient Boosting predictions.
  - Percentage divergence indicator between linear and non-linear estimates.
- **FR-6.3 95% Confidence & Prediction Intervals**:
  - Dynamic lower and upper valuation bands derived from empirical residual variance.
- **FR-6.4 Statistical Diagnostics Suite**:
  - Interactive Q-Q plot and residual distribution curve.
  - Top positive and negative coefficient price drivers from the regularized model.
  - Feature correlation overview.

---

## 5. Non-Functional Requirements

### 5.1 Performance & Latency
- Dashboard load time $< 2.0$ seconds.
- Single property inference latency $< 50$ milliseconds.
- Serialized pipelines must load without requiring retraining on app startup.

### 5.2 Interpretability & Transparency
- Every feature fed to the regularized model must have an interpretable coefficient sign and economic rationale.
- The UI must display the key value drivers explaining how physical attributes affect price.

### 5.3 Reliability & Robustness
- Preprocessing pipelines must handle missing input values gracefully via median and mode imputation.
- Out-of-vocabulary categorical inputs must map safely to the most frequent baseline category or an 'Other' bucket.

---

## 6. Acceptance Criteria & Target Metrics

| Requirement | Metric | Target Threshold | Validation Method |
| :--- | :--- | :--- | :--- |
| Model Generalization | Out-of-fold $R^2$ | $\ge 0.86$ | 5-Fold Repeated CV |
| Log Error | Cross-validated RMSLE | $\le 0.140$ | 5-Fold Repeated CV |
| Percentage Error | Cross-validated MAPE | $\le 10.5\%$ | Out-of-fold holdout |
| Homoscedasticity | Residual vs Fitted plot | No funneling pattern | Breusch-Pagan p-value & visual plot |
| App Stability | Zero runtime errors | 100% test pass | Headless browser & input boundary tests |
| Cleanliness | Unnecessary files eliminated | Zero duplicate/unneeded files | Directory audit |

---

## 7. Deliverables & Implementation Checklist

- [x] **PRD**: Formalized requirements document (`prd.md`).
- [ ] **Data Asset**: Source dataset saved to `house_prices.csv`.
- [ ] **Architecture Blueprint**: Project blueprint and rationale documented in `idea.md`.
- [ ] **Agent Guidelines**: Execution and linting rules documented in `AGENTS.md`.
- [ ] **Diagnostic Notebook**: Fully executed `house_price_modelling.ipynb` displaying assumptions, VIF, CV benchmarks, and residual plots.
- [ ] **Serialized Models**: Calibrated pipelines and `metadata.json` in `models/`.
- [ ] **Interactive Dashboard**: Streamlit application in `app.py`.
- [ ] **Documentation**: Comprehensive `README.md` covering setup and user guide.
- [ ] **Workspace Cleanup**: Redundant, duplicate, and unneeded files purged.
