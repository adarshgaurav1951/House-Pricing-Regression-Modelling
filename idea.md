# Project Blueprint: House Price Regression Modeling & Statistical Validation

## 1. Executive Summary & Philosophy
While typical regression tutorials prioritize chasing leaderboard metrics (like maximizing single-split $R^2$), real estate valuation demands statistical reliability, interpretability, and adherence to regression assumptions. 

This project expands upon standard regression workflows by implementing rigorous diagnostic testing (homoscedasticity, normality of residuals, multicollinearity), feature selection via statistical significance and penalized likelihood, regularization tuning (Ridge, Lasso, ElasticNet), and robust cross-validation (Repeated K-Fold / Out-of-Fold validation) to guarantee models generalize beyond random splits.

---

## 2. Project Directory Structure

```text
House Pricing Regression Modelling/
├── house_prices.csv             # Clean source dataset (Ames Housing dataset, 1460 rows, 81 cols)
├── house_price_modelling.ipynb  # Comprehensive diagnostic & modeling notebook (executed)
├── prd.md                       # Product Requirements Document
├── idea.md                      # Project blueprint, statistical rationale & roadmap
├── AGENTS.md                    # Agent constraints, linting rules & execution guidelines
├── app.py                       # Interactive Streamlit property valuation & diagnostics dashboard
├── README.md                    # Setup, methodology, validation metrics, and user guide
├── models/
│   ├── elasticnet_pipeline.joblib       # Regularized linear pipeline (statistically verified)
│   ├── gradient_boosting_pipeline.joblib# Tree-based non-linear benchmark pipeline
│   └── metadata.json                    # Feature schemas, target transformation parameters & diagnostics
└── venv/                        # Local Python virtual environment
```

---

## 3. Core Objectives & Key Differentiators

| Dimension | Standard Regression (e.g., Used Cars) | House Pricing Regression (This Project) |
| :--- | :--- | :--- |
| **Primary Goal** | Compare baseline vs. advanced regressors | Validate model reliability, stability, and statistical assumptions |
| **Target Treatment** | Direct prediction of raw target | Logarithmic transformation $\log(1 + y)$ to stabilize variance and normalize errors |
| **Collinearity** | Often ignored or handled implicitly by trees | Explicit Variance Inflation Factor (VIF) filtering and correlation clustering |
| **Regularization** | Default hyperparameters | Explicit $L_1$/$L_2$ shrinkage paths, coefficient stability checks, and sparsity induction |
| **Validation** | Single Train-Test Split (e.g., 80/20) | Repeated K-Fold CV, residual distribution tests, and cross-tier stability |

---

## 4. Methodological Phases & Roadmap

### Phase 1: Exploratory Data Analysis & Target Distribution
* **Target Skewness Correction:** Analyze `SalePrice` skewness. Apply $\log(1 + y)$ transformation to address right-skewed pricing and heteroscedasticity.
* **Geographical & Spatial Clustered Pricing:** Examine price distribution across neighborhoods, school districts, and spatial clusters.
* **Missingness Mechanics:** Distinguish missing values that indicate property features (e.g., `No Basement`, `No Garage`) versus true missing data.

### Phase 2: Feature Engineering & Statistical Pruning
* **Derived Quality & Spatial Metrics:**
  * Total Living Area ($ \text{GrLivArea} + \text{TotalBsmtSF} $).
  * Effective Age vs. Remodel Age ($ \text{YearSold} - \text{YearRemodAdd} $).
  * Bathroom totals (Full + $0.5 \times$ Half).
  * Total Porch Area (Open + Enclosed + 3Ssn + Screen).
* **Multicollinearity Diagnostics:**
  * Compute Variance Inflation Factor (VIF). Drop or combine features exceeding $\text{VIF} > 5.0$.
  * Correlation matrix thresholding for overlapping structural metrics (e.g., `GarageCars` vs. `GarageArea`).

### Phase 3: Regression Assumption Verification
Rigorous evaluation of the classical Ordinary Least Squares (OLS) assumptions prior to finalizing deployment candidates:
1. **Linearity:** Pairwise scatterplots and partial residual plots.
2. **Homoscedasticity:** Breusch-Pagan and White tests on residuals.
3. **Normality of Residuals:** Quantile-Quantile (Q-Q) plots and Shapiro-Wilk / Jarque-Bera tests.
4. **Independence & Outliers:** Cook's Distance and Leverage analysis to identify and handle anomalous sales.

### Phase 4: Regularization & Model Training
* **Candidate Models:**
  * **Baseline:** OLS Linear Regression (for benchmark diagnostics).
  * **Regularized Linear Models:** Ridge ($L_2$), Lasso ($L_1$), and ElasticNet with cross-validated $\alpha$ and $l_1\text{-ratio}$.
  * **Non-linear Competitor:** HistGradientBoosting for capturing structural non-linearities and interactions.
* **Cross-Validation Strategy:**
  * 5-Fold Repeated Cross-Validation on log-transformed targets.
  * Evaluate Root Mean Squared Logarithmic Error (RMSLE), Mean Absolute Percentage Error (MAPE), and $R^2$ stability across folds.

### Phase 5: Production Pipeline Serialization & Artifact Generation
* Encapsulate one-hot encoding, target scalers/transformers, median imputers, and estimators into `sklearn.pipeline.Pipeline`.
* Export calibrated models:
  * `elasticnet_pipeline.joblib`: Parsimonious, regularized, interpretable model.
  * `gradient_boosting_pipeline.joblib`: High-capacity, non-linear model.
* Save `metadata.json` containing:
  * Input schema types and allowed categorical values.
  * Transformation parameters ($\mu, \sigma$, log flags).
  * Out-of-fold validation scores and 95% prediction interval error bounds.

### Phase 6: Interactive Dashboard (`app.py`)
Build a modern Streamlit web application providing:
* **Interactive Valuation Tool:** Sliders and dropdowns for property specifications (Square Footage, Quality Grade, Neighborhood, Age, Room counts).
* **Dual-Model Inference:** Real-time side-by-side comparison between ElasticNet and Gradient Boosting predictions.
* **Confidence & Prediction Intervals:** Displaying 95% bounds calculated from cross-validated residual variance.
* **Model Diagnostic Tab:** Interactive Q-Q plots, residual vs. predicted charts, and top coefficient impact summaries.

---

## 5. Deliverables & Acceptance Criteria

1. **`house_prices.csv`:** Structured dataset containing standard real estate tabular features.
2. **`house_price_modelling.ipynb`:** Fully reproducible notebook displaying explicit assumption checks, VIF steps, CV scores, and residual diagnostic plots.
3. **`models/` Directory:** Clean joblib pipelines along with schema metadata ready for app consumption.
4. **`app.py`:** Functional Streamlit app loading pre-trained pipelines without on-the-fly model fitting.
5. **Statistical Stability Target:** 
   * Out-of-fold cross-validation $R^2 \ge 0.86$.
   * Residual mean $\approx 0$, with no severe funneling across price tiers (homoscedastic residuals).
