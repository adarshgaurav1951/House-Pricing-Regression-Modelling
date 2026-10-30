# Ames Real Estate Regression Modeling & Econometric Valuation Platform

An end-to-end, statistically validated property appraisal and econometric diagnostics suite built on the Ames Housing dataset. Unlike typical regression projects that focus solely on single-split leaderboard metrics, this system prioritizes classical econometric assumptions (homoscedasticity, normality of errors, absence of severe multicollinearity via VIF), $L_1/L_2$ regularization, and reproducible production deployment.

---

## 1. Project Architecture & File Hierarchy

```text
House Pricing Regression Modelling/
├── house_prices.csv             # Clean source dataset (Ames Housing, 1,460 observations, 81 attributes)
├── house_price_modelling.ipynb  # Comprehensive diagnostic & modeling notebook (fully executed with outputs)
├── prd.md                       # Product Requirements Document
├── idea.md                      # Statistical rationale, methodology & roadmap blueprint
├── AGENTS.md                    # Operational constraints, governance & linting standards
├── app.py                       # Interactive Streamlit property valuation & diagnostics dashboard
├── README.md                    # Setup guide, benchmarks, and architectural documentation
├── models/
│   ├── elasticnet_pipeline.joblib       # Serialized regularized linear pipeline
│   ├── gradient_boosting_pipeline.joblib# Serialized tree-based non-linear pipeline
│   └── metadata.json                    # Schema definitions, CV metrics, and residual standard error bounds
└── venv/                        # Local Python virtual environment
```

---

## 2. Model Performance Benchmarks (5-Fold Cross-Validation)

| Metric | Regularized ElasticNet ($L_1/L_2$) | HistGradientBoosting (Non-Linear) | Target Baseline |
| :--- | :--- | :--- | :--- |
| **Out-of-Fold $R^2$** | **0.9212** | **0.8969** | $\ge 0.8600$ |
| **Log Error (RMSLE)** | **0.1121** | **0.1283** | $\le 0.1400$ |
| **Mean Absolute Error (MAE)** | **$13,842** | **$15,620** | Minimize |
| **Mean Absolute Percentage Error (MAPE)** | **7.91%** | **9.14%** | $\le 10.5\%$ |
| **Residual Standard Deviation ($\sigma_\epsilon$)** | **0.1121** | **0.1283** | Gaussian spread |
| **Optimal Regularization** | $\alpha = 0.0035, l_1\text{-ratio} = 0.70$ | $L_2 = 0.50, \text{leaf} = 12$ | Cross-validated |

---

## 3. Econometric Assumptions & Statistical Verification

1. **Logarithmic Target Transformation**: Target values are log-transformed ($y' = \log(1+y)$) to correct right-skewness and heteroscedasticity across luxury price tiers.
2. **Multicollinearity Pruning (VIF < 5.0)**:
   - Core structural aggregations (`TotalLivingArea`, `TotalBathrooms`, `PropertyAge`) were tested via Variance Inflation Factor analysis.
   - All final numeric features satisfy $\text{VIF} \le 3.22$, well below the strict threshold of $5.0$.
3. **Homoscedasticity**:
   - Evaluated via the Breusch-Pagan test and residual vs. fitted plots. Residual spread remains stable across all valuation tiers.
4. **Normality of Errors**:
   - Confirmed via Jarque-Bera and Shapiro-Wilk tests; error distribution centers at $\mu_\epsilon \approx 0.0$, justifying $95\%$ empirical prediction intervals ($\pm 1.96 \sigma_\epsilon$).
5. **Outlier Mitigation**:
   - Pruned Dean De Cock's documented non-arm's length sales with $\text{GrLivArea} > 4,000$ sq ft and atypical low pricing.

---

## 4. Setup & Running the Dashboard

### Step 1: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 2: Launch Streamlit Web Application
```bash
streamlit run app.py --server.port 8502
```
The application will launch on `http://localhost:8502`.

---

## 5. Key Dashboard Features (`app.py`)

- **Dual-Model Valuation Simulator**: Real-time side-by-side estimates comparing ElasticNet (interpretable linear) with HistGradientBoosting (non-linear ensemble).
- **95% Confidence Bounds**: Calculated dynamically from out-of-fold residual variance.
- **Econometric Diagnostics Tab**: Interactive Q-Q plots, residual vs fitted plots, and top positive/negative coefficient price drivers.
- **Upgrade ROI Simulator**: Real-time "What-If" sensitivity testing (e.g., adding a bathroom, expanding living space, or upgrading finish quality) with projected equity gains.
- **Fast Inference**: Pre-trained joblib pipelines load once on startup, delivering $< 20\text{ms}$ latency per property appraisal.
