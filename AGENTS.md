# Agent Guidelines & Engineering Standards

## 1. Scope & Execution Principles
This workspace houses the statistical regression modeling and property valuation system for the Ames Housing dataset.
All modifications and automation scripts must adhere to strict reproducibility, statistical validity, and zero-redundancy rules.

## 2. Directory & File Governance
- Only maintain essential project files according to the standard architecture:
  - `house_prices.csv`: Source tabular data.
  - `house_price_modelling.ipynb`: Executed diagnostic and modeling Jupyter notebook.
  - `prd.md`: Product Requirements Document.
  - `idea.md`: Statistical methodology and architecture blueprint.
  - `AGENTS.md`: Operational standards and development rules.
  - `app.py`: Streamlit dashboard and property valuation simulator.
  - `README.md`: Setup, evaluation benchmarks, and run guide.
  - `models/`: Production pipelines (`elasticnet_pipeline.joblib`, `gradient_boosting_pipeline.joblib`) and `metadata.json`.
  - `venv/`: Local virtual environment.
- **Do not create** redundant scratch directories, duplicate notebook copies, or unneeded cache dumps.

## 3. Modeling & Statistical Rigor
- **Logarithmic Target Transformation**: Target values must be log-transformed ($y' = \log(1+y)$) before linear regression modeling and inversely transformed ($e^{y'} - 1$) for human evaluation and dashboard reporting.
- **VIF & Multicollinearity**: Never feed unpruned collinear features into OLS without verifying Variance Inflation Factors ($\text{VIF} < 5.0$).
- **Assumption Verification**: Residuals must be checked for normality (Shapiro-Wilk / Q-Q plot) and homoscedasticity (Breusch-Pagan / residual plots).
- **Out-of-Fold Evaluation**: Performance must be reported with 5-Fold Cross Validation. Baseline threshold: $R^2 \ge 0.86$.

## 4. Web Application Standards
- The Streamlit app (`app.py`) must load pre-trained pipelines from `models/` without retraining on startup.
- Fast inference latency ($< 50\text{ms}$) per evaluation.
- Polished, responsive design with dark mode styling, intuitive input grouping, and side-by-side model comparison.
