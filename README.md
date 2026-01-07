## Problem Context: 

Yes Bank experienced a sharp stock price collapse between 2018 and 2020 driven by governance failures, loss of investor trust, and regulatory intervention. This was not a typical market-driven decline caused by gradual financial underperformance or macroeconomic cycles. Instead, it represented a structural break where historical price behavior and linear assumptions stopped being reliable.

Traditional time-series models such as ARIMA assume stable relationships over time. During the fraud period, these assumptions were violated. This project explicitly models that reality instead of ignoring it.

---

## Objective

The objective is to build a robust and interpretable forecasting framework that:
- Handles structural breaks explicitly
- Demonstrates where classical models fail
- Uses machine learning selectively to correct systematic errors
- Produces stable, conservative long-horizon forecasts
- Makes assumptions transparent to stakeholders

---

## Who This Helps (Stakeholders)

1. Risk and Research Analysts  
These users need to understand whether forecasts are reliable during crisis periods and how sensitive they are to past governance failures. The system helps them compare ARIMA, ML-only, and hybrid forecasts and identify model failure modes.

2. Strategy and Planning Teams  
These users are interested in long-term stabilization versus recovery signals. The system provides conservative projections and makes clear that forecasts are conditional on no new governance shocks.

3. Internal ML / Analytics Teams  
These users benefit from understanding residual diagnostics, feature importance, and why certain models underperform during regime changes.

---

## How the System Helps Stakeholders

The system provides immediate value by:
- Visually exposing how ARIMA overestimates prices during crisis periods
- Showing how ML adapts to nonlinear decline but may overfit if unchecked
- Demonstrating that the hybrid model stabilizes forecasts by correcting systematic ARIMA bias
- Making regime assumptions explicit through the FraudEvent indicator
- Allowing side-by-side comparison of forecasts and backtests
- Enabling CSV export for downstream analysis and reporting

This reduces the risk of overconfident or misleading forecasts and supports informed, cautious decision-making.

---

## Modeling Approach Summary

- ARIMA (1,1,1) models linear trend and short-term autocorrelation after stationarity is ensured.
- Residuals are computed as the difference between actual prices and ARIMA predictions.
- Machine learning models are trained only on residuals, not raw prices, to learn systematic ARIMA errors.
- Random Forest is selected as the final residual model due to robustness to noise, stability under recursion, and better bias–variance balance on limited data.
- Final forecasts are generated as ARIMA mean plus ML-predicted residual correction.
- Recursive forecasting is used to simulate realistic month-by-month updates.

---

## Why Random Forest Was Chosen

Residual errors during crisis periods are noisy, irregular, and regime-dependent. Random Forest builds many independent decision trees using different feature subsets and data samples. Each tree captures a local rule, and the final prediction is an average across trees. This averaging reduces sensitivity to one-off shocks and prevents extreme extrapolation.

In contrast, XGBoost corrects errors sequentially and can amplify rare residual spikes when data is limited, leading to overfitting. In this context, Random Forest’s conservative behavior becomes an advantage.

---

## Evaluation and Interpretation

Evaluation is performed using time-based backtesting on the last 12 months of known data. RMSE is the primary metric because it penalizes large errors, which are especially dangerous in financial contexts. MAE is used as a sanity check, and MAPE is reported with caution due to scale sensitivity during crisis periods.

The hybrid model significantly reduces large forecast errors compared to ARIMA alone during the fraud period, validating the residual correction strategy.

---

## Business Impact:

Its impact lies in:
- Reducing misleading forecasts during structural breaks
- Improving transparency around model assumptions
- Supporting risk-aware long-term planning
- Preventing blind reliance on classical models during crises

---

## Limitations

- Monthly data limits sample size and model complexity
- Recursive forecasting can lead to long-horizon flattening
- No macroeconomic variables are included in the current version
- Forecasts are conditional on no new major governance shock


---

## Future Extensions

- Incorporate macroeconomic indicators such as interest rates, inflation, and banking sector indices
- Add prediction intervals to quantify forecast uncertainty
- Explore regime-switching time-series models
- Implement rolling retraining as new data becomes available
- Productionize with containerization after stakeholder validation

---





