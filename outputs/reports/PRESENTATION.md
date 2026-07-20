# WeatherScope AI — Presentation Outline
*(≈ 10 slides, 5–7 minutes; the Streamlit dashboard is the live demo)*

---

## Slide 1 — Title
**Global Weather Trend Forecasting** — 154 k observations · 200+ countries ·
6 models · 8-page dashboard. Fully reproducible: `python main.py`.

## Slide 2 — Problem & Dataset
Forecast temperature, humidity and precipitation from the Kaggle Global
Weather Repository (daily snapshots of world capitals, 40+ features
including six pollutants). May 2024 – July 2026.

## Slide 3 — Pipeline Architecture
Config-driven stages communicating through artifacts:
cleaning → features → {EDA, anomalies, forecasting → ensembles,
importance, climate, air quality, spatial} → report + dashboard.

## Slide 4 — Data Cleaning Decisions
Physical-range validation; per-location imputation; **IQR winsorization
with percentile guards** (42 mm cloudbursts survive, glitches don't);
Isolation Forest flags kept as a column, excluded from training.

## Slide 5 — EDA Highlights
Annual global temperature cycle; zero-inflated precipitation; hottest
(Sahel/Gulf) vs coldest (high-latitude) cities; correlation structure.
*(Figures: distributions_grid, daily_averages, correlation_heatmap.)*

## Slide 6 — Anomaly Detection
Isolation Forest (global outliers) vs DBSCAN (density noise): 39 %
Jaccard overlap; anomalies are real extremes, not sensor errors.
*(Figure: anomalies_pca.)*

## Slide 7 — Forecasting & Evaluation
Six models, honest rolling one-step-ahead evaluation on a 90-day
holdout. Temperature: ARIMA MAE 0.25 °C, R² 0.94. Precipitation ≈ white
noise — quantified negative result. *(Figure: forecast_temperature_celsius.)*

## Slide 8 — Ensembles & Explainability
Weighted ensemble beats all single models for temperature. SHAP:
yesterday's temperature dominates; rolling means, UV and humidity follow.
*(Figures: ensemble_comparison_temperature_celsius, shap_summary.)*

## Slide 9 — Climate, Air Quality & Maps
STL trend and anomalies vs climatology; EPA AQI breakdown and the wind
ventilation effect; Folium cluster map recovers climate archetypes.
*(Live: dashboard Maps page.)*

## Slide 10 — Takeaways & Future Work
Reproducible, explainable, honestly evaluated. Next: per-city neural
forecasters, exogenous drivers, conformal intervals, live retraining.
