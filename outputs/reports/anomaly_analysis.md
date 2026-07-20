# Anomaly Detection Analysis

## Detector comparison

- Common sample size: **30,000** observations
- Isolation Forest flagged: **300**
- DBSCAN (noise points) flagged: **241**
- Flagged by both: **152** (Jaccard agreement 39.1%)

Isolation Forest isolates points that are *globally* easy to separate,
while DBSCAN flags points in *locally sparse* regions. Their overlap is
therefore the set of observations that are extreme by both definitions.

## Why Isolation Forest anomalies occur

Mean z-score of flagged observations (|z| > 1 marks a driver):

- `precip_mm`: **+4.34**
- `pressure_mb`: **-1.66**
- `wind_kph`: **+1.31**
- `uv_index`: **+0.36**
- `temperature_celsius`: **+0.15**
- `humidity`: **-0.05**

Most affected countries: Kuwait, Iceland, Iraq, Qatar, Norway.
Dominant conditions: Sunny, Light rain, Partly cloudy, Moderate rain, Overcast.

## Why DBSCAN anomalies occur

Mean z-score of flagged observations (|z| > 1 marks a driver):

- `precip_mm`: **+5.32**
- `wind_kph`: **+1.43**
- `pressure_mb`: **-0.99**
- `temperature_celsius`: **-0.69**
- `humidity`: **+0.62**
- `uv_index`: **-0.35**

Most affected countries: Iceland, Norway, New Zealand, Serbia, Ireland.
Dominant conditions: Light rain, Partly cloudy, Moderate rain, Moderate or heavy rain shower, Light rain shower.

## Interpretation

Flagged records are dominated by physically extreme but real weather:
heavy monsoon rainfall, desert heat with near-zero humidity, and
high-wind storm events. They represent rare joint combinations
(e.g. high temperature *and* high rainfall) rather than sensor errors,
which is why they survive the cleaning stage's physical-range checks.
