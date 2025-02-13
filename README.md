# Marketing Campaign Performance Dashboard

## Overview
This project analyzes synthetic marketing campaign data to evaluate campaign performance and forecast Customer Lifetime Value (CLV). The workflow involves data cleaning, feature engineering, and predictive modeling using linear regression. The analysis is complemented by a series of visualizations that provide insights into key performance metrics and the overall campaign effectiveness.

## Data Files
- **data/campaign_data.csv**: Raw synthetic campaign data including Date, Campaign, Impressions, Clicks, Conversions, Spend, and CLV.
- **data/cleaned_campaign_data.csv**: Processed dataset with additional computed metrics:
  - **CTR (Click-Through Rate)**
  - **Conversion Rate**
  - **ROI (Return on Investment)**

## Key Visualizations & Analysis
1. **Metric Distributions**  
   - *File:* `results/metrics_distribution.png`  
   - *Insight:* Histograms with KDE plots for CTR, Conversion Rate, and ROI help identify the distribution and central tendencies of these metrics.
   
2. **Correlation Heatmap**  
   - *File:* `results/correlation_heatmap.png`  
   - *Insight:* This heatmap displays the relationships among Impressions, Clicks, Conversions, Spend, CLV, CTR, Conversion Rate, and ROI, highlighting which metrics are most strongly correlated (e.g., ROI with conversions).

3. **Time Series Analysis**  
   - *File:* `results/time_series_ROI.png`  
   - *Insight:* A line plot of total ROI over time (aggregated by date) helps in identifying trends and seasonal effects in campaign performance.

4. **Predictive Modeling**  
   - A linear regression model predicts CLV using CTR, Conversion Rate, and ROI.  
   - *Model Evaluation:* Mean Squared Error (MSE) is printed in the terminal.
   
5. **Model Diagnostics**  
   - *Actual vs. Predicted CLV:* `results/actual_vs_predicted_CLV.png` shows how well the model predicts CLV.
   - *Residual Distribution:* `results/residual_distribution.png` provides a histogram of prediction errors to assess model fit.

## How to Run
1. **Generate Synthetic Data:**  
   Run the provided `data_generator.py` script to create the necessary CSV files in the `data/` folder.
   ```bash
   python data_generator.py
