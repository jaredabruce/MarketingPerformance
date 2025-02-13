# Marketing Campaign Performance Dashboard

## Overview
This project analyzes synthetic marketing campaign data to evaluate campaign performance and forecast Customer Lifetime Value (CLV). The workflow includes data cleaning, feature engineering, and predictive modeling using linear regression. A variety of visualizations provide insights into key performance metrics and overall campaign effectiveness.

## Data Files
- **data/campaign_data.csv**: Raw synthetic campaign data including Date, Campaign, Impressions, Clicks, Conversions, Spend, and CLV.
- **data/cleaned_campaign_data.csv**: Processed dataset with computed metrics:
  - **CTR (Click-Through Rate)**
  - **Conversion Rate**
  - **ROI (Return on Investment)**

## Key Visualizations & Analysis

1. **Metric Distributions**  
   - **Description:** Histograms with KDE plots for CTR, Conversion Rate, and ROI.  
   - **Insight:** Reveals the spread and central tendencies of these performance metrics.  
     ![Metrics Distribution](/results/metrics_distribution.png)

2. **Correlation Heatmap**  
   - **Description:** Displays the correlation among Impressions, Clicks, Conversions, Spend, CLV, CTR, Conversion Rate, and ROI.  
   - **Insight:** Highlights the relationships between metrics, indicating which factors most influence CLV.  
     ![Correlation Heatmap](/results/correlation_heatmap.png)

3. **Time Series Analysis**  
   - **Description:** Line plot of total ROI over time (aggregated by date).  
   - **Insight:** Identifies trends and potential seasonal effects in campaign performance.  
     ![Time Series ROI](/results/time_series_ROI.png)

4. **Predictive Modeling**  
   - **Method:** Linear regression model using CTR, Conversion Rate, and ROI as predictors for CLV.  
   - **Evaluation:** Mean Squared Error (MSE) is printed during execution.  
   - **Visualizations:**  
     - **Actual vs. Predicted CLV:**  
       ![Actual vs Predicted CLV](/results/actual_vs_predicted_CLV.png)  
     - **Residual Distribution:**  
       ![Residual Distribution](/results/residual_distribution.png)

## How to Run

1. **Generate Synthetic Data:**  
   Run the `data_generator.py` script to create the necessary CSV files in the `data/` folder.
   ```bash
   python data_generator.py
