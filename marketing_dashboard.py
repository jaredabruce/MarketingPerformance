import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # Ensure required directories exist
    os.makedirs('results', exist_ok=True)
    
    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    data = pd.read_csv('data/campaign_data.csv')
    data.dropna(inplace=True)
    
    # Feature Engineering
    data['CTR'] = data['Clicks'] / data['Impressions']
    data['Conversion_Rate'] = data['Conversions'] / data['Clicks']
    data['ROI'] = data['Conversions'] * 100 - data['Spend']  # Assuming $100 per conversion

    # Convert Date to datetime (for time-series analysis)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
    
    print("Sample of Processed Campaign Data:")
    print(data.head())
    
    # -----------------------------------
    # 2. Additional Visualizations & Analysis
    # -----------------------------------
    sns.set(style="whitegrid")
    
    # Visualization A: Distributions of CTR, Conversion Rate, and ROI
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(data['CTR'], bins=20, kde=True, ax=axes[0]).set_title('CTR Distribution')
    sns.histplot(data['Conversion_Rate'], bins=20, kde=True, ax=axes[1]).set_title('Conversion Rate Distribution')
    sns.histplot(data['ROI'], bins=20, kde=True, ax=axes[2]).set_title('ROI Distribution')
    plt.tight_layout()
    plt.savefig('results/metrics_distribution.png')
    plt.close()

    # Visualization B: Correlation Heatmap for selected metrics
    plt.figure(figsize=(10, 8))
    corr = data[['Impressions', 'Clicks', 'Conversions', 'Spend', 'CLV', 'CTR', 'Conversion_Rate', 'ROI']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('results/correlation_heatmap.png')
    plt.close()

    # Visualization C: Time Series of Total ROI Over Time (if Date exists)
    if 'Date' in data.columns:
        roi_time = data.groupby('Date')['ROI'].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y='ROI', data=roi_time, marker='o')
        plt.title('Total ROI Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total ROI')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/time_series_ROI.png')
        plt.close()

    # -----------------------------------
    # 3. Predictive Modeling: Linear Regression for CLV
    # -----------------------------------
    features = data[['CTR', 'Conversion_Rate', 'ROI']]
    target = data['CLV']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse:.2f}')
    
    # Visualization D: Scatter Plot (Actual vs. Predicted CLV)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel('Actual CLV')
    plt.ylabel('Predicted CLV')
    plt.title('Actual vs. Predicted CLV')
    # Diagonal reference line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.savefig('results/actual_vs_predicted_CLV.png')
    plt.close()
    
    # Visualization E: Residual Distribution
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Error')
    plt.ylabel('Frequency')
    plt.savefig('results/residual_distribution.png')
    plt.close()

    # -----------------------------------
    # 4. Export Processed Data
    # -----------------------------------
    data.to_csv('data/cleaned_campaign_data.csv', index=False)
    print("Cleaned data exported to data/cleaned_campaign_data.csv")

if __name__ == '__main__':
    main()
