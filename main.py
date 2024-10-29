import seaborn as sns
import scipy.stats as st
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import matplotlib.pyplot as plt
from utils.decompose import decompose
from utils.load_data import load_data
from utils.visualize_sale_distribution import plot_sales_distribution
from utils.normality_test import perform_normality_test
from utils.plot_best_fit import plot_best_fit
from utils.modeling.xgboostModel import predict_sales_next_8_weeks_xgb
from utils.bestFit import best_fit_distribution
from utils.visualizeMonthlySale import visualize_monthly_sales
from utils.modeling.exponentialSmoonthing_holtW import predict_sales_next_holt_winters
from utils.visualize_montly_sale_only import visualize_seasonality_month_only
from utils.visualize_year_sale import visualize_year_sale, visualize_seasonality_month_and_year, visualize_seasonality_month_and_year_per_store
from utils.date_features import date_features

# Pipeline Execution
def sales_analysis_pipeline(file_path):
    # Step 1: Load data
    df = load_data(file_path)
    print('step 1 completed')

    # Visualization Section
    def perform_visualizations(data):
        # Step 2: Visualize sales distribution
        plot_sales_distribution(data)
        print('step 2 completed')

        # Step 3: Perform normality test
        p_value = perform_normality_test(data)
        print(f'Normality test p-value: {p_value}')
        print('step 3 completed')

        # Step 4: Visualize sales distribution with normal distribution fit
        plot_best_fit(data)
        print('step 4 completed')

        # Step 5: Find and visualize best fit distribution
        best_distribution, best_params = best_fit_distribution(data['sales'].values)
        print(f"Best distribution: {best_distribution.name} with parameters: {best_params}")
        print('step 5 completed')

        # Step 6: Visualize monthly sales
        visualize_monthly_sales(data)
        print('step 6 completed')

        # Step 7: Visualize seasonality by month
        visualize_seasonality_month_only(data)
        print('step 7 completed')

        # Step 8: Visualize yearly sales
        visualize_year_sale(data)
        print('step 8 completed')

        # Step 9: Visualize month and year seasonality
        visualize_seasonality_month_and_year(data)
        print('step 9 completed')

        # Step 10: Visualize all stores' average monthly sales
        visualize_seasonality_month_and_year_per_store(data)
        print('step 10 completed')

    # Modeling Section
    def perform_modeling(data_path):
        # Step 11: Predict next 8 weeks using XGBoost model
        # predict_sales_next_8_weeks_xgb(data_path, train_end_date='2017-01-01', test_start_date='2017-01-02', num_weeks=8)
        # print('step 11 completed')

        # Step 12: Decompose the time series
        decompose(data_path, store_num=1, item_num=1, frequency='M', train_end_date='2016-12-31', test_start_date='2017-01-01', step=8)
        print('step 12 completed')

        # Step 13: Predict sales using Holt-Winters exponential smoothing
        predict_sales_next_holt_winters(data_path, store_num=1, item_num=1, frequency='M', train_end_date='2016-12-31', test_start_date='2017-01-01', step=8)
        print('step 13 completed')

    # Execute visualizations and modeling
    # perform_visualizations(df)
    perform_modeling(file_path)

# Run the pipeline
sales_analysis_pipeline("input/train.csv")
