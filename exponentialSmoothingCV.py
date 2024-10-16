import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def predict_sales_with_cross_validation(train_file, store_num, item_num, frequency='M', seasonal_period=12, step=8):
    # Load data
    train = pd.read_csv(train_file)
    train['date'] = pd.to_datetime(train['date'])

    # Filter the data by the specified store and item
    train = train[(train['store'] == store_num) & (train['item'] == item_num)]

    # Ensure that 'date' is set as the index
    train.set_index('date', inplace=True)

    # Resample data to weekly or monthly level by summing the sales per week or month
    train_resampled = train['sales'].resample(frequency).sum()

    # TimeSeries Cross-Validation with n_splits
    tscv = TimeSeriesSplit(n_splits=5)

    mae_scores = []

    # Iterate through each cross-validation split
    for train_index, test_index in tscv.split(train_resampled):
        train_cv, test_cv = train_resampled.iloc[train_index], train_resampled.iloc[test_index]

        # Apply Holt-Winters Exponential Smoothing with the specified parameters
        holt_winters_model = ExponentialSmoothing(
            train_cv,
            trend='add',  # Additive trend component
            seasonal='mul',  # Multiplicative seasonal component
            seasonal_periods=seasonal_period  # Weekly or Monthly seasonality
        ).fit()
#
#         # Forecast the next period(s) based on the length of the test set
#         y_pred_next = holt_winters_model.forecast(steps=len(test_cv))
#
#         # Compute the MAE for this split
#         mae = mean_absolute_error(test_cv, y_pred_next)
#         mae_scores.append(mae)
#
#         # Debugging: Print MAE for each split
#         print(f"Split MAE: {mae}")
#
#     # Compute the average MAE across all splits
#     avg_mae = sum(mae_scores) / len(mae_scores)
#     print(f'Average MAE across all splits: {avg_mae}')
#
#     return avg_mae
#
#
# # Example of usage
# avg_mae = predict_sales_with_cross_validation(
#     train_file='path_to_your_data.csv',
#     store_num=1,
#     item_num=1,
#     frequency='M',  # 'M' for monthly, 'W' for weekly
#     seasonal_period=12,  # Seasonal period for monthly data
#     step=8  # Number of future periods to forecast
# )
