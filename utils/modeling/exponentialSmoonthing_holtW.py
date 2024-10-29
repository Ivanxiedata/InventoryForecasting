import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

def predict_sales_next_holt_winters(train_file, store_num, item_num, frequency='M', train_end_date='2016-12-31', test_start_date='2017-01-01', step=8):
    # Load data
    train = pd.read_csv(train_file)
    train['date'] = pd.to_datetime(train['date'])

    # Filter the data by the specified store and item
    train = train[(train['store'] == store_num) & (train['item'] == item_num)]

    # Ensure that 'date' is set as the index
    train.set_index('date', inplace=True)

    # Resample data to weekly or monthly level by summing the sales per week or month
    train_resampled = train['sales'].resample(frequency).sum()

    # Split data into train and test based on the specified date ranges
    train_filtered = train_resampled[train_resampled.index <= train_end_date]
    test_filtered = train_resampled[train_resampled.index >= test_start_date]

    # Define the seasonal period
    seasonal_period = 12 if frequency == 'M' else 52

    # Fit the Holt-Winters model
    holt_winters_model = ExponentialSmoothing(
        train_filtered,
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_period
    ).fit()

    # Forecast the next 'step' periods
    y_pred_next = holt_winters_model.forecast(steps=step)

    # Limit the forecast to the available test data
    available_periods = len(test_filtered)
    periods_to_select = min(step, available_periods)

    # Prepare test values for MAE calculation
    y_test_last_periods = test_filtered.values[:periods_to_select]
    test_dates = test_filtered.index[:periods_to_select]

    # Compute the MAE
    mae = mean_absolute_error(y_test_last_periods, y_pred_next[:periods_to_select])
    print(f"Mean Absolute Error (MAE): {mae}")

    # Plot the actual vs predicted sales
    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, y_test_last_periods, label='Actual Sales (Test)', marker='o')
    plt.plot(test_dates, y_pred_next[:periods_to_select], label='Predicted Sales (Holt-Winters)', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'Sales Forecast vs Actual Sales (Next {periods_to_select} Periods) for Store {store_num}, Item {item_num}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    return holt_winters_model
