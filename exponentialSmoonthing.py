import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from features import date_features


def predict_sales_next_holt_winters(train_file, store_num, item_num, frequency='M', train_end_date='2016-12-31',
                                    test_start_date='2017-01-01', step=8):
    # Load data
    train = pd.read_csv(train_file)
    train['date'] = pd.to_datetime(train['date'])

    # Filter the data by the specified store and item
    train = train[(train['store'] == store_num) & (train['item'] == item_num)]

    # Ensure that 'date' is set as the index
    train.set_index('date', inplace=True)

    # Step 1: Resample data to weekly or monthly level by summing the sales per week or month
    train_resampled = train['sales'].resample(frequency).sum()

    # Step 2: Split data into train and test based on the specified date ranges
    train_filtered = train_resampled[train_resampled.index <= train_end_date]
    test_filtered = train_resampled[train_resampled.index >= test_start_date]

    # Debugging: Print out the date ranges to check them
    print("Training Data Date Range: ", train_filtered.index.min(), "to", train_filtered.index.max())
    print("Test Data Date Range: ", test_filtered.index.min(), "to", test_filtered.index.max())

    if frequency == 'M':
        seasonal_period = 12
    elif frequency == 'W':
        seasonal_period = 52
    else:
        raise ValueError("Invalid frequency. Choose either 'M' for monthly or 'W' for weekly.")

    # Step 3: Apply Holt-Winters Exponential Smoothing with weekly or monthly seasonality
    holt_winters_model = ExponentialSmoothing(
        train_filtered,
        trend='add',  # Additive trend component
        seasonal='mul',  # Multiplicative seasonal component to account for growing seasonality
        seasonal_periods=seasonal_period  # Weekly or Monthly seasonality
    ).fit()

    # Step 4: Predict the next 8 weeks/months
    y_pred_next = holt_winters_model.forecast(steps=step)

    # Debugging: Check the forecasted values
    print("Predicted Sales (Next Periods): ", y_pred_next)

    # Step 5: Plot actual vs predicted sales for the next 8 weeks/months
    available_periods = len(test_filtered)
    periods_to_select = min(step, available_periods)

    y_test_last_periods = test_filtered.values[:periods_to_select]
    test_dates = test_filtered.index[:periods_to_select]

    # Debugging: Check the dates and values for test data
    print("Test Dates: ", test_dates)
    print("Test Sales (Last Periods): ", y_test_last_periods)

    # Step 6: Compute the MAE
    mae = mean_absolute_error(y_test_last_periods, y_pred_next[:periods_to_select])
    print(f"Mean Absolute Error (MAE): {mae}")

    # Step 7: Plot the actual vs predicted sales
    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, y_test_last_periods, label='Actual Sales (Test)', marker='o')
    plt.plot(test_dates, y_pred_next[:periods_to_select], label='Predicted Sales (Holt-Winters)', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(
        f'Sales Forecast vs Actual Sales (Next {periods_to_select} Periods) for Store {store_num}, Item {item_num}')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate the date labels for better readability
    plt.show()

    return holt_winters_model
