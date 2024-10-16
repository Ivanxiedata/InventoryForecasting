import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from features import date_features

def predict_sales_next_8_weeks_sarima(train_file, store_num, item_num, train_end_date='2017-01-01', test_start_date='2017-01-02', num_weeks=8, apply_smoothing=False):
    # Load data
    train = pd.read_csv(train_file)
    train['date'] = pd.to_datetime(train['date'])

    # Filter the data by the specified store and item
    train = train[(train['store'] == store_num) & (train['item'] == item_num)]

    # Ensure that 'date' is set as the index
    train.set_index('date', inplace=True)

    # Step 1: Split data into train and test based on the specified date ranges
    train_filtered_date = train[train.index <= train_end_date]
    test_filtered_date = train[train.index >= test_start_date]

    # Debugging: Print out the date ranges to check them
    print("Training Data Date Range: ", train_filtered_date.index.min(), "to", train_filtered_date.index.max())
    print("Test Data Date Range: ", test_filtered_date.index.min(), "to", test_filtered_date.index.max())

    # Step 2: Apply the date_features function to both training and testing data
    train_filtered = date_features(train_filtered_date)
    test_filtered = date_features(test_filtered_date)

    # Apply optional moving average smoothing to reduce noise
    if apply_smoothing:
        train_filtered['sales_smoothed'] = train_filtered['sales'].rolling(window=7).mean()
        y_train = train_filtered['sales_smoothed'].dropna()  # Drop the initial NaN values after smoothing
    else:
        y_train = train_filtered['sales']

    # Drop rows with missing data
    train_filtered = train_filtered.dropna()
    test_filtered = test_filtered.dropna()

    # Step 3: Fit SARIMA model
    # Define seasonal_order based on yearly seasonality (12 months)
    sarima_model = SARIMAX(
        y_train,
        order=(1, 1, 1),  # Non-seasonal order (p, d, q)
        seasonal_order=(1, 1, 1, 12),  # Seasonal order (P, D, Q, s)
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model
    sarima_results = sarima_model.fit(disp=False)

    # Step 4: Predict the next 8 weeks (56 days)
    num_days = num_weeks * 7
    y_pred_last_days = sarima_results.forecast(steps=num_days)

    # Debugging: Check the forecasted values and their dates
    print("Predicted Sales: ", y_pred_last_days)

    # Step 5: Plot actual vs predicted sales for the last 8 weeks using date for alignment
    available_days = len(test_filtered['sales'])
    days_to_select = min(num_days, available_days)

    y_test_last_days = test_filtered['sales'].values[-days_to_select:]

    # Get the corresponding dates from the test data for alignment
    test_dates = test_filtered.index[-days_to_select:]

    # Debugging: Check the dates and values for test data
    print("Test Dates: ", test_dates)
    print("Test Sales (Last Days): ", y_test_last_days)

    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, y_test_last_days, label='Actual Sales (Test)', marker='o')
    plt.plot(test_dates, y_pred_last_days[:days_to_select], label='Predicted Sales (SARIMA)', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'Sales Forecast vs Actual Sales (Next {days_to_select} Days) for Store {store_num}, Item {item_num}')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate the date labels for better readability
    plt.show()

    return sarima_results

# Example usage:
# Apply smoothing, fit SARIMA, and predict for store 1, item 1
# sarima_results = predict_sales_next_8_weeks_sarima('dummy_sales_data.csv', store_num=1, item_num=1, apply_smoothing=True)

