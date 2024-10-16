import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Step 1: Generate dummy data and save to CSV
def generate_dummy_data():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Create a date range for a few months
    dates = pd.date_range(start='2016-01-01', end='2017-01-31')

    # Repeat the date range for multiple stores and items
    num_stores = 2
    num_items = 2
    total_entries = len(dates) * num_stores * num_items

    # Create a DataFrame with dummy sales data
    data = {
        'date': np.tile(dates, num_stores * num_items),  # Repeat dates for each store/item combination
        'store': np.repeat(np.arange(1, num_stores + 1), len(dates) * num_items),  # Repeat store IDs
        'item': np.tile(np.repeat(np.arange(1, num_items + 1), len(dates)), num_stores),  # Repeat item IDs
        'sales': np.random.randint(20, 100, size=total_entries)  # Random sales between 20 and 100
    }

    # Create the DataFrame
    dummy_sales_data = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    dummy_sales_data.to_csv('dummy_sales_data.csv', index=False)

    return 'dummy_sales_data.csv'

# Example usage:
train_file = generate_dummy_data()
print(pd.read_csv(train_file).head())  # Load and preview the data

# Step 2: Apply Holt-Winters Exponential Smoothing with monthly seasonality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Step 2: Apply Holt-Winters Exponential Smoothing with monthly seasonality
def predict_sales_next_8_weeks_holt_winters_monthly(train_file, train_end_date='2017-01-01',
                                                    test_start_date='2017-01-02', num_weeks=8):
    # Load data
    train = pd.read_csv(train_file)
    train['date'] = pd.to_datetime(train['date'])

    # Ensure that 'date' is set as the index
    train.set_index('date', inplace=True)

    # Step 1: Split data into train and test based on the specified date ranges
    train_filtered_date = train[train.index <= train_end_date]
    test_filtered_date = train[train.index >= test_start_date]

    # Step 2: Apply the date_features function to both training and testing data
    def date_features(df):
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
        return df

    train_filtered = date_features(train_filtered_date)
    test_filtered = date_features(test_filtered_date)

    # Calculate daily and monthly averages for both sets
    train_filtered['daily_avg'] = train_filtered.groupby(['item', 'store', 'dayofweek'])['sales'].transform('mean')
    train_filtered['monthly_avg'] = train_filtered.groupby(['item', 'store', 'month'])['sales'].transform('mean')

    test_filtered['daily_avg'] = test_filtered.groupby(['item', 'store', 'dayofweek'])['sales'].transform('mean')
    test_filtered['monthly_avg'] = test_filtered.groupby(['item', 'store', 'month'])['sales'].transform('mean')

    # Drop rows with missing data
    train_filtered = train_filtered.dropna()
    test_filtered = test_filtered.dropna()

    # Prepare the training dataset
    y_train = train_filtered['sales']

    # Step 3: Apply Holt-Winters Exponential Smoothing with monthly seasonality
    holt_winters_model = ExponentialSmoothing(
        y_train,
        trend='add',  # Additive trend component
        seasonal='add',  # Additive seasonal component
        seasonal_periods=30  # Monthly seasonality (~30 days)
    ).fit()

    # Step 4: Predict the next 8 weeks (56 days)
    num_days = num_weeks * 7
    y_pred_last_days = holt_winters_model.forecast(steps=num_days)

    # Step 5: Plot actual vs predicted sales for the last 8 weeks using date for alignment
    available_days = len(test_filtered['sales'])
    days_to_select = min(num_days, available_days)

    y_test_last_days = test_filtered['sales'].values[-days_to_select:]

    # Get the corresponding dates from the test data for alignment
    test_dates = test_filtered.index[-days_to_select:]

    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, y_test_last_days, label='Actual Sales', marker='o')
    plt.plot(test_dates, y_pred_last_days[:days_to_select], label='Predicted Sales', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'Actual vs Predicted Sales (Last {days_to_select} Days) - Holt Winters Exponential Smoothing (Monthly Seasonality)')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate the date labels for better readability
    plt.show()

    return holt_winters_model

# Generate dummy data and save to CSV
train_file = generate_dummy_data()

# Predict sales for the next 8 weeks using Holt-Winters Exponential Smoothing
predict_sales_next_8_weeks_holt_winters_monthly(train_file)
