import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose



def decompose(train_file, store_num, item_num, frequency='M', train_end_date='2016-12-31',
                                    test_start_date='2017-01-01', step=8):
    # Load data
    """
    Decompose a time series into trend, seasonality, and residuals using seasonal decomposition.

    Parameters
    ----------
    train_file : str
        The path to the training data CSV file.
    store_num : int
        The store number to filter by.
    item_num : int
        The item number to filter by.
    frequency : str, optional
        The frequency of the time series. Choose either 'M' for monthly or 'W' for weekly.
    train_end_date : str, optional
        The date to split the training data into training and validation sets.
    test_start_date : str, optional
        The date to split the test data into test and validation sets.
    step : int, optional
        The number of periods to forecast.

    Returns
    -------
    None
    """
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

    print("Training Data Date Range: ", train_filtered.index.min(), "to", train_filtered.index.max())
    print("Test Data Date Range: ", test_filtered.index.min(), "to", test_filtered.index.max())

    # Decompose the time series to extract trend, seasonality, and residuals
    decomposition = seasonal_decompose(train_filtered, model='multiplicative', period=12 if frequency == 'M' else 52)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the decomposition
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(train_filtered, label='Original Series')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    return

