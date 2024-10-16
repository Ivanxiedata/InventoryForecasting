import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.offline import plot
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from exponentialSmoonthing import predict_sales_next_holt_winters
from exponentialSmoothingCV import predict_sales_with_cross_validation
from sarima import predict_sales_next_8_weeks_sarima


# Step 1: Load Data
def load_data(file_path):
    data_type = {'store': 'int8', 'item': 'int8', 'sales': 'int16'}
    df = pd.read_csv(file_path, parse_dates=['date'], dtype=data_type)
    print(df.describe())
    return df


# Step 2: Visualize Sales Distribution
def plot_sales_distribution(df):
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales - for each item, date and store")
    sns.histplot(df['sales'], kde=True, color='blue')
    plt.show()


# Step 3: Perform Normality Test
def perform_normality_test(df):
    stat, p_value = st.normaltest(df['sales'])
    print(f"Test Statistic: {stat}")
    print(f"p-value: {p_value}")
    return p_value


# Step 4: Plot Sales Distribution with Best Fit
def plot_best_fit(df):
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales vs best fit normal distribution")
    sns.distplot(df['sales'], fit=st.norm, kde=True, color='g')
    plt.show()


# Step 5: Find Best Fit Distribution
def best_fit_distribution(data, bins=300):
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0  # Finding the bin center

    # List of distributions to check
    DISTRIBUTIONS = [
        st.alpha, st.beta, st.chi, st.chi2, st.dgamma, st.dweibull, st.erlang,
        st.exponweib, st.f, st.genexpon, st.gausshyper, st.gamma, st.johnsonsb,
        st.johnsonsu, st.norm, st.rayleigh, st.rice, st.recipinvgauss, st.t,
        st.weibull_min, st.weibull_max
    ]

    best_distribution = st.norm
    best_sse = np.inf
    best_params = (0.0, 1.0)

    for distribution in DISTRIBUTIONS:
        params = distribution.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

        if best_sse > sse > 0:
            best_sse = sse
            best_distribution = distribution
            best_params = params

    return best_distribution, best_params


# Step 6: Plot Best Fit Distribution
def plot_best_fit_distribution(df, best_distribution, best_params):
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales vs best fit distribution")
    sns.distplot(df['sales'], fit=best_distribution, kde=True, color='g')
    plt.show()


# Step 7: Visualize Monthly Sales
def visualize_monthly_sales(df):
    monthly_df = df.groupby([df.date.dt.year, df.date.dt.month])['sales'].mean()
    monthly_df.index = monthly_df.index.set_names(['year', 'month'])
    monthly_df = monthly_df.reset_index()

    x_axis = ["{}/{}".format(m, y) for y in range(13, 18) for m in range(1, 13)]
    plott = go.Scatter(
        x=x_axis, y=monthly_df.sales, mode='lines+markers',
        name='Average sales per month', line=dict(width=3)
    )
    layout = go.Layout(autosize=True, title='Sales - average per month', showlegend=True)
    fig = go.Figure(data=[plott], layout=layout)
    plot(fig)


# Step 8: Visualize Seasonality by Month
def visualize_seasonality_month_only(df):
    monthly_seasonality = df.groupby(df.date.dt.month)['sales'].mean()

    trace = go.Scatter(
        x=monthly_seasonality.index, y=monthly_seasonality.values,
        mode='lines+markers', name='Average sales per month', line=dict(width=3)
    )

    layout = go.Layout(
        autosize=True, title='Seasonality - average sales per month',
        xaxis=dict(title='Month'), yaxis=dict(title='Average Sales'), showlegend=True
    )

    fig = go.Figure(data=[trace], layout=layout)
    plot(fig)


# step 9: Visualize sale per year
def visualize_year_sale(df):
    yearly_df = df.groupby(df.date.dt.year)['sales'].sum().to_frame()
    trace = go.Bar(
        x=['2013', '2014', '2015', '2016', '2017', '2018'],
        y=yearly_df.sales,
        marker=dict(color='rgba(100, 100, 0, 0.6)', line=dict(color='rgba(10, 13, 0, 1.0)', width=1)),
        name='Total sales by year', orientation='v'
    )

    layout = go.Layout(
        autosize=True, title='Total Sales by Year', showlegend=True)

    fig = go.Figure(data=[trace], layout=layout)
    plot(fig)


def visualize_seasonality_month_and_year(df):
    month_year_seasonality = df.groupby([df.date.dt.month, df.date.dt.year])['sales'].sum()

    # Flatten the multi-index into a single index for plotting
    x_labels = ["{}/{}".format(month, year) for year, month in month_year_seasonality.index]

    trace = go.Scatter(
        x=x_labels, y=month_year_seasonality.values,
        mode='lines+markers', name='Monthly sales across different year', line=dict(width=3)
    )

    layout = go.Layout(
        autosize=True, title='Monthly Sales Seasonality Across Years',
        xaxis=dict(title='Month'), yaxis=dict(title='Total Sales'), showlegend=True
    )

    fig = go.Figure(data=[trace], layout=layout)
    plot(fig)


def visualize_seasonality_month_and_year_per_store(df):
    month_year_store_seasonality = df.groupby(['store', df.date.dt.month, df.date.dt.year])['sales'].mean()

    # Get unique stores
    stores = month_year_store_seasonality.index.get_level_values(0).unique()

    # Create a trace for each store
    data = []
    for store in stores:
        store_data = month_year_store_seasonality[store]
        x_labels = ["{}/{}".format(month, year) for month, year in store_data.index]
        trace = go.Scatter(
            x=x_labels, y=store_data.values,
            mode='lines+markers', name='Store {}'.format(store), line=dict(width=3)
        )
        data.append(trace)

    layout = go.Layout(
        autosize=True, title='Monthly Sales Seasonality Across Years per Store',
        xaxis=dict(title='Month/Year'), yaxis=dict(title='Average Sales'), showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig)


def date_features(df):
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is datetime
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.isocalendar().week
    df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))

    # Remove the line dropping 'date' so that it is retained
    # df.drop('date', axis=1, inplace=True)  <-- Commented out to keep the 'date' column

    return df


# split the train and test data
def predict_sales_next_8_weeks(train_file, train_end_date='2017-01-01', test_start_date='2017-01-02', num_weeks=8):
    # Load data
    train = pd.read_csv(train_file)
    train['date'] = pd.to_datetime(train['date'])

    # Step 1: Split data into train and test based on the specified date ranges
    train_filtered_date = train[train['date'] <= train_end_date]
    test_filtered_date = train[train['date'] >= test_start_date]

    # Step 2: Apply the date_features function to both training and testing data
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

    # Prepare the training and testing datasets
    X_train = train_filtered.drop(['sales'], axis=1)
    y_train = train_filtered['sales']

    X_test = test_filtered.drop(['sales'], axis=1)
    y_test = test_filtered['sales']

    # Prepare data for XGBoost
    matrix_train = xgb.DMatrix(X_train, label=y_train)
    matrix_test = xgb.DMatrix(X_test, label=y_test)

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',  # Regression with squared error
        'eval_metric': 'mae',  # Evaluation metric is mean absolute error
        'eta': 0.01,  # Learning rate
        'max_depth': 6,  # Maximum depth of a tree
        'subsample': 0.8,  # Fraction of data to use per tree
        'colsample_bytree': 0.8,  # Fraction of features to use per tree
        'seed': 42,  # Random seed for reproducibility
        'gamma': 0.1,  # Minimum loss reduction required for a split
    }

    # Dictionary to store evaluation results
    evals_result = {}

    # Train the XGBoost model
    model = xgb.train(
        params=params,
        dtrain=matrix_train,  # Training data
        num_boost_round=500,  # Number of boosting rounds
        early_stopping_rounds=20,  # Stop early if no improvement after 20 rounds
        evals=[(matrix_train, 'train'), (matrix_test, 'test')],  # Track training and test sets
        evals_result=evals_result,  # Store results of each iteration
        verbose_eval=True  # Print evaluation results
    )

    # Step 3: Plot training and validation loss (MAE)
    epochs = len(evals_result['train']['mae'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, evals_result['train']['mae'], label='Training MAE')
    plt.plot(x_axis, evals_result['test']['mae'], label='Validation MAE')
    plt.xlabel('Boosting Round')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.show()

    # Step 4: Predict the last 8 weeks (56 days)
    num_days = num_weeks * 7
    available_days = len(y_test)
    days_to_select = min(num_days, available_days)

    y_test_last_days = y_test.values[-days_to_select:]
    y_pred_last_days = model.predict(matrix_test)[-days_to_select:]

    # Step 5: Plot actual vs predicted sales for the last 8 weeks
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_last_days, label='Actual Sales', marker='o')
    plt.plot(y_pred_last_days, label='Predicted Sales', marker='x')
    plt.xlabel('Day')
    plt.ylabel('Sales')
    plt.title(f'Actual vs Predicted Sales (Last {days_to_select} Days)')
    plt.legend()
    plt.show()


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Keeping date column for Exponential Smoothing, apply date_features only for feature extraction in other models


# def predict_sales_prophet_per_store(train_file, train_end_date, test_start_date, num_weeks=8):
#     """
#     Use Prophet to predict sales for each store using the existing train-test split.
#     Incorporates July as a special event and adds monthly seasonality.
#     """
#     # Load data
#     train = pd.read_csv(train_file)
#     train['date'] = pd.to_datetime(train['date'])  # Ensure 'date' column is in datetime format
#
#     # Step 1: Split data into train and test based on the specified date ranges
#     train_filtered_date = train[train['date'] <= train_end_date]
#     test_filtered_date = train[train['date'] >= test_start_date]
#
#     # Group by store and apply Prophet for each store
#     stores = train_filtered_date['store'].unique()
#
#     for store in stores:
#         print(f"Processing store {store}...")
#
#         # Filter data for the current store
#         store_train_data = train_filtered_date[train_filtered_date['store'] == store]
#         store_test_data = test_filtered_date[test_filtered_date['store'] == store]
#
#         # Prepare the data for Prophet (rename columns as required by Prophet)
#         store_train_data = store_train_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
#         store_test_data = store_test_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
#
#         # Apply a rolling average to smooth sales data
#         store_train_data['y'] = store_train_data['y'].rolling(window=7).mean()
#         store_train_data.dropna(inplace=True)
#
#         # Define the holidays (July event)
#         july_dates = pd.date_range(start='2017-07-01', end='2017-07-31', freq='Y')  # Add for every year
#         holidays = pd.DataFrame({
#             'holiday': 'july_sales_boost',
#             'ds': july_dates,
#             'lower_window': 0,
#             'upper_window': 30
#         })
#
#         # Fit the Prophet model with custom seasonality and holidays
#         model = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays=holidays)
#         model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Monthly seasonality
#         model.fit(store_train_data)
#
#         # Create a dataframe to hold future dates for prediction
#         future = model.make_future_dataframe(periods=len(store_test_data) + (num_weeks * 7), freq='D')
#
#         # Predict sales
#         forecast = model.predict(future)
#
#         # Separate validation predictions (for comparison with test data)
#         test_forecast = forecast.set_index('ds').loc[store_test_data['ds']].reset_index()
#
#         # Calculate MAE on the test set
#         test_mae = mean_absolute_error(store_test_data['y'], test_forecast['yhat'])
#         print(f"Store {store} - Validation MAE (Test set): {test_mae}")
#
#         # Plot the actual vs forecasted sales for validation period
#         plt.figure(figsize=(10, 6))
#         plt.plot(store_train_data['ds'], store_train_data['y'], label=f'Training Sales (Store {store})')
#         plt.plot(store_test_data['ds'], store_test_data['y'], label=f'Actual Test Sales (Store {store})', marker='o')
#         plt.plot(store_test_data['ds'], test_forecast['yhat'], label=f'Forecasted Test Sales (Store {store})', linestyle='--')
#         plt.xlabel('Date')
#         plt.ylabel('Sales')
#         plt.title(f'Prophet Validation Forecast and Actual Sales (Store {store}) (MAE: {test_mae:.2f})')
#         plt.legend()
#         plt.show()
#
#         # Now predict the next 8 weeks
#         future_forecast = forecast.tail(num_weeks * 7)  # Get the forecast for the next num_weeks
#
#         # Plot the future forecast for each store
#         plt.figure(figsize=(10, 6))
#         plt.plot(store_train_data['ds'], store_train_data['y'], label=f'Historical Sales (Store {store})')
#         plt.plot(store_test_data['ds'], store_test_data['y'], label=f'Validation Sales (Store {store})')
#         plt.plot(future_forecast['ds'], future_forecast['yhat'], label=f'Forecasted Sales (Store {store}) (Next {num_weeks} Weeks)', linestyle='--')
#         plt.xlabel('Date')
#         plt.ylabel('Sales')
#         plt.title(f'Prophet Sales Forecast (Store {store}) (Next {num_weeks} Weeks)')
#         plt.legend()
#         plt.show()
#
#     print("Prophet forecasting for all stores completed.")
#

# from prophet import Prophet
# import pandas as pd
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt
# from scipy.stats import johnsonsb
#
# def predict_sales_prophet_with_johnsonsb_per_store(train_file, train_end_date, test_start_date, num_weeks=8):
#     """
#     Use Prophet to predict sales for each store using Johnson's S_b distribution transformation.
#     This includes monthly seasonality and adding July as a special event.
#     """
#     # Load data
#     train = pd.read_csv(train_file)
#     train['date'] = pd.to_datetime(train['date'])  # Ensure 'date' column is in datetime format
#
#     # Step 1: Split data into train and test based on the specified date ranges
#     train_filtered_date = train[train['date'] <= train_end_date]
#     test_filtered_date = train[train['date'] >= test_start_date]
#
#     # Group by store and apply Prophet for each store
#     stores = train_filtered_date['store'].unique()
#
#     for store in stores:
#         print(f"Processing store {store}...")
#
#         # Filter data for the current store
#         store_train_data = train_filtered_date[train_filtered_date['store'] == store]
#         store_test_data = test_filtered_date[test_filtered_date['store'] == store]
#
#         # Prepare the data for Prophet (rename columns as required by Prophet)
#         store_train_data = store_train_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
#         store_test_data = store_test_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
#
#         # Fit Johnson's S_b distribution to sales data
#         params = johnsonsb.fit(store_train_data['y'])
#         a, b, loc, scale = params
#
#         # Transform sales data using Johnson's S_b distribution
#         store_train_data['y'] = johnsonsb(a, b, loc, scale).ppf(store_train_data['y'].rank(pct=True))
#         store_test_data['y_transformed'] = johnsonsb(a, b, loc, scale).ppf(store_test_data['y'].rank(pct=True))
#
#         # Check for NaN values after transformation
#         if store_train_data['y'].isna().sum() > 0:
#             print(f"NaN values found in store {store}'s training data after transformation")
#         if store_test_data['y_transformed'].isna().sum() > 0:
#             print(f"NaN values found in store {store}'s test data after transformation")
#
#         # Fit the Prophet model with custom seasonality and holidays
#         model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
#         model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Monthly seasonality
#         model.fit(store_train_data[['ds', 'y']])
#
#         # Create a dataframe to hold future dates for prediction
#         future = model.make_future_dataframe(periods=len(store_test_data) + (num_weeks * 7), freq='D')
#
#         # Predict sales
#         forecast = model.predict(future)
#
#         # Inverse transform the forecasted values
#         forecast['yhat_original'] = johnsonsb(a, b, loc, scale).ppf(forecast['yhat'].rank(pct=True))
#
#         # Calculate MAE on the test set (using original scale)
#         test_forecast = forecast.set_index('ds').loc[store_test_data['ds']].reset_index()
#         test_mae = mean_absolute_error(store_test_data['y'], test_forecast['yhat_original'])
#         print(f"Store {store} - Validation MAE (Test set): {test_mae}")
#
#         # Plot the actual vs forecasted sales for validation period
#         plt.figure(figsize=(10, 6))
#         plt.plot(store_train_data['ds'], store_train_data['y'], label=f'Training Sales (Store {store})')
#         plt.plot(store_test_data['ds'], store_test_data['y'], label=f'Actual Test Sales (Store {store})', marker='o')
#         plt.plot(store_test_data['ds'], test_forecast['yhat_original'], label=f'Forecasted Test Sales (Store {store})', linestyle='--')
#         plt.xlabel('Date')
#         plt.ylabel('Sales')
#         plt.title(f'Prophet Validation Forecast and Actual Sales (Store {store}) (MAE: {test_mae:.2f})')
#         plt.legend()
#         plt.show(block=True)  # Force display
#
#         # Now predict the next 8 weeks
#         future_forecast = forecast.tail(num_weeks * 7)  # Get the forecast for the next num_weeks
#
#         # Plot the future forecast for each store
#         plt.figure(figsize=(10, 6))
#         plt.plot(store_train_data['ds'], store_train_data['y'], label=f'Historical Sales (Store {store})')
#         plt.plot(store_test_data['ds'], store_test_data['y'], label=f'Validation Sales (Store {store})')
#         plt.plot(future_forecast['ds'], future_forecast['yhat_original'], label=f'Forecasted Sales (Store {store}) (Next {num_weeks} Weeks)', linestyle='--')
#         plt.xlabel('Date')
#         plt.ylabel('Sales')
#         plt.title(f'Prophet Sales Forecast (Store {store}) (Next {num_weeks} Weeks)')
#         plt.legend()
#         plt.show(block=True)  # Force display
#
#     print("Prophet forecasting for all stores completed.")


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def predict_sales_next_8_weeks_exponential_smoothing(train_file, train_end_date='2017-01-01',
                                                     test_start_date='2017-02-01', num_weeks=8):
    # Load data
    train = pd.read_csv(train_file)
    train['date'] = pd.to_datetime(train['date'])

    # Step 1: Split data into train and test based on the specified date ranges
    train_filtered_date = train[train['date'] <= train_end_date]
    test_filtered_date = train[train['date'] >= test_start_date]

    #
    # # Step 2: Apply the date_features function to both training and testing data
    train_filtered = date_features(train_filtered_date)
    test_filtered = date_features(test_filtered_date)

    #
    # # Step 3: Aggregating data at the weekly level instead of monthly
    train_filtered['week'] = train_filtered['date'].dt.to_period('W').apply(lambda r: r.start_time)
    train_filtered = train_filtered.groupby(['week', 'item', 'store'])['sales'].sum().reset_index()
    #
    # # Step 4: Set 'week' as the time index for Exponential Smoothing
    train_filtered.set_index('week', inplace=True)
    #
    # Apply Exponential Smoothing with seasonality (weekly or monthly)
    # Assuming weekly seasonality here
    model = ExponentialSmoothing(
        train_filtered['sales'],
        trend='add',
        seasonal='add',
        seasonal_periods=52  # Yearly seasonality for weekly aggregated data (52 weeks in a year)
    )

    # Fit the model
    fitted_model = model.fit()

    # Step 5: Forecast the next 8 weeks
    forecast = fitted_model.forecast(steps=num_weeks)

    # Step 6: Prepare test data for comparison (aggregating test data at the weekly level)
    test_filtered['week'] = test_filtered['date'].dt.to_period('W').apply(lambda r: r.start_time)
    test_filtered = test_filtered.groupby(['week', 'item', 'store'])['sales'].sum().reset_index()
    test_filtered.set_index('week', inplace=True)  # Ensure test data has the 'week' index

    # Step 7: Plot the forecasted values against the actual values from the test data
    plt.figure(figsize=(10, 6))
    plt.plot(train_filtered.index[-num_weeks:], train_filtered['sales'][-num_weeks:], label='Actual Sales (Train)',
             marker='o')

    plt.plot(test_filtered.index[:num_weeks], test_filtered['sales'][:num_weeks], label='Actual Sales (Test)',
             marker='o')

    plt.plot(forecast.index, forecast, label='Predicted Sales (Holt-Winters)', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'Sales Forecast vs Actual Sales (Next {num_weeks} Weeks)')
    plt.legend()
    plt.show()

    # Return the forecasted values and actual values
    return forecast, test_filtered['sales'][:num_weeks]


# Pipeline Execution
def sales_analysis_pipeline(file_path):
    # Step 1: Load data
    df = load_data(file_path)
    # print('step 1 completed')
    # #
    # # # # Step 2: Visualize sales distribution
    # plot_sales_distribution(df)
    # print('step 2 completed')
    # # #
    # # # # Step 3: Perform normality test
    # p_value = perform_normality_test(df)
    # print(f'p value is {p_value}')
    # print('step 3 completed')
    # # #
    # # # Step 4: Visualize sales distribution with normal distribution fit
    # plot_best_fit(df)
    # print('step 4 completed')
    # #
    # # # Step 5: Find best fit distribution
    # # best_distribution, best_params = best_fit_distribution(df['sales'].values)
    # # print("Best distribution found: {}, with parameters: {}".format(best_distribution.name, best_params))
    # # print('step 5 completed')
    # #
    # # # Step 6: Visualize sales with best fit distribution
    # # plot_best_fit_distribution(df, best_distribution, best_params)
    # # print('step 6 completed')
    # #
    # # Step 7: Visualize monthly sales
    # visualize_monthly_sales(df)
    # print('step 7 completed')
    # #
    # # Step 8: Visualize seasonality by month
    # visualize_seasonality_month_only(df)
    # print('step 8 completed')
    # # #
    # # # # Step 9: Visualize yearly sales
    # visualize_year_sale(df)
    # print('step 9 completed')
    # # #
    # # # Step 10: Visualize month and year seaonality
    # visualize_seasonality_month_and_year(df)
    # print('step 10 completed')
    #
    # # Step 11: Visualize all stores average monthly sales
    # visualize_seasonality_month_and_year_per_store(df)
    # print('step 11 completed')
    #
    # # Step 12: Predict 8 weeks by selecting train dataset til 2017-01-01 from train data
    # predict_sales_next_8_weeks(file_path, train_end_date='2017-01-01', test_start_date='2017-01-02', num_weeks=8)
    #
    # predict_sales_prophet_with_johnsonsb_per_store(file_path, train_end_date='2017-01-01',
    #                                                  test_start_date='2017-01-02', num_weeks=8)

    predict_sales_next_holt_winters(file_path, store_num=1, item_num=1, frequency='M',
                                           train_end_date='2016-12-31',
                                           test_start_date='2017-01-01', step=8)

    # predict_sales_with_cross_validation(file_path, store_num =1, item_num =1, frequency='M', seasonal_period=12, step=8)


# Run the pipeline
sales_analysis_pipeline("input/train.csv")
