1. how much history
2. do they have seasonality
3. what is the velocity of them? Are most of them high throughtput sales or sparse sales?
    that can tell are these items or stores feasible for me to do exponential smoothing?
    if i aggregrate in item level, what would that look like?
4. could use rolling method or create some lag columns, past day sales, and create each day 

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for heatmap

# Load datasets
train = pd.read_csv('/input/train.csv')
test = pd.read_csv('/input/test.csv')
print("Initial train data:")
print(train.head(10))

# Function to extract date features
def date_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.isocalendar().week
    df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))
    df.drop('date', axis=1, inplace=True)
    return df

# Apply the date_features function
train = date_features(train)
test = date_features(test)

print("Train data after date features extraction:")
print(train.head(10))

# Calculate daily and monthly averages
train['daily_avg'] = train.groupby(['item', 'store', 'dayofweek'])['sales'].transform('mean')
train['monthly_avg'] = train.groupby(['item', 'store', 'month'])['sales'].transform('mean')

train = train.dropna()
print("Train data with daily and monthly averages:")
print(train.head(10))

# Calculate average sales for Day_of_week and Month per Item, Store
daily_avg = train.groupby(['item', 'store', 'dayofweek'])['sales'].mean().reset_index()
monthly_avg = train.groupby(['item', 'store', 'month'])['sales'].mean().reset_index()

# Merge test set with daily and monthly averages
def merge(df1, df2, col, col_name):
    df1 = pd.merge(df1, df2, how='left', on=col)
    df1 = df1.rename(columns={'sales': col_name})
    return df1

test = merge(test, daily_avg, ['item', 'store', 'dayofweek'], 'daily_avg')
test = merge(test, monthly_avg, ['item', 'store', 'month'], 'monthly_avg')




# Generate a heatmap to visualize correlations before dropping features
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Drop highly unnecessary features
for df in [train, test]:
    df.drop(['dayofyear', 'weekofyear', 'daily_avg', 'day', 'month', 'item', 'store'], axis=1, inplace=True)

# Scale features (except target variable)
sales_series = train['sales']
id_series = test['id']

# Compute mean and std / z-score normalization for scaling
mean = train.mean()
std = train.std()

# Apply scaling
train = (train - mean) / std
test = (test - mean) / std

# Restore target and id columns
train['sales'] = sales_series
test['id'] = id_series

# Define features and target
X_train = train.drop('sales', axis=1)
y_train = train['sales']

# Split data into train and validation sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, random_state=123, test_size=0.2
)

# Prepare data for XGBoost
matrix_train = xgb.DMatrix(X_train_split, label=y_train_split)
matrix_test = xgb.DMatrix(X_test_split, label=y_test_split)

# Train XGBoost model
model = xgb.train(params={'objective': 'reg:squarederror', 'eval_metric': 'mae'},
                  dtrain=matrix_train, num_boost_round=500,
                  early_stopping_rounds=20, evals=[(matrix_test, 'test')])

# Make predictions
y_pred = model.predict(matrix_test)

# Select the last 90 days of observations
num_days = 90
available_days = len(y_test_split)
days_to_select = min(num_days, available_days)

y_test_last_days = y_test_split.values[-days_to_select:]
y_pred_last_days = y_pred[-days_to_select:]

# Plot the last 90 days of actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(y_test_last_days, label='Actual Sales', marker='o')
plt.plot(y_pred_last_days, label='Predicted Sales', marker='x')
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title(f'Actual vs Predicted Sales (Last {days_to_select} Days)')
plt.legend()
plt.show()



#future plan
# # Calculate rolling mean for sales
# train['rolling_mean'] = train.groupby('item')['sales'].rolling(window=10).mean().reset_index(level=0, drop=True)
#
# # Add the last 90 days rolling mean sequence to the test data
# rolling_last90 = train.groupby(['item', 'store'])['rolling_mean'].tail(90).reset_index(drop=True)
# test['rolling_mean'] = rolling_last90

# Shift rolling mean by 3 months in the training data
# train['rolling_mean_shifted'] = train.groupby(['item'])['rolling_mean'].shift(90)