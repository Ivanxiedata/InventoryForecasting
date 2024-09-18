
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

# Calculate rolling mean for sales
train['rolling_mean'] = train.groupby(['item', 'store'])['sales'].rolling(window=10).mean().reset_index(level=0, drop=True)

# Shift rolling mean by 3 months in the training data
train['rolling_mean_shifted'] = train.groupby(['item', 'store'])['rolling_mean'].shift(90)

# Drop the original 'rolling_mean' column
train = train.drop(['rolling_mean'], axis=1)

# Drop NaN values
train = train.dropna()

# For the test set, calculate a rolling mean from the training set's last 90 days sales, and append this to your test set
last_90_days = train.groupby(['item', 'store'])['sales'].tail(90)
test['rolling_mean'] = last_90_days.groupby(['item', 'store']).rolling(window=10).mean().reset_index(level=0, drop=True)

# Then, you can shift it by 90 days
test['rolling_mean_shifted'] = test.groupby(['item', 'store'])['rolling_mean'].shift(90)

# And finally, drop the 'rolling_mean' column
test = test.drop(['rolling_mean'], axis=1)

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