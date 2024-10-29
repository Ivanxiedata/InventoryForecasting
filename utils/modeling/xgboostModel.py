def predict_sales_next_8_weeks_xgb(train_file, train_end_date='2017-01-01', test_start_date='2017-01-02', num_weeks=8):
    """
    Predict the sales for the next 8 weeks using XGBoost.

    Parameters
    ----------
    train_file : str
        The path to the training data CSV file.
    train_end_date : str, optional
        The date to split the training data into training and validation sets.
    test_start_date : str, optional
        The date to split the test data into test and validation sets.
    num_weeks : int, optional
        The number of weeks to predict.

    Returns
    -------
    None
    """
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
