from load_data import load_data
from exponentialSmoonthing import predict_sales_next_8_weeks_exponential_smoothing
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

    predict_sales_next_8_weeks_exponential_smoothing(file_path, train_end_date='2017-01-01',test_start_date='2017-01-02', num_weeks=8)



# Run the pipeline
sales_analysis_pipeline("input/train.csv")
