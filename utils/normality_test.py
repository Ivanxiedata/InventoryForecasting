# Step 3: Perform Normality Test
def perform_normality_test(df):
    """
    Perform the D'Agostino and Pearson's chi-squared test for normality.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'sales' column to be tested for normality.

    Returns:
    p_value (float): The p-value of the normality test.

    Notes:
    The D'Agostino and Pearson's chi-squared test is a statistical test that checks whether a given dataset
    follows a normal distribution. The test is based on the skewness and kurtosis of the data.
    """
    stat, p_value = st.normaltest(df['sales'])
    print(f"Test Statistic: {stat}")
    print(f"p-value: {p_value}")
    return p_value
