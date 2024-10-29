# Step 4: Plot Sales Distribution with Best Fit
def plot_best_fit(df):
    """
    Plot the distribution of sales in the given dataframe, and
    overlay it with the best fit normal distribution.
    """
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales vs best fit normal distribution")
    sns.distplot(df['sales'], fit=st.norm, kde=True, color='g')
    plt.show()