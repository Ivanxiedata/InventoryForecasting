import matplotlib.pyplot as plt
import seaborn as sns
# Step 2: Visualize Sales Distribution
def plot_sales_distribution(df):
    """
    Plot the distribution of sales for each item, date and store using seaborn histplot.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the sales data.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales - for each item, date and store")
    sns.histplot(df['sales'], kde=True, color='blue')
    # Show the plot
    plt.show()