import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
def visualize_monthly_sales(df):
    """
    Visualize the average sales per month using a line plot.

    Parameters:
    df (DataFrame): A pandas DataFrame containing sales data with a 'date' and 'sales' column.

    Returns:
    None
    """
    # Group the data by year and month, and calculate the average sales for each group
    monthly_df = df.groupby([df.date.dt.year, df.date.dt.month])['sales'].mean()

    # Set the names for the index and reset the index to convert it back to a DataFrame
    monthly_df.index = monthly_df.index.set_names(['year', 'month'])
    monthly_df = monthly_df.reset_index()

    # Create x-axis labels in the format 'month/year' for the range 2013 to 2017
    x_axis = ["{}/{}".format(m, y) for y in range(13, 18) for m in range(1, 13)]

    # Create a scatter plot of average sales per month
    plott = go.Scatter(
        x=x_axis, y=monthly_df.sales, mode='lines+markers',
        name='Average sales per month', line=dict(width=3)
    )
    layout = go.Layout(autosize=True, title='Sales - average per month', showlegend=True)
    fig = go.Figure(data=[plott], layout=layout)
    plot(fig)