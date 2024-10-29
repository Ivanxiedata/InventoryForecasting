# Step 8: Visualize Seasonality by Month
from plotly.offline import plot
import plotly.graph_objects as go

def visualize_seasonality_month_only(df):
    """
    Visualize the seasonality of the sales data at the monthly level.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing the sales data with a 'date' and 'sales' column.

    Returns
    -------
    None
    """
    # Group the data by the month and calculate the average sales for each group
    monthly_seasonality = df.groupby(df.date.dt.month)['sales'].mean()

    # Create a scatter plot of the average sales per month
    trace = go.Scatter(
        x=monthly_seasonality.index, y=monthly_seasonality.values,
        mode='lines+markers', name='Average sales per month', line=dict(width=3)
    )

    # Set the layout for the plot
    layout = go.Layout(
        autosize=True, title='Seasonality - average sales per month',
        xaxis=dict(title='Month'), yaxis=dict(title='Average Sales'), showlegend=True
    )

    # Create a figure with the trace and layout
    fig = go.Figure(data=[trace], layout=layout)

    # Plot the figure
    plot(fig)