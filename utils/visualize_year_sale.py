# step 9: Visualize sale per year
from plotly.offline import plot
import plotly.graph_objs as go
def visualize_year_sale(df):
    """
    Visualize the total sales per year.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be visualized.

    Returns
    -------
    None
    """
    # Group the data by year and sum the values
    yearly_df = df.groupby(df.date.dt.year)['sales'].sum().to_frame()

    # Bar chart
    trace = go.Bar(
        x=['2013', '2014', '2015', '2016', '2017', '2018'],
        y=yearly_df.sales,
        marker=dict(
            color='rgba(100, 100, 0, 0.6)',
            line=dict(color='rgba(10, 13, 0, 1.0)', width=1)
        ),
        name='Total sales by year',
        orientation='v'
    )

    # Layout
    layout = go.Layout(
        autosize=True,
        title='Total Sales by Year',
        showlegend=True
    )

    # Plot the figure
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
