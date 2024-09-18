import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
import plotly.graph_objects as go
from plotly.offline import iplot


# Step 1: Load Data
def load_data(file_path):
    data_type = {'store': 'int8', 'item': 'int8', 'sales': 'int16'}
    df = pd.read_csv(file_path, parse_dates=['date'], dtype=data_type)
    print(df.describe())
    return df


# Step 2: Visualize Sales Distribution
def plot_sales_distribution(df):
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales - for each item, date and store")
    sns.histplot(df['sales'], kde=True, color='blue')
    plt.show()


# Step 3: Perform Normality Test
def perform_normality_test(df):
    stat, p_value = st.normaltest(df['sales'])
    print(f"Test Statistic: {stat}")
    print(f"p-value: {p_value}")
    return p_value


# Step 4: Plot Sales Distribution with Best Fit
def plot_best_fit(df):
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales vs best fit normal distribution")
    sns.distplot(df['sales'], fit=st.norm, kde=True, color='g')
    plt.show()


# Step 5: Find Best Fit Distribution
def best_fit_distribution(data, bins=300):
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0  # Finding the bin center

    # List of distributions to check
    DISTRIBUTIONS = [
        st.alpha, st.beta, st.chi, st.chi2, st.dgamma, st.dweibull, st.erlang,
        st.exponweib, st.f, st.genexpon, st.gausshyper, st.gamma, st.johnsonsb,
        st.johnsonsu, st.norm, st.rayleigh, st.rice, st.recipinvgauss, st.t,
        st.weibull_min, st.weibull_max
    ]

    best_distribution = st.norm
    best_sse = np.inf
    best_params = (0.0, 1.0)

    for distribution in DISTRIBUTIONS:
        params = distribution.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

        if best_sse > sse > 0:
            best_sse = sse
            best_distribution = distribution
            best_params = params

    return best_distribution, best_params


# Step 6: Plot Best Fit Distribution
def plot_best_fit_distribution(df, best_distribution, best_params):
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of sales vs best fit distribution")
    sns.distplot(df['sales'], fit=best_distribution, kde=True, color='g')
    plt.show()


# Step 7: Visualize Monthly Sales
def visualize_monthly_sales(df):
    monthly_df = df.groupby([df.date.dt.year, df.date.dt.month])['sales'].mean()
    monthly_df.index = monthly_df.index.set_names(['year', 'month'])
    monthly_df = monthly_df.reset_index()

    x_axis = ["{}/{}".format(m, y) for y in range(13, 18) for m in range(1, 13)]
    plott = go.Scatter(
        x=x_axis, y=monthly_df.sales, mode='lines+markers',
        name='Average sales per month', line=dict(width=3)
    )
    layout = go.Layout(autosize=True, title='Sales - average per month', showlegend=True)
    fig = go.Figure(data=[plott], layout=layout)
    iplot(fig)


# Step 8: Visualize Seasonality by Month
def visualize_seasonality_month_only(df):
    monthly_seasonality = df.groupby(df.date.dt.month)['sales'].mean()

    trace = go.Scatter(
        x=monthly_seasonality.index, y=monthly_seasonality.values,
        mode='lines+markers', name='Average sales per month', line=dict(width=3)
    )

    layout = go.Layout(
        autosize=True, title='Seasonality - average sales per month',
        xaxis=dict(title='Month'), yaxis=dict(title='Average Sales'), showlegend=True
    )

    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)


# step 9: Visualize sale per year
def visualize_year_sale(df):
    yearly_df = df.groupby(df.date.dt.year)['sales'].sum().to_frame()
    trace = go.Bar(
        x=['2013', '2014', '2015', '2016', '2017', '2018'],
        y=yearly_df.sales,
        marker=dict(color='rgba(100, 100, 0, 0.6)', line=dict(color='rgba(10, 13, 0, 1.0)', width=1)),
        name='Total sales by year', orientation='v'
    )

    layout = go.Layout(
        autosize = True, title = 'Total Sales by Year', showlegend = True )

    fig = go.Figure(data=[trace], layout = layout)
    iplot(fig)

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
    iplot(fig)



# Pipeline Execution
def sales_analysis_pipeline(file_path):
    # Step 1: Load data
    df = load_data(file_path)

    # # Step 2: Visualize sales distribution
    # plot_sales_distribution(df)
    #
    # # Step 3: Perform normality test
    # p_value = perform_normality_test(df)
    # print(f'p value is {p_value}')
    #
    # # Step 4: Visualize sales distribution with normal distribution fit
    # plot_best_fit(df)
    #
    # # Step 5: Find best fit distribution
    # best_distribution, best_params = best_fit_distribution(df['sales'].values)
    # print("Best distribution found: {}, with parameters: {}".format(best_distribution.name, best_params))
    #
    # # Step 6: Visualize sales with best fit distribution
    # plot_best_fit_distribution(df, best_distribution, best_params)
    #
    # # Step 7: Visualize monthly sales
    # visualize_monthly_sales(df)

    # Step 8: Visualize seasonality by month
    visualize_seasonality_month_only(df)

    # Step 9: Visualize yearly sales
    visualize_year_sale(df)

    # Step 10: Visualize month and year seaonality
    visualize_seasonality_month_and_year(df)


# Run the pipeline
sales_analysis_pipeline("input/train.csv")
