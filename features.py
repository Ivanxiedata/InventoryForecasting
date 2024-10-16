import pandas as pd
import numpy as np
# def date_features(df):
#     df['date'] = pd.to_datetime(df['date'])
#     df['year'] = df.date.dt.year
#     df['month'] = df.date.dt.month
#     df['day'] = df.date.dt.day
#     df['dayofyear'] = df.date.dt.dayofyear
#     df['dayofweek'] = df.date.dt.dayofweek
#     df['weekofyear'] = df.date.dt.isocalendar().week
#     df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))
#     return df
def date_features(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    return df
