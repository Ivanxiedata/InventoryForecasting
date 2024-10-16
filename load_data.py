import pandas as pd
def load_data(file_path):
    data_type = {'store': 'int8', 'item': 'int8', 'sales': 'int16'}
    df = pd.read_csv(file_path, parse_dates=['date'], dtype=data_type)
    print(df.describe())
    return df