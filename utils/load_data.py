import pandas as pd
def load_data(file_path):
    """
    Load sales data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the sales data.

    Returns:
    DataFrame: A pandas DataFrame containing the sales data with specified dtypes and parsed dates.
    """
    # Define the data types for columns to optimize memory usage
    data_type = {'store': 'int8', 'item': 'int8', 'sales': 'int16'}
    
    # Read the CSV file into a DataFrame, parsing the 'date' column as datetime
    df = pd.read_csv(file_path, parse_dates=['date'], dtype=data_type)
    
    # Print summary statistics of the DataFrame
    print(df.describe())
    
    # Return the loaded DataFrame
    return df