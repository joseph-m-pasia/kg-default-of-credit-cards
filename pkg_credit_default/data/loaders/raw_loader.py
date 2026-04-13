# Load the raw data from the CSV file

import pandas as pd
def load_data_from_csv(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}")
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
    
    csv_data = load_data_from_csv('data/raw/credit_card_default.csv')       
    
