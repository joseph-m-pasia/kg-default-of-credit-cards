# Load the raw data from the CSV file

import pandas as pd
from pkg_credit_default.utils.logger import logger


def load_data_from_csv(file_path, expected_columns=None) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data successfully loaded from {file_path}")
    
        # Basic validation
        if df.empty:
            raise ValueError("Loaded dataset is empty")

        # Schema validation (optional but recommended)
        if expected_columns:
            missing = set(expected_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise e

    return df
    
