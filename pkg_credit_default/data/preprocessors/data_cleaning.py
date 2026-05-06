from pkg_credit_default.utils.logger import logger

def clean_data(df):

    logger.info("Cleaning data...")

    # Rename PAY_0 to PAY_1 for consistency
    df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)

    return df