from pkg_credit_default.utils.logger import logger


# non-relevant - bill_amt, age
def _calc_average_balance(df, nmonth):
    logger.info("Calculating average balance over the last {} months...".format(nmonth))
    for i in range(1, nmonth + 1):
        df[f'balance_{i}'] = df[f'BILL_AMT{i}'] - df[f'PAY_AMT{i}']
    balance_vars = [f'balance_{i}' for i in range(1, nmonth + 1)]
    df['AVG_BALANCE_'] = df[balance_vars].mean(axis=1)   
    return df

def _calc_credit_utilization(df, nmonth):
    logger.info("Calculating credit utilization...".format(nmonth))
    # Define credit utilization ratio as average balance divided by credit limit (LIMIT_BAL)
    df['CREDIT_UTILIZATION_'] = df['AVG_BALANCE_'] / df['LIMIT_BAL']
    return df
    
def _calc_late_payment_M1(df, nmonth):
    logger.info("Calculating late payment indicator for M1...".format(nmonth))
    # Define late payment indicator as 1 if the payment in M1 (PAY_1) is late, else 0
    df['LATE_PAYMENT_M1_'] = (df['PAY_1'] > 0).astype(int)
    return df 

def create_new_features(df, config):
    logger.info("Creating new features...")

    nmonth = config["data"]["params"]["n_months"]

    # Average Balance
    df = _calc_average_balance(df, nmonth)

    # Credit Utilization
    df = _calc_credit_utilization(df, nmonth)

    # Late Payment Indicator for M1
    df = _calc_late_payment_M1(df, nmonth)

    return df