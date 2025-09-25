import pandas as pd
import numpy as np

def clean_psid_data(filepath):
    """
    Clean PSID data:
    - Select only male heads of household 25 - 60 ma_king > $7 / hr
    - Create indicator variables

    Parameters
    ----------
    Raw PSID data

    Returns
    -------
    pd.DataFrame
    Cleaned PSID data
    """
    cleaned_df = pd.read_stata(filepath)
    
    # filter for male heads of household
    cleaned_df = cleaned_df[cleaned_df["hsex"] == 1]
    
    # age restriction
    cleaned_df = cleaned_df[(cleaned_df["age"] >= 25) & (cleaned_df["age"] <= 60)]
    
    # Only years 1971, 1980, 1990, 2000
    cleaned_df = cleaned_df[cleaned_df["year"].isin([1971, 1980, 1990, 2000])]

    # hourly wage > 7
    cleaned_df["hourly_wage"] = np.where(cleaned_df["hannhrs"] > 0, cleaned_df["hlabinc"] / cleaned_df["hannhrs"], np.nan)
    
    # Restrict to wage > $7
    cleaned_df = cleaned_df[cleaned_df["hourly_wage"] > 7]
    
    # Create indicator variables + quadratic terms // no hispanic variable
    cleaned_df["educ"] = cleaned_df["hyrsed"]
    cleaned_df["age_sq"] = cleaned_df["age"] ** 2
    cleaned_df["white"] = (cleaned_df["hrace"] == 1).astype(int)
    cleaned_df["black"] = (cleaned_df["hrace"] == 2).astype(int)
    cleaned_df["other"] = (cleaned_df["hrace"] == 3).astype(int)
    cleaned_df = cleaned_df[cleaned_df["hrace"].isin([1, 2, 3])]

    return cleaned_df

cleaned_df = clean_psid_data(r"C:\Users\David Munson\Desktop\git\CompEcon_Fall25\Problemsets\PSID_data.dta")