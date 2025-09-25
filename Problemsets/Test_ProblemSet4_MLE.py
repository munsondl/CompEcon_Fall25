# Test_ProblemSet4_MLE.py
import numpy as np
import pandas as pd
from ProblemSet4_Munson_Cleaning_Function import clean_psid_data
from ProblemSet4_MLE import run_mle_by_year

def test_mle_matches_ols():
    """
    Check that MLE estimates are (numerically) the same as OLS estimates.
    """
    # Load and clean data
    cleaned_df = clean_psid_data(r"C:\Users\David Munson\Desktop\git\CompEcon_Fall25\Problemsets\PSID_data.dta")

    # Run MLE estimation
    mle_results = run_mle_by_year(cleaned_df)

    for _, row in mle_results.iterrows():
        year = row["Year"]

        # Prepare data for OLS
        needed = ["hourly_wage", "educ", "age", "age_sq", "black", "other"]
        data = cleaned_df[cleaned_df["year"] == year].dropna(subset=needed)
        y = np.log(data["hourly_wage"].values)
        X = data[["educ", "age", "age_sq", "black", "other"]].values
        X = np.column_stack([np.ones(len(X)), X])  # add intercept

        # OLS beta
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

        # MLE beta (exclude sigma^2)
        beta_mle = np.array([
            row["Intercept"],
            row["Educ"],
            row["Age"],
            row["Age^2"],
            row["Black"],
            row["Other"]
        ])

        # Assert they are close
        np.testing.assert_allclose(beta_mle, beta_ols, rtol=1e-3, atol=1e-3,
            err_msg=f"MLE and OLS estimates differ for year {year}")
