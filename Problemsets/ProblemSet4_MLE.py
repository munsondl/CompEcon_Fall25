# ProblemSet4_MLE.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ProblemSet4_Munson_Cleaning_Function import clean_psid_data

# ------------------------------------------------------------
# 1. Gaussian log-likelihood function
# ------------------------------------------------------------
def log_likelihood(params, X, y):
    """
    Negative log-likelihood for linear regression with normal errors.

    Parameters
    ----------
    params : array
        [beta0, beta1, ..., betaK, theta] where sigma^2 = exp(theta)
    X : ndarray
        Design matrix (n x k)
    y : ndarray
        Dependent variable (n,)
    """
    beta = params[:-1]
    theta = params[-1]        # unrestricted parameter
    sigma2 = np.exp(theta)    # enforce positivity

    residuals = y - X @ beta
    n = len(y)
    ll = -0.5 * n * np.log(2 * np.pi * sigma2) - (residuals @ residuals) / (2 * sigma2)
    return -ll  # minimize negative LL

# ------------------------------------------------------------
# 2. Back-transform coefficients to raw units
# ------------------------------------------------------------
def back_transform_coefs(beta_scaled, means, stds):
    """
    Convert coefficients estimated on standardized variables back to raw scale.

    Parameters
    ----------
    beta_scaled : ndarray
        Coefficients from scaled regression (includes intercept).
    means : dict
        Means of variables used in scaling.
    stds : dict
        Standard deviations of variables used in scaling.

    Returns
    -------
    beta_raw : ndarray
        Coefficients on raw scale (same length as beta_scaled).
    """
    intercept = beta_scaled[0]
    beta_educ = beta_scaled[1]

    # Adjust for scaling of age
    beta_age = beta_scaled[2] / stds["age"]
    beta_age2 = beta_scaled[3] / stds["age_sq"]

    # Adjust intercept to undo centering
    intercept = (
        intercept
        - beta_age * means["age"]
        - beta_age2 * means["age_sq"]
    )

    beta_black = beta_scaled[4]
    beta_other = beta_scaled[5]

    return np.array([intercept, beta_educ, beta_age, beta_age2, beta_black, beta_other])

# ------------------------------------------------------------
# 3. Run MLE for each year
# ------------------------------------------------------------
def run_mle_by_year(df):
    results = []

    for year in [1971, 1980, 1990, 2000]:
        needed = ["hourly_wage", "educ", "age", "age_sq", "black", "other"]
        data = df[df["year"] == year].dropna(subset=needed)

        n_obs = len(data)
        if n_obs == 0:
            print(f"⚠️ No usable data for {year}")
            continue

        print(f"Year {year}: {n_obs} observations")

        # Dependent variable: log wage
        y = np.log(data["hourly_wage"].values)

        # Independent variables
        X_raw = data[["educ", "age", "age_sq", "black", "other"]].copy()

        # Save means/stds for scaling
        means = {
            "age": X_raw["age"].mean(),
            "age_sq": X_raw["age_sq"].mean()
        }
        stds = {
            "age": X_raw["age"].std(),
            "age_sq": X_raw["age_sq"].std()
        }

        # Standardize age and age_sq
        X_raw["age"] = (X_raw["age"] - means["age"]) / stds["age"]
        X_raw["age_sq"] = (X_raw["age_sq"] - means["age_sq"]) / stds["age_sq"]

        X = X_raw.values
        X = np.column_stack([np.ones(len(X)), X])  # add intercept

        # Initial guesses from OLS
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta_ols
        sigma2_init = np.var(resid, ddof=1) if n_obs > 1 else 1.0
        if sigma2_init <= 0 or np.isnan(sigma2_init):
            sigma2_init = 1.0
        theta_init = np.log(sigma2_init)

        init_params = np.append(beta_ols, theta_init)

        # Optimize likelihood
        res = minimize(
            log_likelihood,
            init_params,
            args=(X, y),
            method="L-BFGS-B"
        )

        # Show diagnostic info
        print(f"  Optimization success: {res.success}, Message: {res.message}")

        if res.success:
            beta_scaled = res.x[:-1]
            beta_raw = back_transform_coefs(beta_scaled, means, stds)
            sigma2 = np.exp(res.x[-1])
        else:
            beta_raw = [np.nan] * 6
            sigma2 = np.nan

        row = {
            "Year": year,
            "Success": res.success,
            "LogLik": -res.fun if res.success else np.nan,
            "Intercept": beta_raw[0],
            "Educ": beta_raw[1],
            "Age": beta_raw[2],
            "Age^2": beta_raw[3],
            "Black": beta_raw[4],
            "Other": beta_raw[5],
            "Sigma^2": sigma2,
        }
        results.append(row)

    return pd.DataFrame(results)

# ------------------------------------------------------------
# 4. Main execution
# ------------------------------------------------------------
if __name__ == "__main__":
    # Load and clean the PSID data
    cleaned_df = clean_psid_data(r"C:\Users\David Munson\Desktop\git\CompEcon_Fall25\Problemsets\PSID_data.dta")

    # Print counts per year to check availability
    print("\nObservations by year after cleaning:")
    print(cleaned_df["year"].value_counts())

    # Run MLE estimation
    results_df = run_mle_by_year(cleaned_df)

    # Print results summary
    print("\nMLE Results:")
    if results_df.empty:
        print("No valid results (check data cleaning).")
    else:
        print(results_df.to_string(index=False))
