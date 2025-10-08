"""
estimate_model.py

Contains `estimate_models(df, out_tex_path)` which estimates:
- Model 1: OLS T5YIE ~ FEDFUNDS
- Model 2: OLS T5YIE ~ FEDFUNDS + FEDFUNDS_lag1
- Model 3: OLS T5YIE ~ FEDFUNDS + CPI_YoY + UNRATE

Saves a LaTeX table to ../results_table.tex and returns fitted results.
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

RESULTS_PATH = Path(__file__).resolve().parent / "results_table.tex"

def estimate_models(df, out_tex_path=RESULTS_PATH):
    """
    Estimate three OLS models and save a combined latex table.

    Args:
        df (pd.DataFrame): cleaned DataFrame from clean_data()
        out_tex_path (Path or str): path to save the latex table.

    Returns:
        dict: {'model1': res1, 'model2': res2, 'model3': res3}
    Side effects:
        - writes out_tex_path with a LaTeX table of coefficients.
    """
    # Prepare variables and lags
    data = df.copy()
    data['FEDFUNDS_L1'] = data['FEDFUNDS'].shift(1)
    # Drop na for models
    data = data.dropna(subset=['T5YIE', 'FEDFUNDS', 'FEDFUNDS_L1', 'CPI_YoY', 'UNRATE'])

    y = data['T5YIE']

    # Model 1
    X1 = sm.add_constant(data[['FEDFUNDS']])
    res1 = sm.OLS(y, X1).fit()

    # Model 2
    X2 = sm.add_constant(data[['FEDFUNDS', 'FEDFUNDS_L1']])
    res2 = sm.OLS(y, X2).fit()

    # Model 3
    X3 = sm.add_constant(data[['FEDFUNDS', 'CPI_YoY', 'UNRATE']])
    res3 = sm.OLS(y, X3).fit()

    # Build a pandas DataFrame summary table for LaTeX (simple approach)
    rows = []
    model_names = ['Model1', 'Model2', 'Model3']
    models = [res1, res2, res3]
    for name, res in zip(model_names, models):
        coefs = res.params
        ses = res.bse
        for var in coefs.index:
            rows.append({
                'model': name,
                'variable': var,
                'coef': coefs[var],
                'se': ses[var],
                'r2': res.rsquared
            })
    table = pd.DataFrame(rows)
    # Pivot to a readable latex table summarizing coefficients per model
    # We'll produce a compact table of coefficients and SEs stacked.
    def make_coef_se_block(res):
        params = res.params
        bse = res.bse
        out = {}
        for p in params.index:
            out[f"{p}"] = f"{params[p]:.4f} ({bse[p]:.4f})"
        out['R2'] = f"{res.rsquared:.4f}"
        return out

    summary_dicts = [make_coef_se_block(r) for r in models]
    summary_df = pd.DataFrame(summary_dicts, index=model_names).fillna("")

    # Save to latex
    out_tex_path = Path(out_tex_path)
    out_tex_path.write_text(summary_df.to_latex(index=True, caption="Regression results: T5YIE on policy and controls", label="tab:results"))

    return {'model1': res1, 'model2': res2, 'model3': res3, 'summary_df': summary_df}

if __name__ == "__main__":
    # Ensure the current folder (Problemset5/) is on sys.path so local modules can be imported
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))

    # Import the local clean_data function (clean_data.py should be in the same folder)
    from clean_data import clean_data

    df = clean_data()
    results = estimate_models(df)
    print("Saved results table to:", RESULTS_PATH)
    print(results['summary_df'])
