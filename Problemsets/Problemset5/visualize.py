"""
visualize.py

Contains `create_visuals(df, out_dir)` which creates and saves three figures:
1) Time series of FEDFUNDS and T5YIE
2) Scatter with fitted linear regression
3) 12-month rolling correlation plot

Each figure is saved into ../images/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

def create_visuals(df, out_dir=IMAGES_DIR):
    """
    Create visuals and save them to disk.

    Args:
        df (pd.DataFrame): DataFrame returned by clean_data (monthly index).
        out_dir (Path or str): directory to save images.

    Returns:
        list of Path: list with saved image file paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    saved = []

    # 1) Time series plot
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df.index, df['T5YIE'], label='5y Breakeven (T5YIE)')
    ax1.set_ylabel('5y Breakeven (%) or points')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['FEDFUNDS'], color='tab:orange', label='Fed Funds Rate')
    ax2.set_ylabel('Fed Funds Rate (%)')
    ax1.set_title('T5YIE and Effective Federal Funds Rate (Monthly)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fname1 = out_dir / "fig1_timeseries.png"
    fig.tight_layout()
    fig.savefig(fname1, dpi=200)
    plt.close(fig)
    saved.append(fname1)

    # 2) Scatter with fitted line
    fig, ax = plt.subplots(figsize=(6,6))
    x = df['FEDFUNDS'].values
    y = df['T5YIE'].values
    ax.scatter(x, y, alpha=0.6)
    # fit linear model
    X = add_constant(x)
    model = OLS(y, X).fit()
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    y_line = model.params[0] + model.params[1]*x_line
    ax.plot(x_line, y_line, label=f"Fit: y={model.params[1]:.3f}x+{model.params[0]:.3f}")
    ax.set_xlabel('Fed Funds Rate (%)')
    ax.set_ylabel('5y Breakeven (T5YIE)')
    ax.set_title('Scatter: Fed Funds Rate vs 5y Breakeven')
    ax.legend()
    fname2 = out_dir / "fig2_scatter.png"
    fig.tight_layout()
    fig.savefig(fname2, dpi=200)
    plt.close(fig)
    saved.append(fname2)

    # 3) Rolling correlation (12-month)
    rolling = df[['T5YIE', 'FEDFUNDS']].rolling(window=12).corr().unstack().iloc[:,1]  # correlation series
    # The above is a bit awkward due to multiindex; compute explicitly:
    corr = df['T5YIE'].rolling(12).corr(df['FEDFUNDS'])
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(corr.index, corr.values)
    ax.set_title('12-month Rolling Correlation: T5YIE and FEDFUNDS')
    ax.set_ylabel('Correlation')
    ax.set_ylim(-1,1)
    fname3 = out_dir / "fig3_rollingcorr.png"
    fig.tight_layout()
    fig.savefig(fname3, dpi=200)
    plt.close(fig)
    saved.append(fname3)

    return saved

if __name__ == "__main__":
    # ensure the current folder is on the import path so we can import local modules
    import sys
    from pathlib import Path

    # add the script's folder (Problemset5/) to sys.path so local modules can be imported
    sys.path.append(str(Path(__file__).resolve().parent))

    # now import the local clean_data function (file clean_data.py should be in same folder)
    from clean_data import clean_data

    df = clean_data()
    files = create_visuals(df)
    print("Saved images:", files)

