import pandas as pd
import numpy as np

from ProblemSet4_Munson_Cleaning_Function import clean_psid_data

def test_cleaned_data():
    cleaned = clean_psid_data(r"C:\Users\David Munson\Desktop\git\CompEcon_Fall25\Problemsets\PSID_data.dta")

    # Example assertions
    assert (cleaned["hsex"] == 1).all()         # all male
    assert cleaned["age"].between(25, 60).all() # ages between 25–60
    assert (cleaned["hourly_wage"] > 7).all()   # all wages > 7
    assert cleaned["year"].isin([1971, 1980, 1990, 2000]).all() # valid years

def test_sum_to_one():
    cleaned = clean_psid_data(r"C:\Users\David Munson\Desktop\git\CompEcon_Fall25\Problemsets\PSID_data.dta")
    # Row sums should equal 1
    row_sums = cleaned[["white", "black", "other"]].sum(axis=1)
    assert (row_sums == 1).all(), "Each row should have exactly one indicator = 1"