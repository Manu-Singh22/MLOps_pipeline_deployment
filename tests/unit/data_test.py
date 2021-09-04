# test integrity of the input data
import os
import numpy as np
import pandas as pd

# get absolute path of csv files from data folder
def get_absPath(filename):
    """Returns the path of the notebooks folder"""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", filename
        )
    )
    return path

# number of features
expected_columns = 10

# distribution of features in the training set
historical_mean = np.array[(850.0256103,
8.466373314,
4.817578461,
75.83282605,
4.435011928,
0.720986971,
15.93033209,
37.24344938,
38.4099894,
27.3731353,
51.30808239,
51.80777047,
50.12557815,
725.0367761,
234.5831629,
567.6925336,
843.2950736,
231.5002685,
584.1165537,
821.5094833,
90.26036448,
111.1956672,
36.51180917,
7.48021018,
7.22484615,
0.078856764,
0.017118789,
0.042967041,
])



historical_std = np.array([17.27363304,
3.549206458,
4.087505761,
13.77791977,
1.243103349,
0.45722278,
2.701799493,
7.237995905,
8.59365581,
4.102795621,
3.9189577,
3.414244438,
3.291732789,
35.91182598,
60.11277038,
113.6062311,
18.71709301,
50.03971774,
110.4030001,
17.34966987,
19.69652402,
42.872054,
7.131863106,
0.114386096,
0.111080391,
0.010433812,
0.003959271,
0.004633413,
])
# maximal relative change in feature mean or standrd deviation
# that we can tolerate
shift_tolerance = 3

def test_check_schema():
    datafile = get_absPath("diabetes.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns == expected_columns


def test_check_bad_schema():
    datafile = get_absPath("diabetes_bad_schema.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns != expected_columns


def test_check_missing_values():
    datafile = get_absPath("diabetes_missing_values.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    n_nan = np.sum(np.isnan(dataset.values))
    assert n_nan > 0


def test_check_distribution():
    datafile = get_absPath("diabetes_bad_dist.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    mean = np.mean(dataset.values, axis=0)
    std = np.mean(dataset.values, axis=0)
    assert (
        np.sum(abs(mean - historical_mean) > shift_tolerance * abs(historical_mean))
        or np.sum(abs(std - historical_std) > shift_tolerance * abs(historical_std)) > 0
    )
