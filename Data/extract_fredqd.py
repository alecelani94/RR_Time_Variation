"""
Python mirror of extract_fredqd.m. Produces fredqd_glp.mat with the same
Y{1}, Y{2}, Y{3}, Ynames, Ydates structure that the MATLAB script writes.

Run from Data/:  py extract_fredqd.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio


HERE = Path(__file__).parent

# auto-pick latest *-QD.csv in this folder
csv_files = sorted(HERE.glob("*-QD.csv"))
if not csv_files:
    raise SystemExit(f"No *-QD.csv files found in {HERE}")
csv_in = csv_files[-1]
print(f"Using CSV: {csv_in.name}")

# auto-detect delimiter from first line
with csv_in.open() as f:
    first = f.readline()
delim = ";" if first.count(";") > first.count(",") else ","

# Line 1: series mnemonics; Lines 2-3: factor indicators + transformation codes (skip)
df = pd.read_csv(csv_in, sep=delim, skiprows=[1, 2])
series = list(df.columns[1:])
raw_dates = pd.to_datetime(df.iloc[:, 0], format="%m/%d/%Y")
raw_values = df.iloc[:, 1:].to_numpy(dtype=float)


def vars_block(rows):
    """rows: list of (fred_mnemonic, label, transform)"""
    return rows


small_3_vars = vars_block([
    ("GDPC1",        "RGDP",          "4log"),
    ("GDPCTPI",      "PGDP",          "4log"),
    ("FEDFUNDS",     "FedFunds",      "raw"),
])

small_4_vars = small_3_vars + vars_block([
    ("UNRATE",       "UnempRate",     "raw"),
])

small_5_vars = small_4_vars + vars_block([
    ("INDPRO",       "IPgrowth",      "4log"),
])

medium_vars = vars_block([
    ("GDPC1",        "RGDP",          "4log"),
    ("GDPCTPI",      "PGDP",          "4log"),
    ("FEDFUNDS",     "FedFunds",      "raw"),
    ("PCECC96",      "Cons",          "4log"),
    ("GPDIC1",       "Inv",           "4log"),
    ("HOANBS",       "EmpHours",      "4log"),
    ("COMPRNFB",     "RealCompHour",  "4log"),
])

large_vars = vars_block([
    ("GDPC1",        "RGDP",          "4log"),
    ("GDPCTPI",      "PGDP",          "4log"),
    ("FEDFUNDS",     "FedFunds",      "raw"),
    ("CPIAUCSL",     "CPIALL",        "4log"),
    ("OILPRICEx",    "ComSpotPrice",  "4log"),
    ("INDPRO",       "IPtotal",       "4log"),
    ("PAYEMS",       "Emptotal",      "4log"),
    ("SRVPRD",       "EmpServices",   "4log"),
    ("PCECC96",      "Cons",          "4log"),
    ("GPDIC1",       "Inv",           "4log"),
    ("PRFIx",        "ResInv",        "4log"),
    ("PNFIx",        "NonResInv",     "4log"),
    ("PCECTPI",      "PCED",          "4log"),
    ("GPDICTPI",     "PGPDI",         "4log"),
    ("TCU",          "CapacityUtil",  "raw"),
    ("UMCSENTx",     "ConsExpect",    "raw"),
    ("HOANBS",       "EmpHours",      "4log"),
    ("COMPRNFB",     "RealCompHour",  "4log"),
    ("GS1",          "GS1",           "raw"),
    ("GS10",         "GS10",          "raw"),
    ("S&P 500",      "SP500",         "4log"),
    ("TWEXAFEGSMTHx","ExRate",        "4log"),
])

datasets = [small_3_vars, small_4_vars, small_5_vars, medium_vars, large_vars]
dsnames = ["Small_3", "Small_4", "Small_5", "Medium", "Large"]

Y_list = []
Ynames_list = []

for d, (vars_, name) in enumerate(zip(datasets, dsnames)):
    nv = len(vars_)
    raw = np.full((len(raw_values), nv), np.nan)
    for j, (mn, _, _) in enumerate(vars_):
        if mn not in series:
            raise KeyError(f"Series {mn!r} not found in FRED-QD.")
        raw[:, j] = raw_values[:, series.index(mn)]

    transformed = np.full_like(raw, np.nan)
    for j, (_, _, tcode) in enumerate(vars_):
        if tcode == "4log":
            transformed[1:, j] = 400.0 * (np.log(raw[1:, j]) - np.log(raw[:-1, j]))
        else:
            transformed[:, j] = raw[:, j]

    transformed = transformed[1:, :]  # drop first row (lost to differencing)
    Y_list.append(transformed)
    Ynames_list.append([lbl for _, lbl, _ in vars_])

    print(
        f"{name:6s} BVAR: {nv:2d} variables, {transformed.shape[0]} quarters "
        f"({raw_dates.iloc[1].date()} to {raw_dates.iloc[-1].date()})"
    )

# Ydates as ISO strings (MATLAB datetime(...) can re-parse easily)
dates_str = np.array(
    [d.strftime("%Y-%m-%d") for d in raw_dates.iloc[1:]],
    dtype=object,
).reshape(-1, 1)

out_files = [
    "fredqd_small_3.mat", "fredqd_small_4.mat", "fredqd_small_5.mat",
    "fredqd_medium.mat", "fredqd_large.mat",
]
sizes = [3, 4, 5, 7, 22]
for d, (fname, sz) in enumerate(zip(out_files, sizes)):
    names_cell = np.empty((1, len(Ynames_list[d])), dtype=object)
    for j, lbl in enumerate(Ynames_list[d]):
        names_cell[0, j] = lbl
    sio.savemat(
        HERE / fname,
        {"data": Y_list[d], "names": names_cell, "dates_": dates_str},
    )
    print(f"Saved {fname:22s}  {Y_list[d].shape[0]} x {Y_list[d].shape[1]}")
