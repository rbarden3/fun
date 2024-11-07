# %%
import json
from pathlib import Path

import pandas as pd

# %%
DROP_NA_COLS = True
MIN_VALUE_COUNT = 3
OUTPUT_NAME = "result" + ".csv"

# %%
load_path = Path(__file__).parent / "load.json"

data = json.load(open(load_path, "r"))

data = [x["result"] for x in data]

df = pd.DataFrame(data)

df.head()

# %%
if DROP_NA_COLS:
    df.dropna(axis=1, how="all", inplace=True)

if MIN_VALUE_COUNT > 0:
    for col in df.columns:  # Loop through columns
        try:
            df[col].unique()
        except:
            df[col] = df[col].apply(json.dumps)

    nunique = df.nunique()
    cols_to_drop = nunique[nunique < MIN_VALUE_COUNT].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    df.head()

# %%
df.to_csv(OUTPUT_NAME, index=False)

# %%
