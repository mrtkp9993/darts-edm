import pandas as pd

from src import *

# Load your dataframe as df
df = pd.read_csv("exampledata.csv", index_col=0)
df.fillna(method="bfill", inplace=True)

modelsimpl = SimplexPredictor()
modelsimpl.fit(df)

fh = 12
preds = modelsimpl.predict(fh)  # fh is forecast horizon
print(preds)
