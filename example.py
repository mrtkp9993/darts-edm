import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.regression_model import RegressionModel

from .src import *

# Load your dataframe as df
df = TimeSeries.from_dataframe(df)

modelsimpl = SimplexPredictor()
modelsimpl.fit(df)
preds = modelsimpl.predict(fh) # fh is forecast horizon
