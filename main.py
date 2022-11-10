import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
from math import log

data = pd.read_csv("speed_price_att.csv")

data = {
    "speed": np.array(data["speed_down"]),
    "race": np.array(data["race_perc_non_white"]),
    "ppl": np.array(data["ppl_per_sq_mile"]),
    "providers": np.array(data["n_providers"]),
    "broadband": np.array(data["internet_perc_broadband"]),
    "income": np.array(data["median_household_income"])
}

df = pd.DataFrame(data)
x = df[["ppl", "broadband", "speed", "providers"]]
y = [log(inc) for inc in data["income"]]

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x)  # adding a constant

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)