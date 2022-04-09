import pandas as pd
import statsmodels.api as sm
import patsy

# Load
df = pd.read_csv('../data/raw.csv')

# Fit model
y, X = patsy.dmatrices('admit ~ gre + gpa + C(rank)', df, return_type='dataframe')
logit = sm.Logit(y, X).fit()
logit.save('../data/logit.pkl')