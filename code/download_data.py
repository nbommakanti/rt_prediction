import pandas as pd
df = pd.read_csv('https://stats.idre.ucla.edu/stat/data/binary.csv')
df.to_csv('../data/raw.csv', index=False)