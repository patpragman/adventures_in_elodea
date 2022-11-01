import pandas as pd

df = pd.read_csv('test.csv')
df = df.sort_values(by=['label'])
df.to_csv('test.csv', index=False)