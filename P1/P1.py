import pandas as pd
df1 = pd.read_csv(r"n_movies.csv")
print(df1.head())
print(df1.tail(n=10))
# print(df1.index)
# print(df1.dtypes)
# print(df1.columns.values)
# print(df1.iloc[5])

print("shape: \n", df1.shape)