import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing


df = pd.read_csv("P2_dataset.csv")

# ------------------ mean --------------------

mean = df.mean()
loc = df.loc[:,'math score'].mean()
ax = df.mean(axis=1)[0:4]

print("mean: \n",mean)
print("loc: ", loc)
print("ax: \n",ax)


# ----------------- median ------------

median = df.median()
median_loc = df.loc[:,'math score'].median()
median_ax = df.median(axis=1)[0:4]

print("median: ",median)
print("median_loc: ", median_loc)
print("median_Ax: ",median_ax)


# ------------------ mode --------------

mode = df.mode()
mode_loc = df.loc[:,'math score'].mode()

print("mode: ", mode)
print("mode_loc",mode_loc)

# ------------------ minimum --------------

minimum = df.min()
min_loc = df.loc[:,'math score'].min()

print("minimum: ", minimum)
print("minimum_loc: ",min_loc)


# ------------------ maximum --------------

maximum = df.max()
max_loc = df.loc[:,'math score'].min()

print("maximum: ", maximum)
print("maximum_loc: ",max_loc)


# ------------------ standard deviation --------------

std_daviation = df.std()
std_loc = df.loc[:,'math score'].std()

print("std deviation: ", std_daviation)
print("std dev loc: ",std_loc)


# ------------------ group by --------------

gp_mean = df.groupby(['math score'])['writing score'].mean()
enc = preprocessing.OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df[['math score']]).toarray())

df_encode = df.join(enc_df)

print("group by mean: \n", gp_mean)
print("enc df: \n", enc_df)
print("df_encode: \n",df_encode)
