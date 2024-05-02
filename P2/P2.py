import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display

df = pd.read_csv(r'P2_dataset.csv')
display(df)

#isnull
df.isnull()
data = pd.isnull(df['math score'])
display(data)

#notnull
df.notnull()
data = pd.notnull(df['math score'])
display(data)

# fill na
print("result of fill na: \n")
df.fillna(1)

# filling missing values with mean, median, std deviation of that column
df['math score'] = df['math score'].fillna(df['math score'].mean())

#replace
df.replace(to_replace=np.nan,value=-99)

#dropna
df.dropna()
df.dropna(axis = 1) # to Drop columns with at least 1 null value.
df.dropna(axis=0)  # drop rows with at least 1 null value in CSV file.


# -------------------- Detecting outlier using Boxplot ----------------

# Boxplot summarizes sample data using 25th, 50th, and 75th percentiles. 
# One can just get insights(quartiles, median, and outliers) into the 
# dataset by just looking at its boxplot

col=['math score','reading score','writing score','placement score']
df.boxplot(col)

print("results of np.where: ")
print(np.where(df['math score']>90))
print(np.where(df['reading score']<25))
print(np.where(df['writing score']<30))

# Detecting outlier using Scatterplot
fig, ax=plt.subplots(figsize=(18,10))
ax.scatter(df['placement score'],df['placement offer count'])
ax.set_xlabel('placement score')
ax.set_ylabel('placement offer count')
ax.set_title('scatter plot')
plt.show()

# Detecting outlier using Z-score
z = np.abs(stats.zscore(df['math score'])) 
print("z score: \n", z)
threshold = 0.18
sample_outliers = np.where(z<threshold)
print("sample outliers: ", sample_outliers)

# Histogram
df['math score'].plot(kind='hist')
df['log_math'] = np.log10(df['math score'])
df['log_math'].plot(kind='hist')

