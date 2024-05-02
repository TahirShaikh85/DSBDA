import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display

# Read the dataset
df = pd.read_csv('students.csv')

# Display the dataset
print("Original Dataset:")
display(df)

# Handle Missing Data
print("\nHandling Missing Data:")
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values in 'math score' with the mean
df['math_score'] = df['math_score'].fillna(df['math_score'].mean())

# Drop rows with missing values
df_cleaned = df.dropna()

# Display cleaned dataset
print("\nCleaned Dataset after handling missing data:")
display(df_cleaned)

# Detect Outliers
print("\nDetecting Outliers:")
# Boxplot
print("\nBoxplot:")
df_cleaned.boxplot(['math_score', 'reading_score', 'writing_score', 'placement_score'])
plt.show()

# Scatterplot
print("\nScatterplot:")
plt.scatter(df_cleaned['placement_score'], df_cleaned['placement_offer_count'])
plt.xlabel('Placement Score')
plt.ylabel('Placement Offer Count')
plt.title('Scatter Plot')
plt.show()

# Z-score
print("\nZ-score:")
z_scores = stats.zscore(df_cleaned['math_score'])
threshold = 3
outliers = np.where(np.abs(z_scores) > threshold)[0]
print("Indices of Outliers:", outliers)

# Visualize Data
print("\nVisualizing Data:")
# Histogram
print("\nHistogram of 'math_score':")
df_cleaned['math_score'].plot(kind='hist', bins=20)
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.title('Histogram of Math Score')
plt.show()
