import seaborn as sns

# Load the iris dataset
dataset = sns.load_dataset('iris')

# Display the first few rows of the dataset
dataset.head()

import matplotlib.pyplot as plt

# Create subplots for histograms of sepal and petal measurements
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
sns.histplot(dataset['sepal_length'], ax=axes[0, 0])
sns.histplot(dataset['sepal_width'], ax=axes[0, 1])
sns.histplot(dataset['petal_length'], ax=axes[1, 0])
sns.histplot(dataset['petal_width'], ax=axes[1, 1])

# Create box plots of measurements grouped by species
sns.boxplot(y='petal_length', x='species', data=dataset)
plt.title('Boxplot of Petal Length by Species')
plt.show()

sns.boxplot(y='petal_width', x='species', data=dataset)
plt.title('Boxplot of Petal Width by Species')
plt.show()

sns.boxplot(y='sepal_length', x='species', data=dataset)
plt.title('Boxplot of Sepal Length by Species')
plt.show()

sns.boxplot(y='sepal_width', x='species', data=dataset)
plt.title('Boxplot of Sepal Width by Species')
plt.show()
