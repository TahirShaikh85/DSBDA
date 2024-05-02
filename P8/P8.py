import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
dataset = sns.load_dataset('titanic')

# Print and display the dataset
print("dataset -> \n ", dataset)
print("head() --> \n", dataset.head())


# ----- Univariate Distribution Plots ----

# Plot the distribution of age
sns.distplot(x=dataset['age'], bins=10)
plt.title('Distribution of Age')
plt.show()

# Plot the distribution of age without kernel density estimation
sns.distplot(dataset['age'], bins=10, kde=False)
plt.title('Distribution of Age (Without KDE)')
plt.show()


# ---------  Bivariate Distribution Plots --------- 

# Plot a scatter plot of age versus fare
sns.jointplot(x=dataset['age'], y=dataset['fare'], kind='scatter')
plt.title('Scatter Plot of Age vs Fare')
plt.show()

# Plot a hexbin plot of age versus fare
sns.jointplot(x=dataset['age'], y=dataset['fare'], kind='hex')
plt.title('Hexbin Plot of Age vs Fare')
plt.show()


# ---------  Categorical Plots --------- 

# Plot a rug plot of fare
sns.rugplot(dataset['fare'])
plt.title('Rug Plot of Fare')
plt.show()

# Plot a bar plot of average age by sex
sns.barplot(x='sex', y='age', data=dataset)
plt.title('Average Age by Sex')
plt.show()

# Plot a bar plot of standard deviation of age by sex
sns.barplot(x='sex', y='age', data=dataset, estimator=np.std)
plt.title('Standard Deviation of Age by Sex')
plt.show()

# Plot a count plot of sex
sns.countplot(x='sex', data=dataset)
plt.title('Count of Passengers by Sex')
plt.show()

# Plot a box plot of age by sex
sns.boxplot(x='sex', y='age', data=dataset)
plt.title('Box Plot of Age by Sex')
plt.show()

# Plot a box plot of age by sex and survival
sns.boxplot(x='sex', y='age', data=dataset, hue='survived')
plt.title('Box Plot of Age by Sex and Survival')
plt.show()

# Plot a violin plot of age by sex
sns.violinplot(x='sex', y='age', data=dataset)
plt.title('Violin Plot of Age by Sex')
plt.show()

# Plot a violin plot of age by sex and survival
sns.violinplot(x='sex', y='age', data=dataset, hue='survived')
plt.title('Violin Plot of Age by Sex and Survival')
plt.show()

# Plot a strip plot of age by sex and survival
sns.stripplot(x='sex', y='age', data=dataset, jitter=True, hue='survived')
plt.title('Strip Plot of Age by Sex and Survival')
plt.show()

# Plot a swarm plot of age by sex
sns.swarmplot(x='sex', y='age', data=dataset)
plt.title('Swarm Plot of Age by Sex')
plt.show()

# Plot a swarm plot of age by sex and survival
sns.swarmplot(x='sex', y='age', data=dataset, hue='survived')
plt.title('Swarm Plot of Age by Sex and Survival')
plt.show()
