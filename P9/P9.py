import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
dataset = sns.load_dataset('titanic')

# Display the first few rows of the dataset
dataset.head()

# Create a box plot of age distribution grouped by gender
sns.boxplot(x='sex', y='age', data=dataset)
plt.title('Age Distribution by Gender')
plt.show()

# Create a box plot of age distribution grouped by gender and survival status
sns.boxplot(x='sex', y='age', data=dataset, hue='survived')
plt.title('Age Distribution by Gender and Survival')
plt.show()
