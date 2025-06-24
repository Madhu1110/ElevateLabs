import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

ds = pd.read_csv("C:/Users/madhu/Downloads/Titanic-Dataset.csv")

# summary 
print("summary statistics:\n",ds.describe(include = 'all'))

# histograms
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
ds[numeric_features].hist(bins=20, figsize=(12, 6), color='skyblue')
plt.tight_layout()
plt.show()

# boxplots
for col in numeric_features:
    sns.boxplot(x=ds[col], color='lightcoral')
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# corelation matrix
# convert categorical data into numerical data 
ds_encoded = ds.copy()
label_encoder = LabelEncoder()
ds_encoded['Sex'] = label_encoder.fit_transform(ds['Sex'])
ds_encoded['Embarked'] = label_encoder.fit_transform(ds['Embarked'])

#pairplot for visualization 
sns.pairplot(ds[['Survived', 'Pclass', 'Age', 'Sex', 'Fare']], hue='Survived')
plt.show()

# Identifying the trends in the data

sns.barplot(x='Sex', y='Survived', data=ds)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Probability")
plt.show()
print(" Survival Rate by Age:",ds.groupby('Sex')['Survived'].mean())


sns.barplot(x='Pclass', y='Survived', data=ds)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Probability")
plt.show()
print(" Survival Rate by Age:",ds.groupby('Pclass')['Survived'].mean())

sns.barplot(x='Age', y='Survived', data=ds)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Probability")
plt.show()
print(" Survival Rate by Age:",ds.groupby('Age')['Survived'].mean())

# feature level inferences 
print("females had higher survival rate")
print("passengers in pclasss1 have higher chance of survival")
