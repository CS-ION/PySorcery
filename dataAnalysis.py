# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 00:36:33 2025

@author: romai
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# Load dataset
df = pd.read_csv("train.csv")
print(df.head())

# df_test=pd.read_csv("test.csv")
# print(df_test.head())


# Inspect dataset
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.describe())   # summary stats

print("")
print("")
print("######################################################")
print("######################################################")
print("######################################################")
print("######################################################")
print("######################################################")
print("")
print("")

# Select only numeric columns from the dataset for simplicity
df_num = df.select_dtypes(include=[np.number]).copy()
print('Numeric features shape before dropna:', df_num.shape)

# Drop rows with missing values (a simple preprocessing approach suitable for labs or demos)
df_num = df_num.dropna(axis=0)
print('After dropna:', df_num.shape)

# Define the target column (what we want to predict)
target_col = 'label'

# Sanity check to ensure the target column exists in the numeric dataframe
if target_col not in df_num.columns:
    raise ValueError('Expected label column in numeric dataframe.')


# Split the data into features (X) and target (y)
X = df_num.drop(columns=[target_col])  # Features: all numeric columns except the target
y = df_num[target_col]                 # Target: label


# Display summary statistics for the first 12 numeric features
print("\n\n      description of X :\n")
print((X.describe().T.head(12)))



print("")
print("")
print("######################################################")
print("######################################################")
print("######################################################")
print("######################################################")
print("######################################################")
print("")
print("")

# tracé de la boite a moustache

feature = "feature_100"
classes = [46, 21, 29]

df_subset = df_num[df_num["label"].isin(classes)]

plt.figure(figsize=(10, 6))
sns.boxplot(x="label", y=feature, data=df_subset)
plt.title(f"Vertical 'boite à moustache' of the feature {feature} for the classes {classes}")
plt.show()


print("")
print("")
print("######################################################")
print("######################################################")
print("######################################################")
print("######################################################")
print("######################################################")
print("")
print("")

# calcul des tailles des classes
# Compter le nombre d'échantillons par classe
class_counts = df_num["label"].value_counts()

# Calculer la taille moyenne des classes
mean_class_size = class_counts.mean()

# Afficher la taille de la classe 46 et la moyenne
print("Nombre d'échantillons par classe :")
print(class_counts)
print(f"\nTaille moyenne des classes : {mean_class_size:.2f}")
print(f"Taille de la classe 21 : {class_counts[21]}")
print(f"\nTaille moyenne des classes : {mean_class_size:.2f}")
print(f"Taille de la classe 29 : {class_counts[29]}")
print(f"\nTaille moyenne des classes : {mean_class_size:.2f}")
print(f"Taille de la classe 46 : {class_counts[46]}")
