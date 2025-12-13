# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 20:05:44 2025

@author: romai
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report


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


#####################################################################
#####################################################################
######      selection of features using correlation matrix      #####
#####################################################################
#####################################################################


# Compute absolute correlation between each feature and the target (label)
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)

# Select the top 10 features that have the highest absolute correlation with label
top_features = corr_with_target.head(10).index.tolist()

# Print the top 10 features and their correlation values
print('Top features by absolute correlation with label:')
print(corr_with_target.head(20))

# Prepare a list of columns to include in the heatmap (top features + target)
cols = top_features + [target_col]

# Plot a heatmap of the correlation matrix for the selected columns
sns.heatmap(df_num[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation matrix: top features vs label')
plt.show()


# Nombre de valeurs uniques dans la colonne 'label'
n_unique_labels = df['label'].nunique()
print(f"Nombre d'outputs différents dans 'label' : {n_unique_labels}")

# Afficher les valeurs uniques (optionnel)
unique_labels = df['label'].unique()
print(f"Valeurs uniques dans 'label' : {unique_labels}")

# Afficher les valeurs uniques (optionnel)
sorted_unique_labels = np.sort(unique_labels) 
print(f"Valeurs uniques triées dans 'label' : {sorted_unique_labels}")


# Select top K features for baseline (K)
K = 7 
selected_features = corr_with_target.head(K).index.tolist()
print('Selected features:', selected_features)

X_sel = X[selected_features].copy()


# #####################################################################
# #####################################################################
# #######################      decision tree      #####################
# #####################################################################
# #####################################################################


# Function to calculate Gini impurity
def gini(y):
    classes = np.unique(y)  # Get unique class labels
    impurity = 1.0
    for c in classes:
        p = np.sum(y == c) / len(y)  # Probability of class c
        impurity -= p ** 2           # Subtract squared probability
    return impurity

# Function to calculate entropy
def entropy(y):
    classes = np.unique(y)  # Get unique class labels
    ent = 0.0
    for c in classes:
        p = np.sum(y == c) / len(y)  # Probability of class c
        ent -= p * np.log2(p + 1e-9)  # Entropy formula with small epsilon to avoid log(0)
    return ent

# Function to split the dataset into left and right branches based on feature threshold
def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] < threshold   # Mask for left split
    right_mask = X[:, feature_index] >= threshold # Mask for right split
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# Class representing a single node in the decision tree
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Threshold value to split at
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.value = value                  # Class prediction if it's a leaf node

# Class implementing the decision tree classifier from scratch
class DecisionTreeScratchClassifier:
    def __init__(self, max_depth=5, criterion='gini'):
        self.max_depth = max_depth                          # Maximum depth of the tree
        self.criterion = gini if criterion == 'gini' else entropy  # Split criterion
        self.root = None                                    # Root node of the tree

    # Recursive function to build the tree
    def fit(self, X, y, depth=0):
        # If all samples have the same class or max depth reached, create a leaf node
        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return DecisionNode(value=np.bincount(y).argmax())  # Majority class
        print("\nentering a new layer....depth =",depth)
        best_gain = 0
        best_criteria = None
        best_sets = None
        current_impurity = self.criterion(y)  # Calculate impurity before the split

        # Try all features and all thresholds to find the best split
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])  # Unique values for the feature
            for threshold in thresholds:
                # Split the dataset
                X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)

                # Skip invalid splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate information gain
                p = len(y_left) / len(y)
                gain = current_impurity - (
                    p * self.criterion(y_left) + (1 - p) * self.criterion(y_right)
                )

                # Update the best split if gain is improved
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (feature_index, threshold)
                    best_sets = (X_left, y_left, X_right, y_right)

        # If no gain, return a leaf node
        if best_gain == 0:
            return DecisionNode(value=np.bincount(y).argmax())

        # Recursively build the left and right branches
        left = self.fit(best_sets[0], best_sets[1], depth + 1)
        right = self.fit(best_sets[2], best_sets[3], depth + 1)

        # Create a decision node with the best split
        self.root = DecisionNode(best_criteria[0], best_criteria[1], left, right)
        return self.root

    # Function to predict a single sample
    def predict_one(self, x, node):
        if node.value is not None:  # If it's a leaf node
            return node.value
        # Recursively traverse the left or right child based on the threshold
        if x[node.feature_index] < node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    # Function to predict a batch of samples
    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])

# Train Scratch Decision Tree using entropy as the criterion
tree_scratch = DecisionTreeScratchClassifier(max_depth=10, criterion='entropy')
tree_scratch.fit(np.array(X_sel), np.array(y))

# Make predictions
y_pred_scratch = tree_scratch.predict(np.array(X_sel))

# Evaluate model performance
print("Scratch Decision Tree Accuracy:", accuracy_score(y, y_pred_scratch))
print(classification_report(y, y_pred_scratch))
 

