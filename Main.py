# Import necessary libraries for data manipulation and machine learning preparation.
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif #Feature selection tool, super useful.
from sklearn.model_selection import train_test_split         #The next two tools are used for training.
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# --- Load the Dataset ---
file_path = 'Add path here'




# --- Data Cleaning Process ---

# 1. Clean up column names by removing any extra whitespace.


# 2. Replace infinite values with NaN (Not a Number), which is a standard marker for missing data.


# 3. Drop any rows that contain at least one NaN value.


# 4. Drop any columns that have only one unique value (constant features), as they provide no information for the model.




# --- Isolate Data for Analysis ---


# 1. Select only the numerical features from our dataframe.


# 2. Select the 'Label' column as our target.


# 3. Machine learning models require numerical inputs, so we use LabelEncoder to convert
# text labels ('Benign', 'FTP-BruteForce', etc.) into numbers (0, 1, 2).


# --- Run SelectKBest ---


# --- Display the Results ---


# Split the data into two parts: a training set and a testing set.
# The model will learn from the training set and its performance will be evaluated on the unseen testing set.
# test_size=0.3 means 30% of the data is reserved for testing.
# random_state=42 ensures that the split is the same every time we run the code, for reproducibility.
# stratify=y ensures that the proportion of each class (Benign, FTP, SSH) is the same in both the train and test sets.

