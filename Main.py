# Import necessary libraries for data manipulation and machine learning preparation.
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif #Feature selection tool, super useful.
from sklearn.model_selection import train_test_split         #The next two tools are used for training.
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
# Import Linear models
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
# --- Load the Dataset ---
file_path = 'phishing_data.csv'  # Add path to your dataset file here
df = pd.read_csv(file_path)




# --- Data Cleaning Process ---
# 1. Clean up column names by removing any extra whitespace.
df.columns = [col.strip() for col in df.columns]
# 2. Replace infinite values with NaN (Not a Number), which is a standard marker for missing data.
df = df.replace ([np.inf, -np.inf], np.nan)
# 3. Drop any rows that contain at least one NaN value.
df = df.dropna(axis = 0)
# 4. Drop any columns that have only one unique value (constant features), as they provide no information for the model.
for col in list(df.columns):
  if col != "Label":
    if df[col].nunique() <= 1:
      df = df.drop(columns = [col])
# --- Isolate Data for Analysis ---


# 1. Select only the numerical features from our dataframe.
# We are excluding 'Label' because that is the target.
feature_cols = [c for c in df.columns if c != "Label"]
X = df[feature_cols].select_dtypes(include=[np.number])
# 2. Select the 'Label' column as our target.
y_raw = df["Label"]
# 3. Machine learning models require numerical inputs, so we use LabelEncoder to convert
# text labels ('Benign', 'FTP-BruteForce', etc.) into numbers (0, 1, 2).
le = LabelEncoder()
y = le.fit_transform(y_raw)

# --- Run SelectKBest ---
# SelectKBest automatically selects the best features based on statistical tests.
selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features
X_new = selector.fit_transform(X, y)
selected_features = np.array(feature_cols)[selector.get_support()]

# --- Display the Results ---
# Show the top features that were selected by SelectKBest.
print("Top Selected Features:")
for feature in selected_features:
    print("-", feature)


# Split the data into two parts: a training set and a testing set.
# The model will learn from the training set and its performance will be evaluated on the unseen testing set.
# test_size=0.3 means 30% of the data is reserved for testing.
# random_state=42 ensures that the split is the same every time we run the code, for reproducibility.
# stratify=y ensures that the proportion of each class (Benign, FTP, SSH) is the same in both the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42, stratify=y)

# --- Train and Evaluate the Model ---
# Here, we train a simple model (Logistic Regression) to get initial results for evaluation.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print classification results including Precision, Recall, F1-Score, and Accuracy.
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
