# --- Import necessary libraries ---
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- Load Datasets ---
data_path = 'Dataset/dataset_phishing.csv'   # phishing dataset
df = pd.read_csv(data_path)

# --- Data Cleaning ---
df.columns = [col.strip() for col in df.columns]        # clean column names
df = df.replace([np.inf, -np.inf], np.nan)              # replace infinities
df = df.dropna(axis=0)                                  # drop missing rows

# Drop constant (non-informative) columns
for col in list(df.columns):
    if col.lower() != "label" and df[col].nunique() <= 1:
        df = df.drop(columns=[col])


# Try to automatically detect the label column
possible_labels = ["label", "Label", "LABEL", "Result", "result", "Status", "status", "class", "Class"]
label_col = None
for col in df.columns:
    if col.strip() in possible_labels:
        label_col = col
        break

if label_col is None:
    raise ValueError("Could not find a label column in your dataset. Please check the CSV header.")

print(f"Using '{label_col}' as the label column.")

# --- Separate Features and Label ---
feature_cols = [c for c in df.columns if c not in [label_col, "URL", "url"]]
X = df[feature_cols].select_dtypes(include=[np.number])  # numeric features only
y_raw = df[label_col]

# --- Encode the Labels ---
le = LabelEncoder()
y = le.fit_transform(y_raw)  # e.g., benign=0, phishing=1

# --- Feature Selection ---
selector = SelectKBest(score_func=f_classif, k=min(30, X.shape[1]))  # top 10 features
X_new = selector.fit_transform(X, y)
selected_features = np.array(feature_cols)[selector.get_support()]

print("Top Selected Features:")
for feature in selected_features:
    print("-", feature)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.3, random_state=42, stratify=y
)

# --- Model 1: Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred_log, target_names=le.classes_))

# --- Model 2: Random Forest ---
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))