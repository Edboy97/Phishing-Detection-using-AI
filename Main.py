# --- Import necessary libraries ---
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
F_SCORE_RANGES = [
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500)
]

# --- Load Datasets ---
data_path = 'Dataset/dataset_phishing.csv'
df = pd.read_csv(data_path)

# --- Data Cleaning ---
df.columns = [col.strip() for col in df.columns]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(axis=0)

# Drop constant columns
for col in list(df.columns):
    if col.lower() != "label" and df[col].nunique() <= 1:
        df = df.drop(columns=[col])

# Detect label column
possible_labels = ["label", "Label", "LABEL", "Result", "result", "Status", "status", "class", "Class"]
label_col = None
for col in df.columns:
    if col.strip() in possible_labels:
        label_col = col
        break

if label_col is None:
    raise ValueError("Could not find a label column in your dataset.")

print(f"Using '{label_col}' as the label column.\n")

# --- Separate Features and Label ---
feature_cols = [c for c in df.columns if c not in [label_col, "URL", "url"]]
X = df[feature_cols].select_dtypes(include=[np.number])
numeric_feature_cols = X.columns.tolist()
y_raw = df[label_col]

# --- Encode Labels ---
le = LabelEncoder()
y = le.fit_transform(y_raw)

# --- Compute F-scores once ---
f_scores, p_values = f_classif(X, y)

# --- Storage for results ---
results = []

# --- Experiment Loop ---
for fs_min, fs_max in F_SCORE_RANGES:
    print(f"\n{'='*60}")
    print(f"Testing F-score range: [{fs_min}, {fs_max}]")
    print(f"{'='*60}")
    
    # Select features in this range
    mask = (f_scores >= fs_min) & (f_scores <= fs_max)
    
    if mask.sum() == 0:
        print(f"  No features in range [{fs_min}, {fs_max}]. Skipping...")
        continue
    
    selected_features = np.array(numeric_feature_cols)[mask]
    selected_scores = f_scores[mask]
    X_selected = X[selected_features].values
    
    print(f" Selected {len(selected_features)} features")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
        # 3,429 URLs are used to train
    )
    
    # --- Model 1: Logistic Regression (LR) ---
    log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    y_proba_log = log_reg.predict_proba(X_test)
    
    # --- Model 2: Random Forest (RF) ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)

    # --- Model 3: Decision Tree (DT) ---
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)
    
    # --- ENSEMBLES ---
    
    # 1. LR + RF
    y_pred_vote_lr_rf = np.where((y_pred_log + y_pred_rf) >= 1, 1, 0)
    y_proba_avg_lr_rf = (y_proba_log + y_proba_rf) / 2
    y_pred_soft_lr_rf = np.argmax(y_proba_avg_lr_rf, axis=1)

    # 2. DT + RF
    y_pred_vote_dt_rf = np.where((y_pred_dt + y_pred_rf) >= 1, 1, 0)
    y_proba_avg_dt_rf = (y_proba_dt + y_proba_rf) / 2
    y_pred_soft_dt_rf = np.argmax(y_proba_avg_dt_rf, axis=1)

    # 3. DT + LR
    y_pred_vote_dt_lr = np.where((y_pred_dt + y_pred_log) >= 1, 1, 0)
    y_proba_avg_dt_lr = (y_proba_dt + y_proba_log) / 2
    y_pred_soft_dt_lr = np.argmax(y_proba_avg_dt_lr, axis=1)
    
    # --- Helper to calculate and store metrics ---
    def add_result(y_true, y_pred, model_name):
        results.append({
            'F-score Range': f"{fs_min}-{fs_max}",
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'Num Features': len(selected_features)
        })
    
    # Store Base Models
    add_result(y_test, y_pred_log, 'Logistic Regression')
    add_result(y_test, y_pred_rf, 'Random Forest')
    add_result(y_test, y_pred_dt, 'Decision Tree')

    # Store Ensembles
    add_result(y_test, y_pred_vote_lr_rf, 'Ensemble: LR + RF (Hard)')
    add_result(y_test, y_pred_soft_lr_rf, 'Ensemble: LR + RF (Soft)')
    
    add_result(y_test, y_pred_vote_dt_rf, 'Ensemble: DT + RF (Hard)')
    add_result(y_test, y_pred_soft_dt_rf, 'Ensemble: DT + RF (Soft)')

    add_result(y_test, y_pred_vote_dt_lr, 'Ensemble: DT + LR (Hard)')
    add_result(y_test, y_pred_soft_dt_lr, 'Ensemble: DT + LR (Soft)')
    
    print(f"  > Processed results for range [{fs_min}, {fs_max}]")

# --- Create Results DataFrame ---
results_df = pd.DataFrame(results)
print(f"\n\n{'='*80}")
print("COMPLETE RESULTS SUMMARY")
print(f"{'='*80}\n")
print(results_df.head(10).to_string(index=False))

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Model Performance Comparison (Base Models vs Ensembles)', fontsize=16, fontweight='bold')

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    pivot_data = results_df.pivot(index='F-score Range', columns='Model', values=metric)
    pivot_data.plot(kind='bar', ax=ax, width=0.85)
    ax.set_title(f'{metric} by F-score Range', fontsize=12, fontweight='bold')
    ax.set_xlabel('F-score Range', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('model_comparison_expanded.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Best Model Analysis ---
print(f"\n\n{'='*80}")
print("BEST PERFORMING MODELS OVERALL")
print(f"{'='*80}\n")

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    best_idx = results_df[metric].idxmax()
    best_row = results_df.iloc[best_idx]
    print(f"Best {metric:12s}: {best_row['Model']:30s} | Range: {best_row['F-score Range']:8s} | Score: {best_row[metric]:.4f}")

# --- Ensemble Improvement Analysis ---
print(f"\n\n{'='*80}")
print("DETAILED ENSEMBLE IMPROVEMENT ANALYSIS (By Pair)")
print(f"{'='*80}\n")

# Define the pairs we want to analyze
# Format: (Model 1 Name, Model 2 Name, Ensemble Key Identifier)
pairs = [
    ('Logistic Regression', 'Random Forest', 'LR + RF'),
    ('Decision Tree', 'Random Forest', 'DT + RF'),
    ('Decision Tree', 'Logistic Regression', 'DT + LR')
]

for fs_range in results_df['F-score Range'].unique():
    print(f"--- F-score Range: {fs_range} ---")
    range_data = results_df[results_df['F-score Range'] == fs_range]
    
    for m1_name, m2_name, ens_key in pairs:
        # Get scores for individual models
        score_m1 = range_data.loc[range_data['Model'] == m1_name, 'Accuracy'].values[0]
        score_m2 = range_data.loc[range_data['Model'] == m2_name, 'Accuracy'].values[0]
        
        # Determine the better baseline (the one to beat)
        best_individual_score = max(score_m1, score_m2)
        best_individual_name = m1_name if score_m1 >= score_m2 else m2_name
        
        # Get scores for the corresponding ensembles (Hard and Soft voting)
        # Note: We look for models that contain the 'ens_key' (e.g., "LR + RF")
        ens_hard_name = f'Ensemble: {ens_key} (Hard)'
        ens_soft_name = f'Ensemble: {ens_key} (Soft)'
        
        score_hard = range_data.loc[range_data['Model'] == ens_hard_name, 'Accuracy'].values[0]
        score_soft = range_data.loc[range_data['Model'] == ens_soft_name, 'Accuracy'].values[0]
        
        # Calculate improvements
        imp_hard = ((score_hard - best_individual_score) / best_individual_score) * 100
        imp_soft = ((score_soft - best_individual_score) / best_individual_score) * 100
        
        print(f"  Combination: {m1_name} & {m2_name}")
        print(f"    Best Individual: {best_individual_name} ({best_individual_score:.4f})")
        print(f"    Ensemble (Hard): {score_hard:.4f} (Improvement: {imp_hard:+.2f}%)")
        print(f"    Ensemble (Soft): {score_soft:.4f} (Improvement: {imp_soft:+.2f}%)")
        print("")
    print("-" * 60)

print(f"{'='*80}")
print("Experiment complete!")
print(f"{'='*80}")