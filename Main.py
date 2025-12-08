# --- Import necessary libraries ---
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
F_SCORE_RANGES = [
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
    (50, 150),   # Additional range for comparison
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
    
    print(f"✓ Selected {len(selected_features)} features")
    print(f"Top 5 features by F-score:")
    for feat, score in sorted(zip(selected_features, selected_scores), key=lambda x: -x[1])[:5]:
        print(f"  - {feat}: {score:.2f}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # --- Model 1: Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    y_proba_log = log_reg.predict_proba(X_test)
    
    # --- Model 2: Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    
    # --- Ensemble Method 1: Hard Voting ---
    y_pred_vote = np.where((y_pred_log + y_pred_rf) >= 1, 1, 0)
    
    # --- Ensemble Method 2: Soft Voting (Average Probabilities) ---
    y_proba_avg = (y_proba_log + y_proba_rf) / 2
    y_pred_soft = np.argmax(y_proba_avg, axis=1)
    
    # --- Calculate Metrics ---
    def get_metrics(y_true, y_pred, model_name):
        return {
            'F-score Range': f"{fs_min}-{fs_max}",
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1-Score': f1_score(y_true, y_pred, average='weighted'),
            'Num Features': len(selected_features)
        }
    
    results.append(get_metrics(y_test, y_pred_log, 'Logistic Regression'))
    results.append(get_metrics(y_test, y_pred_rf, 'Random Forest'))
    results.append(get_metrics(y_test, y_pred_vote, 'Hard Voting (Ensemble)'))
    results.append(get_metrics(y_test, y_pred_soft, 'Soft Voting (Ensemble)'))
    
    print(f"\n Results Summary:")
    print(f"  Logistic Regression - Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
    print(f"  Random Forest       - Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  Hard Voting         - Accuracy: {accuracy_score(y_test, y_pred_vote):.4f}")
    print(f"  Soft Voting         - Accuracy: {accuracy_score(y_test, y_pred_soft):.4f}")

# --- Create Results DataFrame ---
results_df = pd.DataFrame(results)
print(f"\n\n{'='*80}")
print("COMPLETE RESULTS SUMMARY")
print(f"{'='*80}\n")
print(results_df.to_string(index=False))

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison Across F-score Ranges', fontsize=16, fontweight='bold')

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    # Pivot for easier plotting
    pivot_data = results_df.pivot(index='F-score Range', columns='Model', values=metric)
    
    # Plot grouped bar chart
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{metric} by F-score Range', fontsize=12, fontweight='bold')
    ax.set_xlabel('F-score Range', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print(f"\n Visualization saved as 'model_comparison_results.png'")
plt.show()

# --- Best Model Analysis ---
print(f"\n\n{'='*80}")
print("BEST PERFORMING MODELS")
print(f"{'='*80}\n")

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    best_idx = results_df[metric].idxmax()
    best_row = results_df.iloc[best_idx]
    print(f"Best {metric:12s}: {best_row['Model']:25s} | Range: {best_row['F-score Range']:8s} | Score: {best_row[metric]:.4f}")

# --- Ensemble vs Individual Comparison ---
print(f"\n\n{'='*80}")
print("ENSEMBLE IMPROVEMENT ANALYSIS")
print(f"{'='*80}\n")

for fs_range in results_df['F-score Range'].unique():
    range_data = results_df[results_df['F-score Range'] == fs_range]
    
    lr_acc = range_data[range_data['Model'] == 'Logistic Regression']['Accuracy'].values[0]
    rf_acc = range_data[range_data['Model'] == 'Random Forest']['Accuracy'].values[0]
    hard_acc = range_data[range_data['Model'] == 'Hard Voting (Ensemble)']['Accuracy'].values[0]
    soft_acc = range_data[range_data['Model'] == 'Soft Voting (Ensemble)']['Accuracy'].values[0]
    
    best_individual = max(lr_acc, rf_acc)
    best_ensemble = max(hard_acc, soft_acc)
    improvement = ((best_ensemble - best_individual) / best_individual) * 100
    
    print(f"Range {fs_range}:")
    print(f"  Best Individual Model: {best_individual:.4f}")
    print(f"  Best Ensemble Model:   {best_ensemble:.4f}")
    print(f"  Improvement:           {improvement:+.2f}%")
    print()

print(f"{'='*80}")
print("Experiment complete!")
print(f"{'='*80}")