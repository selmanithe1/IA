import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_data(data_dir="data"):
    print("Loading data...")
    
    # 1. Load X_train
    x_train_path = os.path.join(data_dir, "train", "X.csv.gz")
    if not os.path.exists(x_train_path):
        print(f"Error: {x_train_path} not found.")
        return None, None
    
    X_train = pd.read_csv(x_train_path)
    
    # Scale features (as per notebook)
    # The last 2 columns are 'subject' and potentially 'L_path' if we added it, but here raw X has 'subject' at end.
    # We should exclude 'subject' from scaling.
    feature_cols = [c for c in X_train.columns if c not in ['subject', 'L_path']]
    X_train[feature_cols] *= 1e12
    
    print(f"X_train shape: {X_train.shape}")
    
    # 2. Load Lead Fields
    lead_pattern = os.path.join(data_dir, "*L.npz")
    lead_field_files = sorted(glob.glob(lead_pattern))
    
    lead_subject = {}
    for subject in np.unique(X_train["subject"]):
        matches = [f for f in lead_field_files if f"{subject}_L" in f]
        if matches:
            lead_subject[subject] = matches[0]
        else:
            print(f"Warning: Lead field for subject {subject} not found.")

    X_train["L_path"] = X_train["subject"].map(lead_subject)
    
    # 3. Load y_train
    y_train_path = os.path.join(data_dir, "train", "target.npz")
    y_train = sparse.load_npz(y_train_path).toarray()
    print(f"y_train shape: {y_train.shape}")
    
    return X_train, y_train

def explore_data(X, y):
    print("\n--- Data Exploration ---")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1] - 2}") # Subtract subject and L_path
    print(f"Number of targets (brain parcels): {y.shape[1]}")
    
    n_sources = np.sum(y, axis=1)
    print(f"Average sources per sample: {np.mean(n_sources):.2f}")
    print(f"Max sources per sample: {np.max(n_sources)}")
    
    plt.figure(figsize=(8, 5))
    plt.hist(n_sources, bins=np.arange(0, 10)-0.5, rwidth=0.8)
    plt.title("Distribution of Number of Active Sources")
    plt.xlabel("Number of Sources")
    plt.ylabel("Count")
    plt.xticks(range(10))
    # plt.show() # Uncomment to show plot
    print("Exploration finished.")

def run_knn(X_train, X_val, y_train, y_val):
    print("\n--- Running KNN ---")
    # For simplicity, we drop metadata columns for the model input
    feature_cols = [c for c in X_train.columns if c.startswith('e')]
    
    # KNN
    k = 5
    print(f"Training KNN with k={k}...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[feature_cols], y_train)
    
    y_pred = knn.predict(X_val[feature_cols])
    
    acc = accuracy_score(y_val, y_pred)
    hl = hamming_loss(y_val, y_pred)
    
    print(f"KNN Results:")
    print(f"  Accuracy (Subset): {acc:.4f}")
    print(f"  Hamming Loss: {hl:.4f}")
    return acc

def run_lasso(X_train, X_val, y_train, y_val):
    print("\n--- Running Lasso (Regression) ---")
    # Lasso requires numeric features. Removing subject/L_path if present in passed df (though we handle cols above)
    feature_cols = [c for c in X_train.columns if c.startswith('e')]
    
    print("Training Lasso...")
    # increasing max_iter for convergence
    lasso = Lasso(alpha=0.2, max_iter=2000) 
    lasso.fit(X_train[feature_cols], y_train)
    
    y_pred_continuous = lasso.predict(X_val[feature_cols])
    
    # Thresholding
    y_pred = (y_pred_continuous > 0.1).astype(int) 
    
    acc = accuracy_score(y_val, y_pred)
    hl = hamming_loss(y_val, y_pred)
    
    print(f"Lasso Results (Threshold=0.1):")
    print(f"  Accuracy (Subset): {acc:.4f}")
    print(f"  Hamming Loss: {hl:.4f}")
    return acc

def run_lassolars(X_train, X_val, y_train, y_val):
    print("\n--- Running LassoLars ---")
    from sklearn.linear_model import LassoLars
    feature_cols = [c for c in X_train.columns if c.startswith('e')]
    
    # LassoLars
    print("Training LassoLars...")
    # alpha roughly corresponds to Lasso alpha but scale might differ
    ll = LassoLars(alpha=0.01, max_iter=500) 
    ll.fit(X_train[feature_cols], y_train)
    
    y_pred_continuous = ll.predict(X_val[feature_cols])
    y_pred = (y_pred_continuous > 0.1).astype(int)
    
    acc = accuracy_score(y_val, y_pred)
    hl = hamming_loss(y_val, y_pred)
    
    print(f"LassoLars Results:")
    print(f"  Accuracy (Subset): {acc:.4f}")
    print(f"  Hamming Loss: {hl:.4f}")
    return acc

def run_nn(X_train, X_val, y_train, y_val):
    print("\n--- Running Neural Network ---")
    feature_cols = [c for c in X_train.columns if c.startswith('e')]
    
    print("Training MLP Classifier...")
    # MLP with early stopping
    mlp = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=50, random_state=42, verbose=True, early_stopping=True)
    mlp.fit(X_train[feature_cols], y_train)
    
    y_pred = mlp.predict(X_val[feature_cols])
    
    acc = accuracy_score(y_val, y_pred)
    hl = hamming_loss(y_val, y_pred)
    
    print(f"Neural Network Results:")
    print(f"  Accuracy (Subset): {acc:.4f}")
    print(f"  Hamming Loss: {hl:.4f}")
    return acc

def main():
    X, y = load_data()
    if X is None:
        return

    explore_data(X, y)
    
    # Split data for evaluation
    print("\nSplitting data into Train/Validation (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scores = {}
    
    # Run Models
    scores['KNN'] = run_knn(X_train, X_val, y_train, y_val)
    scores['Lasso'] = run_lasso(X_train, X_val, y_train, y_val)
    scores['LassoLars'] = run_lassolars(X_train, X_val, y_train, y_val)
    scores['NN'] = run_nn(X_train, X_val, y_train, y_val)
    
    print("\n--- Final Comparison ---")
    for name, score in scores.items():
        print(f"{name}: Accuracy = {score:.4f}")

    print("\nDetailed Analysis for Neural Network (Sample):")
    # Doing a classification report on a subset of targets might be too big (450 classes).
    # calculating F1 score (samples average)
    from sklearn.metrics import f1_score
    y_pred_nn = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=50, random_state=42, verbose=False, early_stopping=True).fit(X_train[[c for c in X_train.columns if c.startswith('e')]], y_train).predict(X_val[[c for c in X_train.columns if c.startswith('e')]])
    f1 = f1_score(y_val, y_pred_nn, average='samples')
    print(f"NN F1 Score (Samples): {f1:.4f}")

if __name__ == "__main__":
    main()
