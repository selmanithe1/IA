import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, hamming_loss
import seaborn as sns

# Set style
sns.set_style("whitegrid")

def load_data(data_dir="data"):
    print("Loading subset of data for analysis...")
    x_train_path = os.path.join(data_dir, "train", "X.csv.gz")
    y_train_path = os.path.join(data_dir, "train", "target.npz")
    
    if os.path.exists(x_train_path) and os.path.exists(y_train_path):
        # Load larger subset for better hyperparam curves
        X_train = pd.read_csv(x_train_path, nrows=5000) 
        feature_cols = [c for c in X_train.columns if c not in ['subject', 'L_path']]
        X = X_train[feature_cols].values * 1e12 
        y = sparse.load_npz(y_train_path).toarray()[:5000]
        return X, y
    return None, None

def analyze_knn(X_train, X_val, y_train, y_val):
    print("Analyzing KNN Hyperparameters...")
    k_values = range(1, 21)
    accuracies = []
    hamming_losses = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
        hamming_losses.append(hamming_loss(y_val, y_pred))
        
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
    plt.title("KNN: Effet de k sur l'Exactitude (Subset Accuracy)")
    plt.xlabel("Nombre de voisins (k)")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig("knn_analysis.png")
    print("Saved knn_analysis.png")

def analyze_lasso(X_train, X_val, y_train, y_val):
    print("Analyzing Lasso Hyperparameters...")
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    hamming_losses = []
    
    for alpha in alphas:
        # Lasso is slow, use fewer iters for sweeping
        lasso = Lasso(alpha=alpha, max_iter=500) 
        lasso.fit(X_train, y_train)
        y_pred_cont = lasso.predict(X_val)
        y_pred = (y_pred_cont > 0.1).astype(int)
        hamming_losses.append(hamming_loss(y_val, y_pred))
        
    # Plot Hamming Loss (Accuracy is likely 0)
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, hamming_losses, marker='s', linestyle='--', color='red')
    plt.title("Lasso: Effet de Alpha sur l'Erreur (Hamming Loss)")
    plt.xlabel("Alpha (Régularisation)")
    plt.ylabel("Hamming Loss (Plus bas est mieux)")
    plt.grid(True)
    plt.savefig("lasso_analysis.png")
    print("Saved lasso_analysis.png")

def analyze_nn_training(X_train, y_train):
    print("Analyzing Neural Network Training...")
    # Increase max_iter to show full convergence curve
    # Add early_stopping and validation_fraction to populate validation_scores_
    mlp = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=100, random_state=42, verbose=True,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_train, y_train)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.set_ylabel('Loss', color='purple')
    ax1.plot(mlp.loss_curve_, color='purple', label="Training Loss")
    ax1.tick_params(axis='y', labelcolor='purple')
    
    # Validation Score (Accuracy) - usually available if early_stopping=True
    if hasattr(mlp, 'validation_scores_'):
        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Score (Accuracy)', color='green')
        ax2.plot(mlp.validation_scores_, color='green', linestyle='--', label="Validation Score")
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title("Courbe d'Apprentissage (Training Loss vs Validation)")
    plt.grid(True)
    plt.savefig("nn_training.png")
    print("Saved nn_training.png")

def plot_target_stem(y):
    print("Generating Target Stem Plot...")
    # Take the 2nd sample (index 1) which has 3 sources as an example
    sample_idx = 1 
    target_vector = y[sample_idx]
    
    plt.figure(figsize=(12, 4))
    plt.stem(target_vector, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title(f"Visualisation de la Sparsité (Cible Échantillon #{sample_idx})")
    plt.xlabel("Indice de la Parcelle (0-449)")
    plt.ylabel("Activation (0 ou 1)")
    plt.xlim(0, 450)
    plt.grid(True, alpha=0.3)
    plt.savefig("target_stem.png")
    print("Saved target_stem.png")

def plot_comparison():
    print("Generating Comparison Bar Chart...")
    # Data manually aggregated from previous runs for visualization
    models = ['KNN', 'Lasso', 'MLP (Neural Net)']
    f1_scores = [0.03, 0.01, 0.08] # Approx values
    hamming_losses = [0.005, 0.008, 0.0045] # Lower is better
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # F1 Score Bar
    rects1 = ax1.bar(x - width/2, f1_scores, width, label='F1-Score (Higher is better)', color='skyblue')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 0.1)
    
    # Hamming Loss Bar (on secondary axis to compare scales)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, hamming_losses, width, label='Hamming Loss (Lower is better)', color='salmon')
    ax2.set_ylabel('Hamming Loss')
    ax2.set_ylim(0, 0.01)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title("Comparaison des Performances des Modèles")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig("model_comparison.png")
    print("Saved model_comparison.png")

def analyze_sparsity(X_train, X_val, y_train, y_val):
    print("Analyzing Lasso Sparsity...")
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    non_zero_counts = []
    accuracies = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=500)
        lasso.fit(X_train, y_train)
        
        # Count non-zeros in coefficients (for one target, but here we have 450)
        # We need average non-zeros per sample in prediction
        y_pred_cont = lasso.predict(X_val)
        y_pred = (y_pred_cont > 0.1).astype(int)
        
        # Average number of active sources predicted
        avg_active = np.mean(np.sum(y_pred, axis=1))
        non_zero_counts.append(avg_active)
        accuracies.append(accuracy_score(y_val, y_pred))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:green'
    ax1.set_xlabel('Alpha (Régularisation)')
    ax1.set_ylabel('Nombre moyen de sources actives prédites', color=color)
    ax1.plot(alphas, non_zero_counts, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=1.5, color='gray', linestyle='--', label="Moyenne Réelle (~1.5)") # Approx real average
    
    plt.title("Lasso : Impact de la Régularisation sur la Sparsité")
    plt.grid(True)
    plt.legend()
    plt.savefig("lasso_sparsity.png")
    print("Saved lasso_sparsity.png")

if __name__ == "__main__":
    X, y = load_data()
    if X is not None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        analyze_knn(X_train, X_val, y_train, y_val)
        analyze_lasso(X_train, X_val, y_train, y_val)
        analyze_nn_training(X_train, y_train)
        analyze_sparsity(X_train, X_val, y_train, y_val)
        plot_target_stem(y)
        plot_comparison()
