import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import sparse
import seaborn as sns

def load_data(data_dir="data"):
    # Load X and y for plots
    x_train_path = os.path.join(data_dir, "train", "X.csv.gz")
    y_train_path = os.path.join(data_dir, "train", "target.npz")
    
    if os.path.exists(x_train_path) and os.path.exists(y_train_path):
        X_train = pd.read_csv(x_train_path, nrows=1000) # Load subset for speed
        feature_cols = [c for c in X_train.columns if c not in ['subject', 'L_path']]
        X = X_train[feature_cols].values * 1e12 # Apply scaling
        y = sparse.load_npz(y_train_path).toarray()[:1000]
        return X, y
    return None, None

def plot_distribution(y):
    if y is None: return
    n_sources = np.sum(y, axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(n_sources, bins=np.arange(0, 10)-0.5, kde=False, color='skyblue', edgecolor='black')
    plt.title("Distribution du Nombre de Sources Actives (Sparsité)")
    plt.xlabel("Nombre de Sources (Parcelles actives)")
    plt.ylabel("Nombre d'Échantillons")
    plt.xticks(range(10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("distribution_sources.png")
    print("Saved distribution_sources.png")

def plot_signal_sample(X):
    if X is None: return
    # Plot the signal of the first sample across all 204 sensors
    plt.figure(figsize=(12, 5))
    plt.plot(X[0], color='#2c3e50')
    plt.title("Exemple de Signal MEG (Échantillon 0) sur 204 Capteurs")
    plt.xlabel("Index du Capteur (0-203)")
    plt.ylabel("Amplitude du Champ Magnétique (pT)")
    plt.grid(True, alpha=0.3)
    plt.savefig("signal_sample.png")
    print("Saved signal_sample.png")

def plot_correlation_matrix(X):
    if X is None: return
    # Correlation between first 50 sensors to show redundancy
    plt.figure(figsize=(10, 8))
    corr = np.corrcoef(X[:, :50].T)
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .5})
    plt.title("Matrice de Corrélation des 50 Premiers Capteurs")
    plt.savefig("sensor_correlation.png")
    print("Saved sensor_correlation.png")

def plot_results():
    # Hardcoded results from our analysis
    models = ['KNN (k=5)', 'Lasso', 'Neural Network']
    accuracies = [3.42, 0.0, 2.32]
    hamming = [0.0051, 0.0050, 0.0048]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Modèle', fontsize=12)
    ax1.set_ylabel('Exactitude (Subset Accuracy) [%]', color=color, fontsize=12)
    bars = ax1.bar(models, accuracies, color=['#4CAF50', '#F44336', '#2196F3'], alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 5)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Hamming Loss (Plus bas est mieux)', color=color, fontsize=12)  
    ax2.plot(models, hamming, color=color, marker='o', linestyle='dashed', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.0040, 0.0060)
    
    plt.title("Performance des Modèles : Accuracy vs Hamming Loss")
    fig.tight_layout()  
    plt.savefig("resultats_complets.png")
    print("Saved resultats_complets.png")

if __name__ == "__main__":
    X, y = load_data()
    if X is not None:
        plot_distribution(y)
        plot_signal_sample(X)
        plot_correlation_matrix(X)
    plot_results()
