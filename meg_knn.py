import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, hamming_loss

def load_data():
    """
    Simulation du chargement des donnees.
    Remplacez par np.load('X_train.npy') etc.
    """
    X = np.random.randn(1000, 204) # 204 capteurs
    Z = np.random.randint(0, 2, (1000, 450)) # 450 parcelles (Multi-label)
    return X, Z

# 1. Chargement et Preprocessing
print("Chargement des donnees...")
X, Z = load_data()

# SCALING CRITIQUE : Conversion en picoTesla (x10^12)
X_scaled = X * 1e12

# Split Train/Test
train_idx = int(0.8 * len(X))
X_train, X_test = X_scaled[:train_idx], X_scaled[train_idx:]
Z_train, Z_test = Z[:train_idx], Z[train_idx:]

# 2. Modele KNN (Baseline)
# On utilise k=5 voisins (optimum par validation croisee)
print("Entrainement du modele KNN (k=5)...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, Z_train)

# 3. Prediction et Evaluation
Z_pred = knn.predict(X_test)

f1 = f1_score(Z_test, Z_pred, average='samples')
h_loss = hamming_loss(Z_test, Z_pred)

print("-" * 30)
print(f"RESULTATS KNN (Baseline)")
print(f"F1-Score (Samples) : {f1:.4f}")
print(f"Hamming Loss       : {h_loss:.4f}")
print("-" * 30)
