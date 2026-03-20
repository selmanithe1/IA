import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score, hamming_loss

def load_data():
    X = np.random.randn(1000, 204)
    Z = np.random.randint(0, 2, (1000, 450))
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

# 2. Modele Lasso (Regularisation L1 pour la Sparsite)
# Alpha controle la force de la regularisation
print("Calcul de l'inference par Lasso (alpha=0.1)...")
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Z_train)

# 3. Prediction et Seuil (Thresholding)
# Le Lasso renvoie des valeurs continues, on seuille a 0.5 pour le binaire
Z_pred_cont = lasso.predict(X_test)
Z_pred = (Z_pred_cont > 0.5).astype(int)

f1 = f1_score(Z_test, Z_pred, average='samples')
h_loss = hamming_loss(Z_test, Z_pred)

print("-" * 30)
print(f"RESULTATS LASSO (Sparsity L1)")
print(f"F1-Score (Samples) : {f1:.4f}")
print(f"Hamming Loss       : {h_loss:.4f}")
print("-" * 30)
