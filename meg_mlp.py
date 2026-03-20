import numpy as np
from sklearn.neural_network import MLPRegressor
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

# 2. Modele MLP (Deep Learning)
# Architecture : 204 (in) -> 200 -> 100 -> 450 (out)
# Activation : ReLU pour les couches cachees
print("Entrainement du Multi-Layer Perceptron (ReLU)...")
mlp = MLPRegressor(
    hidden_layer_sizes=(200, 100), 
    activation='relu', 
    solver='adam', 
    max_iter=500,
    random_state=42,
    verbose=True
)
mlp.fit(X_train, Z_train)

# 3. Prediction et Evaluation
Z_pred_cont = mlp.predict(X_test)
Z_pred = (Z_pred_cont > 0.5).astype(int)

f1 = f1_score(Z_test, Z_pred, average='samples')
h_loss = hamming_loss(Z_test, Z_pred)

print("-" * 30)
print(f"RESULTATS MLP (Deep Learning)")
print(f"F1-Score (Samples) : {f1:.4f}")
print(f"Hamming Loss       : {h_loss:.4f}")
print("-" * 30)
