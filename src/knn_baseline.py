"""
knn_baseline.py
==================
Phase 5.1: K-Nearest Neighbors (OOP Implementation)

Implements a pure NumPy KNN Classifier following standard ML API design
(fit/predict). Sweeps for the best K on the validation set and plots the results.
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────
FEAT_DIR = r"..\features"
FIG_DIR  = r"..\figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  1. The KNN Classifier Class
# ══════════════════════════════════════════════════════════════════════════

class KNNClassifier:
    
    def __init__(self,k=5):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        'Trains' the model. For KNN, this simply means memorizing the 
        training dataset in memory.
        """
        self.X_train = X
        self.Y_train = y.astype(int)

    def predict(self,X_query:np.ndarray) -> np.ndarray:
        """
        Predicts labels for a query set using vectorized Euclidean distance.
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before calling predict.")
        
        predictions = np.zeros(len(X_query),dtype=int)
        
        # Process one query at a time to prevent RAM overflow,
        # while leveraging fast vectorized subtraction across the train set
        for i in range(len(X_query)):
            # Broadcast subtraction: (N_train, features) - (features,)
            distances = np.sqrt(np.sum((self.X_train - X_query[i])**2,axis=1))

            # Get indices of the top K smallest distances
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.Y_train[nearest_indices]

            # Find the Mode (most common label among the K neighbors)
            predictions[i] = np.bincount(nearest_labels).argmax()
        
        return predictions
    
# ══════════════════════════════════════════════════════════════════════════
#  2. The K-Sweep & Evaluation Loop
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading MRMR-selected feature matrices...")
    X_train = np.load(os.path.join(FEAT_DIR, "X_train_selected.npy"))
    Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(FEAT_DIR, "X_val_selected.npy"))
    Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))

    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape:   {X_val.shape}")

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = []

    print("\nStarting K-Sweep on Validation Set...")
    t0 = time.time()

    for k in tqdm(k_values, desc="K-Sweep", unit="k"):
        # 1. Instantiate the model
        model = KNNClassifier(k=k)
        
        # 2. Fit the model
        model.fit(X_train, Y_train)
        
        # 3. Predict on the validation set
        preds = model.predict(X_val)
        
        # 4. Calculate raw accuracy
        correct = np.sum(preds == Y_val)
        acc = correct / len(Y_val)
        accuracies.append(acc)

    elapsed = time.time() - t0

# ══════════════════════════════════════════════════════════════════════════
#  3. Reporting & Plotting
# ══════════════════════════════════════════════════════════════════════════

    best_idx = np.argmax(accuracies)
    best_k = k_values[best_idx]
    best_acc = accuracies[best_idx]

    print("\n" + "="*50)
    print("  KNN Sweep Complete")
    print("="*50)
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print(f"  Optimal K:    {best_k}")
    print(f"  Validation Accuracy: {best_acc * 100:.2f}%")
    for k, acc in zip(k_values, accuracies):
        print(f"    K={k:2d} -> Acc: {acc * 100:.2f}%")
    print("="*50)

    # Plot the results for the Phase 9 Report
    plt.figure(figsize=(8, 5))
    
    # Set dark theme to match your previous panels
    plt.gca().set_facecolor("#1a1a2e")
    plt.gcf().patch.set_facecolor("#0f1117")
    
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='#3498db', linewidth=2)
    
    # Highlight the best K
    plt.plot(best_k, best_acc, marker='*', color='#e74c3c', markersize=15, label=f'Best K ({best_k})')
    
    plt.title("KNN Validation Accuracy vs. K", color="white", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Neighbors (K)", color="white", fontsize=12)
    plt.ylabel("Validation Accuracy", color="white", fontsize=12)
    plt.xticks(k_values, color="white")
    plt.yticks(color="white")
    
    for sp in plt.gca().spines.values(): 
        sp.set_edgecolor("#444")
        
    plt.grid(True, linestyle='--', alpha=0.3, color="#fff")
    plt.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "knn_k_sweep.png")
    plt.savefig(out_path, dpi=150, facecolor=plt.gcf().get_facecolor())
    print(f"\nSaved sweep plot to {out_path}")
