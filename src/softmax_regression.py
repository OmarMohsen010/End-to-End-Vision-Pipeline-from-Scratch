"""
05_softmax_regression.py
========================
Phase 5.2: Softmax Regression (OOP Implementation)

Trains a multiclass Softmax classifier from scratch using Mini-Batch Gradient Descent.
Includes a numerically stable Softmax, epsilon-clipped Cross-Entropy Loss, 
and epoch logging to track the learning curve.
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from optimizers import Adam , SGD

FEAT_DIR = r"..\features"
FIG_DIR  = r"..\figures"

# ══════════════════════════════════════════════════════════════════════════
#  1. The Softmax Classifier Class
# ══════════════════════════════════════════════════════════════════════════

class SoftmaxRegression:
    def __init__(self, input_dim=100, num_classes=10, optimizer=None, batch_size=32, epochs=50):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        
        # If no optimizer is passed, default to basic SGD
        if optimizer is None:
            self.optimizer = SGD(lr=0.01)
        else:
            self.optimizer = optimizer
        
        # Initialize Weights (W) and Bias (b) randomly
        # W shape: (100, 10), b shape: (10,)
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        
        # History tracking for plotting
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _stable_softmax(self, Z):
        """Computes Softmax safely by preventing np.exp() overflow."""
        # Subtract the max value of each row for numerical stability
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_true_onehot, y_pred_probs, l2_lambda=0.0):
        """Calculates loss with Epsilon clipping and L2 Regularization."""
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
        base_loss = -np.sum(y_true_onehot * np.log(y_pred_clipped)) / y_true_onehot.shape[0]
        
        # Add L2 Penalty: (lambda / 2) * sum(W^2)
        l2_penalty = (l2_lambda / 2) * np.sum(self.W ** 2)
        return base_loss + l2_penalty

    def _one_hot(self, y):
        """Converts integer labels [2, 0] into one-hot arrays [[0,0,1..], [1,0,0..]]"""
        one_hot = np.zeros((y.size, self.num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def fit(self, X_train, Y_train, X_val, Y_val, l2_lambda=0.0, patience=15, clip_value=1.0, log_file="softmax_logs.csv"):
        import csv, json
        num_samples = X_train.shape[0]
        Y_train_oh = self._one_hot(Y_train.astype(int))
        Y_val_oh = self._one_hot(Y_val.astype(int))
        
        # Early Stopping & Checkpoint Setup
        best_val_loss = float('inf')
        patience_counter = 0
        log_file = "logs.csv"
        
        # Initialize CSV
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"])

        print(f"Training Softmax with {self.optimizer.__class__.__name__}...")
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled, Y_shuffled = X_train[indices], Y_train_oh[indices]
            epoch_loss = 0.0
            epoch_correct = 0
            
            # --- MINI-BATCH LOOP ---
            num_batches = int(np.ceil(num_samples / self.batch_size))
            for i in range(num_batches):
                start, end = i * self.batch_size, min((i + 1) * self.batch_size, num_samples)
                X_b, Y_b = X_shuffled[start:end], Y_shuffled[start:end]
                N_b = X_b.shape[0]
                
                # Forward Pass
                Z = np.dot(X_b, self.W) + self.b
                A = self._stable_softmax(Z)
                epoch_loss += self._cross_entropy_loss(Y_b, A, l2_lambda) * N_b
                
                # Count correct predictions in this batch
                batch_preds = np.argmax(A, axis=1)
                batch_true = np.argmax(Y_b, axis=1)
                epoch_correct += np.sum(batch_preds == batch_true)


                # Backward Pass (With L2 Gradient)
                dZ = A - Y_b
                dW = (np.dot(X_b.T, dZ) / N_b) + (l2_lambda * self.W)  # Added L2 Derivative
                db = np.sum(dZ, axis=0) / N_b
                
                # Gradient Clipping
                dW = np.clip(dW, -clip_value, clip_value)
                db = np.clip(db, -clip_value, clip_value)
                
                # Optimizer Update
                self.W = self.optimizer.update('W', self.W, dW)
                self.b = self.optimizer.update('b', self.b, db)
                
            # --- EPOCH EVALUATION ---
            train_loss = epoch_loss / num_samples
            train_acc = epoch_correct / num_samples
            val_preds = self.predict(X_val)
            val_acc = np.sum(val_preds == Y_val) / len(Y_val)
            
            
            A_val = self._stable_softmax(np.dot(X_val, self.W) + self.b)
            val_loss = self._cross_entropy_loss(Y_val_oh, A_val, l2_lambda)
            
            # Logging
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            with open(log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch+1, train_loss, val_loss, train_acc, val_acc, self.optimizer.lr])
            
            # Print Progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
                
            # --- EARLY STOPPING & CHECKPOINTING ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save Checkpoint
                with open("best_model.json", "w") as f:
                    json.dump({"W": self.W.tolist(), "b": self.b.tolist(), "epoch": epoch+1}, f)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[Early Stopping] Triggered at epoch {epoch+1}. Restoring best weights.")
                    # Load best weights
                    with open("best_model.json", "r") as f:
                        best_weights = json.load(f)
                        self.W = np.array(best_weights["W"])
                        self.b = np.array(best_weights["b"])
                    break
    
    def load_checkpoint(self, filepath="best_model.json"):
        """Loads saved weights and biases to resume training or run inference."""
        import json
        import os
        
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint '{filepath}' not found. Starting from scratch.")
            return
            
        with open(filepath, "r") as f:
            checkpoint = json.load(f)
            self.W = np.array(checkpoint["W"])
            self.b = np.array(checkpoint["b"])
            
        epoch_saved = checkpoint.get("epoch", "Unknown")
        print(f"Successfully loaded model weights from Epoch {epoch_saved}.")


    def predict(self, X_query):
        """Runs the forward pass and returns the class with the highest probability."""
        Z = np.dot(X_query, self.W) + self.b
        A = self._stable_softmax(Z)
        return np.argmax(A, axis=1)

# ══════════════════════════════════════════════════════════════════════════
#  2. Execution and Plotting
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading MRMR-selected features...")
    X_train = np.load(os.path.join(FEAT_DIR, "X_train_selected.npy"))
    Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(FEAT_DIR, "X_val_selected.npy"))
    Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))

    # Initialize and Train
    t0 = time.time()
    
    # Instantiate the optimizer
    # sgd_opt = SGD(lr=0.001)
    adam_opt = Adam(lr=0.001)
    # Create the model, batch_size = 1 (SGD)
    model = SoftmaxRegression(input_dim=100, num_classes=10, optimizer=adam_opt, batch_size=16, epochs=100)

    # 2. LOAD PREVIOUS WEIGHTS (If they exist)
    # model.load_checkpoint("best_model.json")

    model.fit(X_train, Y_train, X_val, Y_val)
    
    print(f"\nTraining Complete in {time.time() - t0:.1f}s")
    
    # Evaluate Accuracy
    val_preds = model.predict(X_val)
    acc = np.sum(val_preds == Y_val) / len(Y_val)
    print(f"Final Validation Accuracy: {acc * 100:.2f}%")
    
    # Plot Learning Curve for the Report
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    
    # Left Plot: Loss
    ax1.set_facecolor("#1a1a2e")
    ax1.plot(model.history['train_loss'], label="Train Loss", color="#3498db", linewidth=2)
    ax1.plot(model.history['val_loss'], label="Val Loss", color="#e74c3c", linewidth=2)
    ax1.set_title("Cross-Entropy Loss", color="white", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white", fontsize=12)
    ax1.set_ylabel("Loss", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#444")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#fff")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")
    
    # Right Plot: Accuracy
    ax2.set_facecolor("#1a1a2e")
    # Multiply by 100 to show percentages
    train_acc_pct = [acc * 100 for acc in model.history['train_acc']]
    val_acc_pct = [acc * 100 for acc in model.history['val_acc']]
    
    ax2.plot(train_acc_pct, label="Train Acc", color="#2ecc71", linewidth=2)
    ax2.plot(val_acc_pct, label="Val Acc", color="#f39c12", linewidth=2)
    ax2.set_title("Classification Accuracy", color="white", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#fff")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "softmax_learning_curves_ADAM_2.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved learning curves to {out_path}")