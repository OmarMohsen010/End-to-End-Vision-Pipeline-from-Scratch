"""
09_cnn_network.py
=================
Phase 5.3: CNN Network Assembler

Stitches together the custom NumPy layers into a full Convolutional Neural Network.
Executes the forward pass, computes Softmax/Cross-Entropy, and runs the 
backward pass through the entire computational graph.
"""

import os, time, json, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom modules
from optimizers import Adam, SGD
from cnn_layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense 

DATA_DIR = r"..\data"
FIG_DIR  = r"..\figures"

# ══════════════════════════════════════════════════════════════════════════
#  1. The Network Compiler
# ══════════════════════════════════════════════════════════════════════════

class CustomCNN:
    def __init__(self, optimizer, input_shape=(128, 128, 3), num_classes=10):
        self.optimizer = optimizer
        
        # Calculate the dimension going into the Dense layer
        # 128x128 -> Conv(pad=1, stride=1) -> 128x128 -> MaxPool(stride=2) -> 64x64
        # 64 * 64 * 8 channels = 32,768 flat features
        H, W, C = input_shape
        conv_out_h, conv_out_w = H, W # Because padding=1, stride=1
        pool_out_h, pool_out_w = conv_out_h // 2, conv_out_w // 2
        flat_dim = pool_out_h * pool_out_w * 8
        
        print(f"Building Network... Flattened dimension will be {flat_dim}")

        # The Computational Graph
        self.layers = [
            Conv2D(filter_size=3, in_channels=C, out_channels=8, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_dim=flat_dim, output_dim=num_classes)
        ]
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def forward(self, X):
        """Pushes data through the network, layer by layer."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_out):
        """Pushes gradients backwards through the network in reverse order."""
        dout = d_out
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_weights(self, l2_lambda=0.0):
        """Asks the optimizer to update weights, injecting L2 regularization."""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                # Add L2 penalty to the gradient: dW_total = dW_base + (lambda * W)
                dW_reg = layer.gradients['W'] + (l2_lambda * layer.params['W'])
                
                layer.params['W'] = self.optimizer.update(f'L{i}_W', layer.params['W'], dW_reg)
                layer.params['b'] = self.optimizer.update(f'L{i}_b', layer.params['b'], layer.gradients['b'])

    # --- Standard Math Helpers ---
    def _stable_softmax(self, Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_true_onehot, y_pred_probs, l2_lambda=0.0):
        """Calculates Base Loss + L2 Penalty across all weighted layers."""
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
        base_loss = -np.sum(y_true_onehot * np.log(y_pred_clipped)) / y_true_onehot.shape[0]
        
        # Add L2 Penalty: (lambda / 2) * sum(W^2) for all Conv and Dense layers
        l2_penalty = 0.0
        if l2_lambda > 0:
            for layer in self.layers:
                if hasattr(layer, 'params') and 'W' in layer.params:
                    l2_penalty += np.sum(layer.params['W'] ** 2)
                    
        return base_loss + (l2_lambda / 2) * l2_penalty

    def _one_hot(self, y, num_classes=10):
        oh = np.zeros((y.size, num_classes))
        oh[np.arange(y.size), y] = 1
        return oh

# ══════════════════════════════════════════════════════════════════════════
#  2. The Training Loop (Mini-Batch)
# ══════════════════════════════════════════════════════════════════════════

    def fit(self, X_train, Y_train, X_val, Y_val, epochs=15, batch_size=32, l2_lambda=0.001, patience=5):
        num_samples = X_train.shape[0]
        Y_train_oh = self._one_hot(Y_train.astype(int))
        Y_val_oh = self._one_hot(Y_val.astype(int))
        
        print(f"\nStarting CNN Training on {num_samples} images with L2={l2_lambda}...")
        
        # --- EARLY STOPPING SETUP ---
        best_val_loss = float('inf')
        patience_counter = 0
        log_file = "cnn_logs.csv"
        
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"])

        for epoch in range(epochs):
            t_epoch = time.time()
            indices = np.random.permutation(num_samples)
            X_shuffled, Y_shuffled = X_train[indices], Y_train_oh[indices]
            
            epoch_loss = 0.0
            epoch_correct = 0
            num_batches = int(np.ceil(num_samples / batch_size))
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1:02d}/{epochs}")
            for i in pbar:
                start, end = i * batch_size, min((i + 1) * batch_size, num_samples)
                X_b, Y_b = X_shuffled[start:end], Y_shuffled[start:end]
                N_b = X_b.shape[0]
                
                # 1. FORWARD PASS
                Z = self.forward(X_b)
                A = self._stable_softmax(Z)
                
                # We now pass l2_lambda to the loss function
                batch_loss = self._cross_entropy_loss(Y_b, A, l2_lambda)
                epoch_loss += batch_loss * N_b
                epoch_correct += np.sum(np.argmax(A, axis=1) == np.argmax(Y_b, axis=1))
                
                # 2. BACKWARD PASS 
                dZ = (A - Y_b) / N_b  
                self.backward(dZ)
                
                # 3. OPTIMIZER UPDATE (Now with L2 gradients!)
                self.update_weights(l2_lambda)
                
                pbar.set_postfix({'loss': f"{batch_loss:.3f}"})
                
            # --- EPOCH EVALUATION ---
            train_loss = epoch_loss / num_samples
            train_acc = epoch_correct / num_samples
            
            val_preds = []
            val_loss = 0.0
            for i in range(0, X_val.shape[0], batch_size):
                X_v_b = X_val[i:i+batch_size]
                Y_v_b = Y_val_oh[i:i+batch_size]
                Z_v = self.forward(X_v_b)
                A_v = self._stable_softmax(Z_v)
                val_loss += self._cross_entropy_loss(Y_v_b, A_v, l2_lambda) * X_v_b.shape[0]
                val_preds.extend(np.argmax(A_v, axis=1))
                
            val_loss /= X_val.shape[0]
            val_acc = np.sum(np.array(val_preds) == Y_val) / len(Y_val)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Write to CSV
            with open(log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch+1, train_loss, val_loss, train_acc, val_acc, self.optimizer.lr])
                
            print(f"--> Time: {time.time()-t_epoch:.1f}s | T-Loss: {train_loss:.3f} | V-Loss: {val_loss:.3f} | T-Acc: {train_acc*100:.2f}% | V-Acc: {val_acc*100:.2f}%")
            
            # --- EARLY STOPPING LOGIC ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch=epoch+1, filepath="cnn_best_model.json")
                print("    [*] New best validation loss! Checkpoint saved.")
            else:
                patience_counter += 1
                print(f"    [!] No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n[Early Stopping] Triggered at epoch {epoch+1}. Restoring best weights from file.")
                    self.load_checkpoint("cnn_best_model.json")
                    break

    def save_checkpoint(self, epoch, filepath="cnn_best_model.json"):
        """Saves the weights of all layers to a JSON file."""
        model_state = {"epoch": epoch, "weights": {}}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                model_state["weights"][f"L{i}_W"] = layer.params['W'].tolist()
                model_state["weights"][f"L{i}_b"] = layer.params['b'].tolist()
        with open(filepath, "w") as f:
            json.dump(model_state, f)

    def load_checkpoint(self, filepath="cnn_best_model.json"):
        """Loads weights back into the layers."""
        if not os.path.exists(filepath):
            print(f"Checkpoint {filepath} not found.")
            return
        with open(filepath, "r") as f:
            model_state = json.load(f)
            
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                layer.params['W'] = np.array(model_state["weights"][f"L{i}_W"])
                layer.params['b'] = np.array(model_state["weights"][f"L{i}_b"])
        print(f"Loaded CNN weights from Epoch {model_state.get('epoch', 'Unknown')}.")
# ══════════════════════════════════════════════════════════════════════════
#  3. Execution
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
        print("Loading 4D Image Tensors...")
        X_train = np.load(os.path.join(DATA_DIR, "X_train_augmented.npy")).astype(np.float32) / 255.0
        Y_train = np.load(os.path.join(DATA_DIR, "Y_train_augmented.npy"))
        
        X_val = np.load(os.path.join(DATA_DIR, "X_val.npy")).astype(np.float32) / 255.0
        Y_val = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
        
        opt = Adam(lr=0.001)
        cnn = CustomCNN(optimizer=opt, input_shape=(128, 128, 3))
        
        # --- THE UPDATE ---
        # We removed the [:500] slices and bumped the batch size slightly
        print("\n--- RUNNING FULL VECTORIZED CNN ---")
        cnn.fit(X_train, Y_train, X_val, Y_val, epochs=5, batch_size=32)

        # Plot CNN Learning Curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0f1117")
        
        # Loss Plot
        ax1.set_facecolor("#1a1a2e")
        ax1.plot(cnn.history['train_loss'], label="Train Loss", color="#3498db", linewidth=2)
        ax1.plot(cnn.history['val_loss'], label="Val Loss", color="#e74c3c", linewidth=2)
        ax1.set_title("CNN Cross-Entropy Loss", color="white", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch", color="white")
        ax1.set_ylabel("Loss", color="white")
        ax1.tick_params(colors="white")
        ax1.grid(True, linestyle="--", alpha=0.3, color="#fff")
        ax1.legend()
        
        # Accuracy Plot
        ax2.set_facecolor("#1a1a2e")
        ax2.plot([a * 100 for a in cnn.history['train_acc']], label="Train Acc", color="#2ecc71", linewidth=2)
        ax2.plot([a * 100 for a in cnn.history['val_acc']], label="Val Acc", color="#f39c12", linewidth=2)
        ax2.set_title("CNN Accuracy", color="white", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch", color="white")
        ax2.set_ylabel("Accuracy (%)", color="white")
        ax2.tick_params(colors="white")
        ax2.grid(True, linestyle="--", alpha=0.3, color="#fff")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "cnn_learning_curves2.png"), dpi=150, facecolor=fig.get_facecolor())