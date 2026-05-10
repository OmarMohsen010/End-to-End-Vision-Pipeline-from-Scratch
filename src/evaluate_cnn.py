"""
10_evaluate_cnn.py
==================
Phase 8: CNN Evaluation

Loads the best CNN weights, evaluates on the held-out Test Set in batches, 
and calculates Precision, Recall, and F1 scores from scratch.
"""

import os, json
import numpy as np
from optimizers import Adam
from cnn_network import CustomCNN

DATA_DIR = r"..\data"
FEAT_DIR = r"..\features"

def evaluate_predictions(y_true, y_pred, num_classes=10):
    """From-scratch metrics calculator."""
    correct = np.sum(y_true == y_pred)
    accuracy = correct / len(y_true)
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
        
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    
    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    recall    = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    f1        = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)
    
    macro_f1 = np.mean(f1)
    class_counts = np.sum(cm, axis=1)
    weighted_f1 = np.sum(f1 * class_counts) / np.sum(class_counts)
    
    return accuracy, cm, precision, recall, f1, macro_f1, weighted_f1

if __name__ == "__main__":
    # 1. Load Test Images
    print("Loading 4D Test Tensors...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32) / 255.0
    Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy")).astype(int)
    
    # 2. Build Model & Load Checkpoint
    cnn = CustomCNN(optimizer=Adam(), input_shape=(128, 128, 3))
    cnn.load_checkpoint("cnn_best_model.json")
    
    # 3. Predict in Batches (To save RAM)
    print("Predicting on Test Set (Batched)...")
    batch_size = 32
    preds = []
    
    for i in range(0, X_test.shape[0], batch_size):
        X_b = X_test[i:i+batch_size]
        Z = cnn.forward(X_b)
        A = cnn._stable_softmax(Z)
        preds.extend(np.argmax(A, axis=1))
        
    preds = np.array(preds)
    
    # 4. Evaluate
    acc, cm, prec, rec, f1, mac_f1, wt_f1 = evaluate_predictions(Y_test, preds)
    CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    
    print("\n" + "="*60)
    print("               PHASE 8: CNN TEST REPORT")
    print("="*60)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1 Score:    {mac_f1:.4f}")
    print(f"Weighted F1 Score: {wt_f1:.4f}\n")
    
    # 5. Save JSON Report
    report_dict = {
        "model": "CNN_From_Scratch",
        "overall_metrics": {"accuracy": float(acc), "macro_f1": float(mac_f1), "weighted_f1": float(wt_f1)},
        "class_metrics": {name: {"precision": float(prec[i]), "recall": float(rec[i]), "f1_score": float(f1[i])} for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(FEAT_DIR, "cnn_evaluation_report.json"), "w") as f:
        json.dump(report_dict, f, indent=4)
    print("Report saved successfully.")