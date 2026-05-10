"""
07_evaluate_softmax.py
======================
Phase 8: Models Evaluation (Softmax)

Loads the best Softmax Checkpoint and evaluates it on the held-out Test set,
calculating Accuracy, Confusion Matrix, and F1 scores from scratch.
"""

import os, json
import numpy as np

FEAT_DIR = r"..\features"

def evaluate_predictions(y_true, y_pred, num_classes=10):
    """Calculates all required Phase 8 metrics from scratch."""
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

def stable_softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

if __name__ == "__main__":
    # 1. Load Test Data
    X_test  = np.load(os.path.join(FEAT_DIR, "X_test_selected.npy"))
    Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy")).astype(int)
    
    # 2. Load Best Weights
    print("Loading best Softmax checkpoint...")
    with open("best_model.json", "r") as f:
        chkpt = json.load(f)
        W = np.array(chkpt["W"])
        b = np.array(chkpt["b"])
        
    # 3. Predict
    print("Predicting on Test Set...")
    Z = np.dot(X_test, W) + b
    A = stable_softmax(Z)
    preds = np.argmax(A, axis=1)
    
    # 4. Evaluate
    acc, cm, prec, rec, f1, mac_f1, wt_f1 = evaluate_predictions(Y_test, preds)
    
    CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    
    print("\n" + "="*60)
    print("             PHASE 8: SOFTMAX TEST REPORT")
    print("="*60)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1 Score:    {mac_f1:.4f}")
    print(f"Weighted F1 Score: {wt_f1:.4f}\n")
    
    # Save to JSON
    report_dict = {
        "model": "Softmax_Regression",
        "checkpoint_epoch": chkpt.get("epoch", "N/A"),
        "overall_metrics": {"accuracy": float(acc), "macro_f1": float(mac_f1), "weighted_f1": float(wt_f1)},
        "class_metrics": {name: {"precision": float(prec[i]), "recall": float(rec[i]), "f1_score": float(f1[i])} for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist()
    }
    
    report_path = os.path.join(FEAT_DIR, "softmax_evaluation_report.json")
    with open(report_path, "w") as f: json.dump(report_dict, f, indent=4)
    print(f"Saved evaluation record to: {report_path}")