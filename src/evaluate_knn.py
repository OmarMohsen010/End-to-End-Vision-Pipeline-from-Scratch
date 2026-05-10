"""
04_evaluate_knn.py
==================
Phase 8: Models Evaluation (From Scratch)

Runs the optimal KNN model on the held-out Test set and calculates
Accuracy, Confusion Matrix, Precision, Recall, and F1 Scores entirely
from scratch using pure NumPy.
"""

import os, sys, time
import numpy as np
import json

# Import your OOP model from the previous script
from knn_baseline import KNNClassifier

FEAT_DIR = r"..\features"

# ══════════════════════════════════════════════════════════════════════════
#  1. From-Scratch Metrics Engine
# ══════════════════════════════════════════════════════════════════════════

def evaluate_predictions(y_true, y_pred, num_classes=10):
    """Calculates all required Phase 8 metrics from scratch."""
    
    # 1. Accuracy
    correct = np.sum(y_true == y_pred)
    accuracy = correct / len(y_true)
    
    # 2. Confusion Matrix (Row = True, Col = Predicted)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    # Extract True Positives, False Positives, False Negatives from CM
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    
    # 3. Precision, Recall, F1 (Per Class)
    # Using np.divide with safety checks to prevent division by zero
    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    recall    = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    
    f1 = np.divide(2 * (precision * recall), (precision + recall), 
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    
    # 4. Macro and Weighted F1
    macro_f1 = np.mean(f1)
    
    # Weighted F1 accounts for class imbalance
    class_counts = np.sum(cm, axis=1)
    weighted_f1 = np.sum(f1 * class_counts) / np.sum(class_counts)
    
    return accuracy, cm, precision, recall, f1, macro_f1, weighted_f1

# ══════════════════════════════════════════════════════════════════════════
#  2. Execution 
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading datasets...")
    X_train = np.load(os.path.join(FEAT_DIR, "X_train_selected.npy"))
    Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
    X_test  = np.load(os.path.join(FEAT_DIR, "X_test_selected.npy"))
    Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))

    # The optimal K you found during the sweep
    OPTIMAL_K = 13
    
    print(f"\nInitializing KNN with optimal K = {OPTIMAL_K}")
    model = KNNClassifier(k=OPTIMAL_K)
    
    print("Fitting model (memorizing train set)...")
    model.fit(X_train, Y_train)
    
    print("Predicting on held-out Test set")
    t0 = time.time()
    preds = model.predict(X_test)
    print(f"Prediction complete in {time.time() - t0:.1f}s")
    
    # Run the from-scratch evaluation
    acc, cm, prec, rec, f1, mac_f1, wt_f1 = evaluate_predictions(Y_test.astype(int), preds)
    
    # ══════════════════════════════════════════════════════════════════════════
    #  3. Final Report Output
    # ══════════════════════════════════════════════════════════════════════════
    
    CLASS_NAMES = [
        "butterfly", "cat", "chicken", "cow", "dog", 
        "elephant", "horse", "sheep", "spider", "squirrel"
    ]

    print("\n" + "="*60)
    print("                  KNN TEST REPORT")
    print("="*60)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1 Score:    {mac_f1:.4f}")
    print(f"Weighted F1 Score: {wt_f1:.4f}\n")
    
    print("Class-Level Metrics:")
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12} | {prec[i]:<10.4f} | {rec[i]:<10.4f} | {f1[i]:<10.4f}")
        
    print("\nConfusion Matrix:")
    # Print a clean, formatted confusion matrix
    header = "      " + "".join([f"{name[:3]:>5}" for name in CLASS_NAMES])
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:>5}" for val in row])
        print(f"{CLASS_NAMES[i][:3]:<4} |{row_str}")
    print("="*60)


    # ══════════════════════════════════════════════════════════════════════════
    #  4. Save Evaluation Report to Disk
    # ══════════════════════════════════════════════════════════════════════════
    
    report_dict = {
        "model": "KNN_Baseline",
        "optimal_k": OPTIMAL_K,
        "overall_metrics": {
            "accuracy": float(acc),
            "macro_f1": float(mac_f1),
            "weighted_f1": float(wt_f1)
        },
        "class_metrics": {
            name: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1_score": float(f1[i])
            } for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist()
    }
    
    report_path = os.path.join(FEAT_DIR, "knn_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
        
    print(f"\nSaved complete evaluation record to: {report_path}")