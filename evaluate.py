"""
evaluate.py  (PPT lead)

Evaluates the trained model on the test set (as in Practice Lab 7, section 2.9):
predictions, accuracy, confusion matrix, precision, recall, F1-score,
classification report and graphs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)


def evaluate(model, X_test, y_test):
    # Predict the classes of all test matches
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\n===== RESULTS ON THE TEST SET =====")
    print("Accuracy:", round(accuracy, 3), f"({accuracy * 100:.1f}%)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    print(f"True Negative (TN) : {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive (TP) : {tp}")

    # Precision, recall and F1-score
    print("\nPrecision:", round(precision_score(y_test, y_pred), 3))
    print("Recall   :", round(recall_score(y_test, y_pred), 3))
    print("F1-score :", round(f1_score(y_test, y_pred), 3))

    # Classification report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Graph 1: confusion matrix as a heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Graph 2: actual vs predicted class counts
    actual_counts = y_test.value_counts().reindex([0, 1], fill_value=0)
    predicted_counts = pd.Series(y_pred).value_counts().reindex([0, 1], fill_value=0)

    x = np.arange(2)
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, actual_counts.values, width, label="Actual")
    plt.bar(x + width / 2, predicted_counts.values, width, label="Predicted")
    plt.xticks(x, ["Class 0", "Class 1"])
    plt.title("Actual vs Predicted Class Counts")
    plt.xlabel("Class")
    plt.ylabel("Number of matches")
    plt.legend()
    plt.savefig("outputs/actual_vs_predicted_counts.png")
    plt.close()

    # Graph 3: the 10 features with the biggest coefficients (b1, b2, ... of the model)
    coefficients = pd.Series(model.named_steps["classifier"].coef_[0], index=X_test.columns)
    top = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index).head(10)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=top.values, y=top.index, color="steelblue")
    plt.title("Top 10 features (positive = pushes towards class 1)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig("outputs/top_features.png")
    plt.close()

    print("Graphs saved in the outputs/ folder")
    return y_pred


if __name__ == "__main__":
    from data_prep import load_and_split
    from train_model import train
    X_train, X_test, y_train, y_test = load_and_split()
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
