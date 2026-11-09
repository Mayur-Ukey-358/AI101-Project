"""
data_prep.py  (Documentation lead)

Steps follow Practice Lab 7 (Logistic Regression):
read the data -> understand it -> preprocess -> prepare X and y -> split.
The other files use it like this:
    X_train, X_test, y_train, y_test = load_and_split()
(Scaling is done later, inside the model Pipeline in train_model.py.)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------------- SETTINGS FOR OUR DATASET ----------------
DATA_FILE = "synthetic_basketball_dataset.csv"   # path of the CSV inside the repo
target_column = "Winner"                              # 1 = Team A wins, 0 = Team B wins
WIN_VALUE = None                                      # Winner is already 0/1, so no change needed
drop_columns = ["Match_ID"]                           # ID column, it does not help prediction
# -----------------------------------------------------------


def load_and_split():
    # 1. Read the dataset
    df = pd.read_csv(DATA_FILE)
    print("Dataset shape (rows, columns):", df.shape)
    print("Column names:", df.columns.tolist())

    # 2. Understand the dataset
    print("\nFirst five rows:")
    print(df.head())
    print("\nData types and non-null counts:")
    df.info()
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # 3. Preprocess
    df = df.drop_duplicates()
    df = df.drop(columns=drop_columns)                 # remove ids / columns that give the answer away
    df = df.dropna(subset=[target_column])             # rows without a known target cannot be used

    if WIN_VALUE is not None:                          # text target -> 1 / 0
        df[target_column] = (df[target_column] == WIN_VALUE).astype(int)
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError("The target column has text values. Set WIN_VALUE in data_prep.py "
                         "(for example WIN_VALUE = 'Team A').")

    df = df.fillna(df.median(numeric_only=True))       # fill missing numbers with the median
    df = pd.get_dummies(df, drop_first=True, dtype=int)  # text columns -> numbers (one-hot encoding)
    print("\nDataset shape after encoding:", df.shape)

    # 4. Prepare features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Class counts:", y.value_counts().sort_index().to_dict())

    # 5. Exploration graphs (saved as pictures)
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.countplot(x=y)
    plt.title("Number of matches per class")
    plt.xlabel("Class")
    plt.ylabel("Number of matches")
    plt.savefig("outputs/class_counts.png")
    plt.close()

    first_columns = X.iloc[:, :10].copy()              # correlation of the first 10 features with the target
    first_columns[target_column] = y
    plt.figure(figsize=(9, 7))
    sns.heatmap(first_columns.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

    # 6. Split into training and testing sets (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_split()
