"""
data_prep.py  (Documentation lead)

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------------- SETTINGS FOR OUR DATASET ----------------
DATA_FILE = "data/synthetic_basketball_dataset.csv"   # path of the CSV inside the repo
target_column = "Winner"                              # 1 = Team A wins, 0 = Team B wins
WIN_VALUE = None                                      
drop_columns = ["Match_ID"]                           
# -----------------------------------------------------------


def load_and_split():
    #Read 
    df = pd.read_csv(DATA_FILE)
    print("Dataset shape (rows, columns):", df.shape)
    print("Column names:", df.columns.tolist())

    #Understand 
    print("\nFirst five rows:")
    print(df.head())
    print("\nData types and non-null counts:")
    df.info()
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    #Preprocess
    df = df.drop_duplicates()
    df = df.drop(columns=drop_columns)                 
    df = df.dropna(subset=[target_column])         

    if WIN_VALUE is not None:                         
        df[target_column] = (df[target_column] == WIN_VALUE).astype(int)
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError("The target column has text values. Set WIN_VALUE in data_prep.py "
                         "(for example WIN_VALUE = 'Team A').")

    df = df.fillna(df.median(numeric_only=True))      
    df = pd.get_dummies(df, drop_first=True, dtype=int)  
    print("\nDataset shape after encoding:", df.shape)

    #Prepare features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Class counts:", y.value_counts().sort_index().to_dict())

    #Exploration graphs
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.countplot(x=y)
    plt.title("Number of matches per class")
    plt.xlabel("Class")
    plt.ylabel("Number of matches")
    plt.savefig("outputs/class_counts.png")
    plt.close()

    first_columns = X.iloc[:, :10].copy()              
    first_columns[target_column] = y
    plt.figure(figsize=(9, 7))
    sns.heatmap(first_columns.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

    
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
