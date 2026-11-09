from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(X_train, y_train):
    # Pipeline: first scale the features, then apply Logistic Regression
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train the model using the training data
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    print("Logistic Regression model trained successfully!")
    print(f"Accuracy on the training data: {train_accuracy:.3f}")
    return model


if __name__ == "__main__":
    from data_prep import load_and_split
    X_train, X_test, y_train, y_test = load_and_split()
    train(X_train, y_train)
