"""
main.py  (Execution lead)

Runs everything in order:  prepare data -> train model -> evaluate.
Run from the repo folder with:  python src/main.py
"""

from data_prep import load_and_split
from evaluate import evaluate
from train_model import train

print("STEP 1/3: preparing the data")
X_train, X_test, y_train, y_test = load_and_split()

print("\nSTEP 2/3: training the model")
model = train(X_train, y_train)

print("\nSTEP 3/3: evaluating the model")
evaluate(model, X_test, y_test)

print("\nDone. To predict individual matches run:  python src/predict.py")
