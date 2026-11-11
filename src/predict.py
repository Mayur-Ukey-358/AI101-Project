import pandas as pd

from data_prep import load_and_split
from train_model import train

class_names = {0: "Team B wins", 1: "Team A wins"}

NEW_MATCH = {
    "A_Off_Rating": 118, "A_Def_Rating": 105, "A_Net_Rating": 13,
    "B_Off_Rating": 104, "B_Def_Rating": 110, "B_Net_Rating": -6,
    "Home_Team_B": 0,
}

X_train, X_test, y_train, y_test = load_and_split()
model = train(X_train, y_train)

print("\n===== PREDICTING UNSEEN MATCHES =====")
for index in [0, 5, 10, 15, 20]:                       
   
    sample = X_test.iloc[[index]]

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)[0][1]

    predicted_class = prediction[0]
    actual_class = y_test.iloc[index]
    result = "correct" if predicted_class == actual_class else "incorrect"

    print(f"Test match {index}: predicted = {class_names[predicted_class]} "
          f"(probability of class 1 = {probability:.2f}), "
          f"actual = {class_names[actual_class]} -> {result}")

if NEW_MATCH:
    values = X_train.median()                         
    for column, value in NEW_MATCH.items():
        values[column] = value
    new_match = pd.DataFrame([values])
    new_prediction = model.predict(new_match)[0]
    new_probability = model.predict_proba(new_match)[0][1]
    print(f"\nYour own match: predicted = {class_names[new_prediction]} "
          f"(probability of class 1 = {new_probability:.2f})")
