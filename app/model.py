import numpy as np
import pandas as pd
import pickle

# Import data
data = pd.read_csv("data_banknote_authentication.txt", names=["variance", "skewness", "curtosis", "entropy", "class"])

# Data preprocessing
cleaned_data = data[(data["variance"] >= 0) & (data["entropy"] >= 0)]

new_cleaned_data = data.copy()
new_cleaned_data.loc[new_cleaned_data["variance"] < 0, "variance"] = cleaned_data["variance"].median()
new_cleaned_data.loc[new_cleaned_data["entropy"] < 0, "entropy"] = cleaned_data["entropy"].median()

# Train Test Split
from sklearn.model_selection import train_test_split

X = new_cleaned_data.drop("class", axis=1)
Y = new_cleaned_data["class"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Model Building
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.001, gamma=100)
xgb.fit(X_train, Y_train)

# Save the model
with open("model.pkl", "wb") as model_file:
    pickle.dump(xgb, model_file)