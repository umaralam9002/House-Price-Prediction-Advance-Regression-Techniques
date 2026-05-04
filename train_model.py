import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# LOAD DATA
# ==============================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ==============================
# TARGET
# ==============================
y = train["SalePrice"]
X = train.drop("SalePrice", axis=1)

# ==============================
# HANDLE MISSING VALUES
# ==============================
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
test[num_cols] = test[num_cols].fillna(test[num_cols].median())

X[cat_cols] = X[cat_cols].fillna("None")
test[cat_cols] = test[cat_cols].fillna("None")

# ==============================
# ENCODING
# ==============================
X = pd.get_dummies(X)
test = pd.get_dummies(test)

X, test = X.align(test, join='left', axis=1, fill_value=0)

# ==============================
# TRAIN
# ==============================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, pred))
r2 = r2_score(y_val, pred)

print("RMSE:", rmse)
print("R2:", r2)

# ==============================
# SAVE MODEL
# ==============================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("features.pkl", "wb"))

print(" Model and features saved successfully")

# print("Model saved as model.pkl")