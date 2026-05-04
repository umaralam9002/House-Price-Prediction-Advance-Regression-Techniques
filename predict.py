import pandas as pd
import pickle

# ==============================
# LOAD MODEL + FEATURES
# ==============================
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ==============================
# LOAD TEST DATA
# ==============================
test = pd.read_csv("test.csv")

# SAVE Id before processing
test_ids = test["Id"]

# ==============================
# PREPROCESSING
# ==============================
num_cols = test.select_dtypes(include=['int64','float64']).columns
cat_cols = test.select_dtypes(include=['object']).columns

test[num_cols] = test[num_cols].fillna(test[num_cols].median())
test[cat_cols] = test[cat_cols].fillna("None")

test = pd.get_dummies(test)

# ==============================
# FIX FEATURE MISMATCH
# ==============================
test = test.reindex(columns=features, fill_value=0)

# ==============================
# PREDICT
# ==============================
pred = model.predict(test)

# ==============================
# SAVE SUBMISSION
# ==============================
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred
})

submission.to_csv("submission.csv", index=False)

print(" submission.csv created successfully")