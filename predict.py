import pandas as pd
import pickle


model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))


test = pd.read_csv("test.csv")


test_ids = test["Id"]


test = test.drop("Id", axis=1)


num_cols = test.select_dtypes(include=['int64', 'float64']).columns
cat_cols = test.select_dtypes(include=['object']).columns


test[num_cols] = test[num_cols].fillna(test[num_cols].median())
test[cat_cols] = test[cat_cols].fillna("None")


test = pd.get_dummies(test)


test = test.reindex(columns=features, fill_value=0)


pred = model.predict(test)


submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred
})

submission.to_csv("submission3.csv", index=False)

print("submission2.csv created successfully")