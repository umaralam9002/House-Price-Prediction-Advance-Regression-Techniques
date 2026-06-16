# House Price Prediction (Ames Dataset)

## Project Overview

This project predicts house prices using machine learning techniques on the Ames Housing dataset.
The goal is to build a regression model that can accurately estimate property prices based on various features such as area, quality, and location.

---

## Dataset

* Source: Kaggle (House Prices - Advanced Regression Techniques)
* Files used:

  * `train.csv`
  * `test.csv`
  * `data_description.txt`

---

## Technologies Used

* Python 
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## Workflow

### 1. Data Preprocessing

* Handled missing values:

  * Numerical → median
  * Categorical → "None"
* Removed irrelevant inconsistencies

### 2. Feature Engineering

* Converted categorical variables using one-hot encoding
* Aligned train and test features
* (Optional) Created new features like total area

### 3. Model Training

* Used regression model:

  * Ridge Regression (baseline)
* Split dataset into training and validation sets

### 4. Evaluation Metrics

* **RMSE (Root Mean Squared Error)** → measures prediction error
* **R² Score** → measures model accuracy

---

## Results

* RMSE: ~25,000
* R² Score: ~0.85

---

## Project Structure

```
house-price-project/
│
├── train.csv
├── test.csv
├── data_description.txt
│
├── notebook.ipynb
├── train_model.py
├── predict.py
│
├── model.pkl
├── features.pkl
└── submission.csv
```

---

##  How to Run

### 1. Install dependencies

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Train the model

```
python train_model.py
```

### 3. Generate predictions

```
python predict.py
```

---

## Key Learnings

* Data preprocessing is critical for model performance
* Feature alignment between train and test is essential
* RMSE and R² are important evaluation metrics
* Simple models can perform well with proper preprocessing

---

## Future Improvements

* Apply log transformation to reduce skewness
* Use advanced models (Gradient Boosting, XGBoost)
* Perform hyperparameter tuning
* Build a Streamlit web app for deployment

---

## Author

Muhammad Umar Alam
Sehar Ajmal
Talha Khalid

---

## Conclusion

This project demonstrates a complete machine learning pipeline from data preprocessing to model evaluation and prediction. It highlights the importance of clean data and proper feature handling in achieving good performance.
