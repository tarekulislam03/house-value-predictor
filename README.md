## 🏠 Housing Price Prediction Model (ML)

### Project Overview

In this project, we built a **complete machine learning pipeline** to predict **California housing prices** using various regression algorithms.

---

### Steps Involved

#### 1. **Data Loading & Preprocessing**

* Loaded the dataset: `housing.csv`
* Handled missing values carefully
* Applied **scaling** and **encoding** through a **custom pipeline**
* Ensured clean and consistent data for model training

#### 2. **Stratified Sampling**

* Used **Stratified Shuffle Split** to maintain the **income category distribution** between training and test sets.
* This ensured fair evaluation and representative data splitting.

---

### Model Training & Evaluation

We trained and compared multiple regression models:

* **Linear Regression**
* **Decision Tree Regressor**
* **Random Forest Regressor**
* **Gradient Boosting Regressor**
* **XGBoost Regressor**

Through **cross-validation**, the **Random Forest Regressor** achieved the **lowest RMSE** and demonstrated the **most stable performance** across folds.

---

### Final Model Pipeline

The final pipeline includes:

* Training the **Random Forest** model
* Saving the trained model using `joblib`
* Implementing an **if-else logic** to skip retraining if a saved model already exists
* Predicting `median_house_value` for new data from `input.csv`
* Saving the predictions to `output.csv`

---

### Key Features

* **End-to-end automated workflow** (from preprocessing to prediction)
* **Model persistence** for efficiency
* **Reusable structure** ready for **production deployment**

---

### 📈 Results

* **Best Model:** RandomForestRegressor
* **RMSE:** ~18,293
* **R² Score:** 0.9749 (Excellent fit — 97.5% variance explained)

---

### Conclusion

This project demonstrates how to design a **robust, production-ready regression pipeline** using Python and Scikit-learn.
The final Random Forest model provides **highly accurate** predictions and can easily be **integrated into real-world applications**.
