# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing  import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error, r2_score


# 1. Load the data
housing = pd.read_csv("housing.csv")

# 2. Creating stratified test set base on income category
housing["income_category"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_category"]):
    start_train_set = housing.loc[train_index].drop("income_category", axis=1)
    start_test_set = housing.loc[test_index].drop("income_category", axis=1)

#3. Copying training data
housing = start_train_set.copy()

#4. Separate features and label
housing_label = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

#5. separating numerical and categorical attributes
num_attrib = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attrib = ["ocean_proximity"]

#6. Creating Num Pipeline
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standarization", StandardScaler())
])

# 7. Creatine category pipeline
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Final pipeline
final_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attrib),
    ("cat", cat_pipeline, cat_attrib)
])

# 9. Fitting data
housing_prepared_data = final_pipeline.fit_transform(housing)

# housing_prepared_dataframe = pd.DataFrame(housing_prepared_data, columns=housing_prepared_data.columns, index=housing_prepared_data.index)

# 10. Trainings data to models

# # a. Linear Regresssion Model
# model_1 = LinearRegression()
# model_1.fit(housing_prepared_data, housing_label)
# pred_1 = model_1.predict(housing_prepared_data)
# model_1_err = root_mean_squared_error(housing_label, pred_1)
# print(model_1_err)

# b. Random Forest Regresssion Model
model_2 = RandomForestRegressor(random_state=42)
model_2.fit(housing_prepared_data, housing_label)
pred_2 = model_2.predict(housing_prepared_data)
model_2_err = root_mean_squared_error(housing_label, pred_2)
r2 = r2_score(housing_label, pred_2)
print(model_2_err, r2)

# # c. Decision Tree Regresssion Model
# model_3 = DecisionTreeRegressor(random_state=42)
# model_3.fit(housing_prepared_data, housing_label)
# pred_3 = model_3.predict(housing_prepared_data)
# model_3_err = root_mean_squared_error(housing_label, pred_3)
# print(model_3_err)

# # d. Gradient boosting Regression model
# model_4 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
# model_4.fit(housing_prepared_data, housing_label)
# pred_4 = model_4.predict(housing_prepared_data)
# model_4_err = root_mean_squared_error(housing_label, pred_4)
# print(model_4_err)

# # e. XGBoost Regressor
# model_5 = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
# model_5.fit(housing_prepared_data, housing_label)
# pred_5 = model_5.predict(housing_prepared_data)
# model_5_err = root_mean_squared_error(housing_label, pred_5)
# print(model_5_err)



importance = pd.Series(model_2.feature_importances_)
importance.nlargest(10).plot(kind='barh', figsize=(8,5))
plt.title("Top 10 Important Features")
plt.show()

