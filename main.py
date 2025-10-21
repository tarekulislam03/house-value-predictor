import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing  import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def pipeline_main(num_attrib, cat_atrrib):
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standarization", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    final_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrib),
        ("cat", cat_pipeline, cat_atrrib)
    ])

    return final_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")

    housing["income_category"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                        labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_category"]):
        housing.loc[test_index].drop("income_category", axis=1).to_csv("input.csv")
        housing = housing.loc[train_index].drop("income_category", axis=1)

    housing_label = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)
    
    num_attrib = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attrib = ["ocean_proximity"]

    pipeline = pipeline_main(num_attrib, cat_attrib)

    housing_prepared_data = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared_data, housing_label)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model Trained!")

else:

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input = pd.read_csv("input.csv")
    tr_input = pipeline.transform(input)
    pred = model.predict(tr_input)

    input["median_house_value"] = pred
 
    input.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")

