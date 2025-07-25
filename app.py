from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import Dict

app = FastAPI()

class HouseFeatures(BaseModel):
    input_features: Dict[str, float]
    
@app.post("/predict")
def predict(features: HouseFeatures):

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open("regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
   
    input_df = pd.DataFrame([features.input_features])
    print("Input DataFrame:", input_df)
    scaled_features = scaler.transform(input_df)


    # Make prediction
    prediction = model.predict(scaled_features)

    return {"prediction": prediction[0]}