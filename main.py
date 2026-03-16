import joblib
import pandas as pd
from fastapi import FastAPI
from schema.schema import DiabetesData

app=FastAPI()

model=joblib.load("models/model.pkl")

@app.get("/")
def home():

    return {"message": "Diabetes Prediction API Running"}

@app.post("/predict")
def predict(data: DiabetesData):
    input_data=pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Have Diabetes"
    else:
        result = "No Diabetes"

    return {"prediction": result}