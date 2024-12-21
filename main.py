from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model and encoders
knn_model = joblib.load('knn_model.pkl')
category_encoder = joblib.load('category_encoder.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define the input data schema
class InputData(BaseModel):
    Age: int
    Gender: int  # 0 = Female, 1 = Male
    Score: int

@app.get("/")
async def root():
    return {"message": "Welcome to the Internet Addiction Prediction API"}

@app.post("/predict/")
async def predict(input_data: InputData):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction using the KNN model
    prediction = knn_model.predict(input_df)
    
    # Decode the prediction to the original category label
    predicted_category = category_encoder.inverse_transform(prediction)

    # Return the prediction result
    return {
        "prediction": predicted_category[0]
    }
