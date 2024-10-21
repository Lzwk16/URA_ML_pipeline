# pylint: disable=no-name-in-module
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Property Prediction API",
    description="API for predicting property values based on various parameters.",
    version="1.0.0",
)

model = joblib.load("models/weights/weights.pkl")


# Define the input data model
class InputData(BaseModel):
    area: float
    type_of_sale: int
    district: int
    market_segment: str
    x_coordinate: float
    y_coordinate: float
    year: int
    month: str
    remaining_lease: int
    middle_story: int


@app.post(
    "/predict", summary="Make a prediction", response_description="The predicted value"
)
async def predict(input_data: InputData):
    """
    Make a prediction based on input features.

    - **area**: Area of the property
    - **type_of_sale**: Type of sale (integer)
    - **district**: District number
    - **market_segment**: Market segment (string)
    - **x_coordinate**: X coordinate of the property
    - **y_coordinate**: Y coordinate of the property
    - **year**: Year of the transaction
    - **month**: Month of the transaction (string, e.g., '01')
    - **remaining_lease**: Remaining lease duration (in years)
    - **middle_story**: Middle story of the property
    """
    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Predict using the trained model
        prediction = model.predict(input_df)

        # Return prediction result
        return {"prediction": np.round(prediction, 3).tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
