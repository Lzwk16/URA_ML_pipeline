import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the saved model
model = joblib.load("models/weights/weights.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input features from the request
        data = request.get_json(force=True)
        # Convert the data into a DataFrame
        input_df = pd.DataFrame([data])

        # Predict using the trained model
        prediction = model.predict(input_df)

        # Send the prediction as a response
        return jsonify({"prediction": np.round(prediction, 3).tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
