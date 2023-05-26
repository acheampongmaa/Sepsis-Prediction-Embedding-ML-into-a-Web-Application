from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# Function to load pickle file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data

# Load pickle file
ml_components = load_pickle('sepsis.pkl')

# Components in the pickle file
ml_model = ml_components['model']
full_pipeline = ml_components['pipeline']
num_pipeline = ml_components['num_pipe']

# Setup the BaseModel
class Input(BaseModel):
    Plasma_glucose: int
    Blood_Work_Result_1: int
    Blood_Pressure: int
    Blood_Work_Result_2: int
    Blood_Work_Result_3: int
    Body_mass_index: float
    Blood_Work_Result_4: float
    Age: int

def make_prediction(data):
    # Create a DataFrame from the input data
    df = pd.DataFrame(data)

    # Perform transformations using the full_pipeline
    df_prepared = full_pipeline.transform(df)

    # Make the prediction and return output
    model_output = ml_model.predict(df_prepared).tolist()
    return model_output

# Endpoint
@app.post("/predict")
def predict_sepsis(input_data:Input):
    # Convert the input data to a dictionary
    input_dict = input_data.dict()

    # Make the prediction
    prediction = make_prediction([input_dict])

    # Label the model output
    if prediction[0] == 0:
        output_pred = "Sepsis status is Negative"
    else:
        output_pred = "Sepsis status is Positive"

    return {"prediction": output_pred, "input": input_data.dict()}









