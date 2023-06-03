# Import key libraries and packages
from fastapi import FastAPI
import pickle
import uvicorn
from pydantic import BaseModel
import pandas as pd
import os
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define directory paths
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_component= os.path.join(DIRPATH, "..", "src", "assets", "ml_sepsis.pkl")

# Function to load pickle file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data
    

# Load pickle file
ml_components = load_pickle(ml_component) 

# Components in the pickle file
ml_model = ml_components['model']
pipeline_processing = ml_components['pipeline']

# API base configuration
app = FastAPI()


@app.get('/Predict_Sepsis')
async def predict(Plasma_glucose: int, Blood_Work_Result_1: int,
                  Blood_Pressure: int, Blood_Work_Result_2: int,
                    Blood_Work_Result_3: int, Body_mass_index: float, 
                    Blood_Work_Result_4: float,Age: int, Insurance:float):
    
    data = pd.DataFrame({'Plasma glucose': [Plasma_glucose], 'Blood Work Result-1':	[Blood_Work_Result_1],
                         'Blood Pressure': [Blood_Pressure], 'Blood Work Result-2': [Blood_Work_Result_2],
                        'Blood Work Result-3': [Blood_Work_Result_3], 'Body mass index': [Body_mass_index],
                        'Blood Work Result-4':	[Blood_Work_Result_4], 'Age': [Age], 'Insurance':[Insurance]})
    
    data_prepared = pipeline_processing.transform(data)

    model_output = ml_model.predict(data_prepared).tolist()

    prediction = make_prediction(model_output)

    return prediction
    
    

def make_prediction(data_prepared):

    output_pred = data_prepared

    if output_pred == 0:
        output_pred = "Sepsis status is Negative"
    else:
        output_pred = "Sepsis status is Positive"
        
    return output_pred

if __name__=='__main__':
    uvicorn.run('main:app', reload=True)

   
    




    


