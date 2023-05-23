from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

#function to load pickle file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


#load  pickle file
ml_components= load_pickle('sepsis.pkl')

#components in pickle file
ml_model = ml_components['model']
full_pipeline = ml_components['pipeline']
num_pipeline =ml_components['num_pipe']

# Setup the BaseModel
class ModelInput(BaseModel):
    Plasma_glucose: int
    Blood_Work_Result_1: int
    Blood_Pressure: int
    Blood_Work_Result_2: int
    Blood_Work_Result_3: int
    Body_mass_index: float
    Blood_Work_Result_4: float
    Age: int

def make_prediction(Plasma_glucose, Blood_Work_Result_1, Blood_Pressure, Blood_Work_Result_2, Blood_Work_Result_3, Body_mass_index, Blood_Work_Result_4, Age):
    df = pd.DataFrame([['Plasma glucose', 'Blood Work Result-1', 'Blood Pressure', 'Blood Work Result-2', 'Blood Work Result-3', 'Body mass index', 'Blood Work Result-4', 'Age']],
                      columns = ['Plasma glucose', 'Blood Work Result-1', 'Blood Pressure', 'Blood Work Result-2', 'Blood Work Result-3', 'Body mass index', 'Blood Work Result-4', 'Age']
                      )
    
    # Perform transformations using the full_pipeline
    df_prepared= full_pipeline.transform(df)

    # Make the prediction and return output
    model_output = ml_model.predict(df_prepared).tolist()
    return model_output

# Endpoints
@app.post("/Sepsis Status")
async def predict(input: ModelInput):
    output_pred = make_prediction(
        Plasma_glucose= input.Plasma_glucose,
        Blood_Work_Result-1= input.Blood_Work_Result-1,
        Blood_Pressure=input.Blood_Pressure,
        Blood_Work_Result-2=input.Blood_Work_Result-2,
        Blood_Work_Result-3=input.Blood_Work_Result-3,
        Body_mass_index= input.Body_mass_index,
        Blood_Work_Result-4=input.Blood_Work_Result-4,
        Age=input.Age
       
        )

    # Labelling Model output
    if output_pred == 0:
        output_pred = "Sepsis status is Negative"
    else:
        output_pred = "Sepsis status is Positive"
    #return output_pred
    return {"prediction": output_pred,
            "input": input
            }

# Set the API to run
if __name__ == "__main__":
    uvicorn.run("api:app",
                reload=True
                )    
    
# from typing import Optional

# class Package(BaseModel):
#     name:str
#     number:str
#     description:Optional[str]=None

# app = FastAPI()

# @app.get('/')

# async def hello_world():
#     return {'hello':'world'}

# @app.post('/package/(priority)')
# async def make_package(priority: int, package:Package, value:bool):
#     return {'priority':priority, **package.dict(), 'value':value}


# @app.get('/component/(component_id)')    #path parameter
# async def get_component(component_id :int):
#     return {'component_id':component_id}

# @app.get('/component/')    #query parameter
# async def read_component(number :int, text:str):
#     return {'number':number, 'text':text}

