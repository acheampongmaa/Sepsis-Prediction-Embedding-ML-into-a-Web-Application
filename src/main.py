# Import key libraries and packages
from fastapi import FastAPI
import pickle
import uvicorn
from pydantic import BaseModel
import pandas as pd



# Function to load pickle file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data
    

# Load pickle file
ml_components = load_pickle('src/assets/sepsis.pkl') 

# Components in the pickle file
ml_model = ml_components['model']
full_pipeline = ml_components['pipeline']

# API base configuration
app = FastAPI()



# Setup the BaseModel
# class ModelInput(BaseModel):
#     Plasma_glucose: int
#     Blood_Work_Result_1: int
#     Blood_Pressure: int
#     Blood_Work_Result_2: int
#     Blood_Work_Result_3: int
#     Body_mass_index: float
#     Blood_Work_Result_4: float
#     Age: int

# data = pd.DataFrame({'Plasma glucose': ['Plasma_glucose'], 'Blood Work Result-1':	['Blood_Work_Result_1'],
#                          'Blood Pressure': ['Blood_Pressure'], 'Blood Work Result-2': ['Blood_Work_Result_2'],
#                         'Blood Work Result-3': ['Blood_Work_Result_3'], 'Body mass index': ['Body_mass_index'],
#                         'Blood Work Result-4':	['Blood_Work_Result_4'], 'Age': ['Age']})

@app.get('/Predict Sepsis')
async def predict(Plasma_glucose: int, Blood_Work_Result_1: int,
                  Blood_Pressure: int, Blood_Work_Result_2: int,
                    Blood_Work_Result_3: int, Body_mass_index: float, 
                    Blood_Work_Result_4: float,Age: int ):
    
    data = pd.DataFrame({'Plasma glucose': [Plasma_glucose], 'Blood Work Result-1':	[Blood_Work_Result_1],
                         'Blood Pressure': [Blood_Pressure], 'Blood Work Result-2': [Blood_Work_Result_2],
                        'Blood Work Result-3': [Blood_Work_Result_3], 'Body mass index': [Body_mass_index],
                        'Blood Work Result-4':	[Blood_Work_Result_4], 'Age': [Age]})
    
    data_prepared = full_pipeline.transform(data)

    model_output = ml_model.predict(data_prepared).tolist()

    return model_output

def make_prediction():

    output_pred = make_prediction()

    if output_pred == 0:
        output_pred = "Sepsis status is Negative"
    else:
        output_pred = "Sepsis status is Positive"
        
        return {"prediction": output_pred}



# def make_prediction(Plasma_glucose, Blood_Work_Result_1, Blood_Pressure, 
#                     Blood_Work_Result_2, Blood_Work_Result_3, Body_mass_index, 
#                     Blood_Work_Result_4, Age):
# data = pd.DataFrame({'Plasma glucose': [Plasma_glucose], 'Blood Work Result-1':	[Blood_Work_Result_1],
#                          'Blood Pressure': [Blood_Pressure], 'Blood Work Result-2': [Blood_Work_Result_2],
#                         'Blood Work Result-3': [Blood_Work_Result_3], 'Body mass index': [Body_mass_index],
#                         'Blood Work Result-4':	[Blood_Work_Result_4], 'Age': [Age]})
# def make_prediction():
    # data_prepared = full_pipeline.transform(data)

    # Make the prediction and return output
    # model_output = ml_model.predict(data_prepared).tolist()

    # if output_pred == 0:
    #     output_pred = "Sepsis status is Negative"
    # else:
    #     output_pred = "Sepsis status is Positive"

    # return model_output
    

# Endpoints
# @app.post("/Sepsis")
# async def predict(input: ModelInput):
#     output_pred = make_prediction(
#         Plasma_glucose=input.Plasma_glucose,
#         Blood_Work_Result_1=input.Blood_Work_Result_1,
#         Blood_Pressure=input.Blood_Pressure,
#         Blood_Work_Result_2=input.Blood_Work_Result_2,
#         Blood_Work_Result_3=input.Blood_Work_Result_3,
#         Body_mass_index=input.Body_mass_index,
#         Blood_Work_Result_4=input.Blood_Work_Result_4,
#         Age=input.Age
#         )

    # Labelling Model output
    # if output_pred == 0:
    #     output_pred = "Sepsis status is Negative"
    # else:
    #     output_pred = "Sepsis status is Positive"
    # #return output_pred
    # return {"prediction": output_pred,
    #         "input": input
    #         }

if __name__=='__main__':
    uvicorn.run('main:app', reload=True)
