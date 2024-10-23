# Import key libraries and packages
import gradio as gr
import pickle
import pandas as pd
import os

print("Starting the Gradio app...")

# Define the directory paths and load the model
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_component = os.path.join(DIRPATH, "..", "src", "assets", "ml_sepsis.pkl")

# Function to load the pickle file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data

# Load the model components
ml_components = load_pickle(ml_component)
ml_model = ml_components['model']
pipeline_processing = ml_components['pipeline']

# Function to make predictions
def predict_sepsis(Plasma_glucose, Blood_Work_Result_1, Blood_Pressure, 
                   Blood_Work_Result_2, Blood_Work_Result_3, Body_mass_index, 
                   Blood_Work_Result_4, Age, Insurance):
    data = pd.DataFrame({
        'Plasma glucose': [Plasma_glucose],
        'Blood Work Result-1': [Blood_Work_Result_1],
        'Blood Pressure': [Blood_Pressure],
        'Blood Work Result-2': [Blood_Work_Result_2],
        'Blood Work Result-3': [Blood_Work_Result_3],
        'Body mass index': [Body_mass_index],
        'Blood Work Result-4': [Blood_Work_Result_4],
        'Age': [Age],
        'Insurance': [Insurance]
    })
    
    data_prepared = pipeline_processing.transform(data)
    model_output = ml_model.predict(data_prepared)[0]  # Get the first prediction

    return "Sepsis status is Positive" if model_output else "Sepsis status is Negative"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Sepsis Prediction Model")
    
    with gr.Row():
        Plasma_glucose = gr.Number(label="Plasma Glucose")
        Blood_Work_Result_1 = gr.Number(label="Blood Work Result 1")
        Blood_Pressure = gr.Number(label="Blood Pressure")
        Blood_Work_Result_2 = gr.Number(label="Blood Work Result 2")
        Blood_Work_Result_3 = gr.Number(label="Blood Work Result 3")
        Body_mass_index = gr.Number(label="Body Mass Index")
        Blood_Work_Result_4 = gr.Number(label="Blood Work Result 4")
        Age = gr.Number(label="Age")
        Insurance = gr.Number(label="Insurance")

    predict_button = gr.Button("Predict Sepsis")
    prediction_output = gr.Textbox(label="Prediction Result")

    # Set up the button click action
    predict_button.click(
        fn=predict_sepsis,
        inputs=[Plasma_glucose, Blood_Work_Result_1, Blood_Pressure, 
                Blood_Work_Result_2, Blood_Work_Result_3, Body_mass_index, 
                Blood_Work_Result_4, Age, Insurance],
        outputs=prediction_output,
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
