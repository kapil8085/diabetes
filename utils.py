import numpy as np
import pandas as pd
import pickle
import config
import sklearn

with open(config.MODEL_FILE_PATH,"rb") as f:
    model= pickle.load(f)

with open (config.SCALER_FILE_PATH,"rb") as f1:
    scaler=pickle.load(f1)    

def get_prediction(data):
    Glucose = eval(data["Glucose"])
    BloodPressure= eval(data["BloodPressure"])
    SkinThickness= eval(data["SkinThickness"])
    Insulin= eval(data["Insulin"])
    Bmi= eval(data["Bmi"])
    DiabetesPedigreeFunction= eval(data["DiabetesPedigreeFunction"])
    Age= eval(data["Age"])

    test_array= np.array([Glucose,BloodPressure,SkinThickness,Insulin,Bmi,DiabetesPedigreeFunction,Age],ndmin=2)
    scaler_test_array= scaler.transform(test_array)
    predict_value= model.predict_proba(scaler_test_array)[0,1]
    return f"{predict_value*100}%"
