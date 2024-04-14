

from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

# data = {
#     'gender': ['Female'],
#     'race_ethnicity':['Group A'],
#     'parental_level_of_education':["associate's degree"],
#     'lunch': ['free/reduced'],
#     'test_preparation_course':[None],
#     'reading_score':[0],
#     'writing_score':[1]
# }

data=CustomData(
    gender= ['Female'],
    race_ethnicity=['Group A'],
    parental_level_of_education=["associate's degree"],
    lunch=['free/reduced'],
    test_preparation_course=[None],
    reading_score=[0],
    writing_score=[1]

        )
    
custom_data = CustomData     
pred_df=data.get_data_as_data_frame()
print(pred_df)
predict_pipeline = PredictPipeline()
results = predict_pipeline.predict(data)

