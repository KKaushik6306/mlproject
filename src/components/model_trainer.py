import os
import sys
from dataclasses import dataclass
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,adjusted_rand_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model(self,train_array,test_array):
        try:
            logging.info('Split Training & Test Input Data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'XGB Classifier': XGBRegressor(),
                'CatBoosting': CatBoostRegressor(verbose=True),
                'AdaBoost Classifier': AdaBoostRegressor()

            }

            model_report :dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            model_df = pd.DataFrame(model_report)
            model_df.sort_values('Model Score',ascending=False)

            ##To get best model score from dict
            best_model_score = model_df['Model Score'][0]
            ##To get best model name from dict
            best_model_name = model_df['Model Name'][0]

            ##Raise Exception if model score is less than 60%
            if best_model_score <0.6:
                raise CustomException("No best model found!")
            logging.info('Best model found on trainging & test dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model_name
            )
            print(best_model_name)
            print(best_model_score)
            
        except Exception as e:
            raise CustomException(e,sys)
             









