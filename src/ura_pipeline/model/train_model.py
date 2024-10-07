import logging
import joblib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor


class ModelTrainer:
    def __init__(self, config_file):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.model = RandomForestRegressor
        self.scale_features = self.config_file["model_cfg"]["scale_features"]
        self.encode_features = self.config_file["model_cfg"]["encode_features"]

    def configure_column_transformer(self, rgr_transformer):
        
        #TODO: Finish up the configuration of the column transformer
        if rgr_transformer == "MLP":
            ct = ColumnTransformer(
            [   
                ('ss', StandardScaler(), self.scale_features),
                ('ohe', OneHotEncoder(), self.encode_features),
            ]
            remainder='passthrough', n_jobs=-1
        )
        else:
            ct = ColumnTransformer(
            [   
                ('ohe', OneHotEncoder(), self.encode_features),
            ]
            remainder='passthrough', n_jobs=-1
        )
        return ct
    def configure_training_pipeline(self, model_name):
        
        #TODO: Finish up the configuration of the training pipeline
        col_transformer = self.configure_column_transformer(model_name)
        if model_name == "MLP":
            mlp_pipeline = Pipeline(
                [
                    ('ct_MLP', col_transformer),
                    ('mlp', MLPRegressor()),    
                ]
            )
            return mlp_pipeline
        elif model_name == "RF":
            rf_pipeline = Pipeline(
                [
                    ('ct_RF', col_transformer),
                    ('rf', RandomForestRegressor()),
                ]
            )
            return rf_pipeline
        else:
            return None
    
    def train(self, X_train, y_train):
        
        
        #TODO: Code to take in the model pipeline and train using gridsearchCV
        # to search the best hyperparameters, and save, return the model
        pass
    
    def evaluate(self, X_test, y_test, trained_model):
        #TODO: code that takes in best model.pkl file and evaluates on test set
        y_pred = self.model.predict(X_test)
        pass
    
    def run(self):
        #TODO: Orchestrate configuration, training and evaluation
        pass