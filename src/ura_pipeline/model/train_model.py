import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pylint: disable=no-name-in-module
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ura_pipeline.utils import load_config


class ModelTrainer:
    def __init__(self, config_file):
        self.config_file = load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.scale_features = self.config_file["model_cfg"]["scale_features"]
        self.encode_features = self.config_file["model_cfg"]["encode_features"]
        self.model_params_dict = self.config_file["model_cfg"]
        self.seed = self.config_file["model_cfg"]["random_state"]
        self.cv = self.config_file["model_cfg"]["cv"]
        self.scoring = self.config_file["model_cfg"]["scoring"]
        self.model_save_path = self.config_file["model_cfg"]["model_save_path"]

    def _configure_column_transformer(self, rgr_transformer):
        """
        Configure ColumnTransformer for the given regressor.

        Parameters
        ----------
        rgr_transformer : str
            Regressor type, either "MLP" or "RF" or "XGB".

        Returns
        -------
        ct : ColumnTransformer
            Configured ColumnTransformer.
        """
        if rgr_transformer == "MLP":
            ct = ColumnTransformer(
                [
                    ("ss", StandardScaler(), self.scale_features),
                    (
                        "ohe",
                        OneHotEncoder(drop="first", handle_unknown="ignore"),
                        self.encode_features,
                    ),
                ],
                remainder="passthrough",
                n_jobs=-1,
            )
        else:
            ct = ColumnTransformer(
                [
                    (
                        "ohe",
                        OneHotEncoder(drop="first", handle_unknown="ignore"),
                        self.encode_features,
                    ),
                ],
                remainder="passthrough",
                n_jobs=-1,
            )
        return ct

    def _configure_training_pipeline(self, model_name):
        """
        Configure the training pipeline according to the model name.

        Parameters
        ----------
        model_name : str
            The name of the model to be trained. Can be "MLP" or "RF".

        Returns
        -------
        pipeline : Pipeline
            The configured pipeline for the specified model.
        """
        col_transformer = self._configure_column_transformer(model_name)
        if model_name == "MLP":
            mlp_pipeline = Pipeline(
                [
                    ("ct_MLP", col_transformer),
                    ("mlp", MLPRegressor(random_state=self.seed)),
                ]
            )
            return mlp_pipeline
        elif model_name == "RF":
            rf_pipeline = Pipeline(
                [
                    ("ct_RF", col_transformer),
                    ("rf", RandomForestRegressor(random_state=self.seed)),
                ]
            )
            return rf_pipeline
        else:
            return None

    def train(self, X_train, y_train, model_name):
        """
        Train the model with GridSearchCV.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data features.
        y_train : pd.Series
            The training data target.
        model_name : str
            The name of the model to be trained. Can be "MLP" or "RF".

        Returns
        -------
        best_model : Pipeline
            The best model with the best hyperparameters.
        training_rmse : float
            The root mean squared error of the best model on the training set.
        """
        pipeline = self._configure_training_pipeline(model_name)

        if model_name == "MLP":
            param_grid = self.model_params_dict["multi_layer_perceptron"]["params"]
        elif model_name == "RF":
            param_grid = self.model_params_dict["random_forest"]["params"]
        else:
            self.logger.error(f"Model {model_name} is not configured correctly.")
            return None

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1
        )

        # Train the model with GridSearchCV
        self.logger.info(f"Training the {model_name} model...")
        grid_search.fit(X_train, y_train)

        # Best model and parameters
        best_model = grid_search.best_estimator_
        trainig_rmse = -grid_search.best_score_
        self.logger.info(
            f"Best hyperparameters for {model_name}: {grid_search.best_params_}"
        )

        return best_model, trainig_rmse

    def evaluate(self, X_train, y_train, X_test, y_test, trained_model):
        """
        Evaluate the best model and save it to file.

        This method takes in the best model and evaluates it on the test set.
        The model is then saved to file.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.
        trained_model : object
            Best model from GridSearchCV.

        Returns
        -------
        None
        """
        self.logger.info("Evaluating the model...")

        # Fit and make predictions
        trained_model.fit(X_train, y_train)
        y_pred = trained_model.predict(X_test)

        # Calculate evaluation metrics for best model
        rmse = root_mean_squared_error(y_test, y_pred)

        # Save the best model
        model_weights_dir = os.path.dirname(self.model_save_path)
        if not os.path.exists(model_weights_dir):
            os.makedirs(model_weights_dir)

        joblib.dump(trained_model, self.model_save_path)
        self.logger.info(f"Best model saved as {self.model_save_path}")
        self.logger.info(f"Best model RMSE: {rmse}")

    def run(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models and evaluate them on the test set.

        This method takes in the pre-processed training and test data and trains
        multiple models based on the configurations in the config file. The best
        model is selected based on the lowest RMSE and evaluated on the test set.

        Parameters
        ----------
        X_train : pd.DataFrame
            The pre-processed training data features.
        y_train : pd.Series
            The pre-processed training data target.
        X_test : pd.DataFrame
            The pre-processed test data features.
        y_test : pd.Series
            The pre-processed test data target.
        """
        models_config = {
            "RF": self.config_file["model_cfg"]["random_forest"]["name"],
            "MLP": self.config_file["model_cfg"]["multi_layer_perceptron"]["name"],
        }

        best_model_overall = None
        lowest_rmse = float("inf")  # Initialize with a large number
        best_model_name = None

        # Loop over models and train them
        for model_name, config_name in models_config.items():
            if config_name == model_name:
                self.logger.info(f"Running training for {model_name} model.")
                best_model, best_rmse = self.train(X_train, y_train, model_name)

                # Update the best model based on RMSE
                if best_rmse < lowest_rmse:
                    lowest_rmse = best_rmse
                    best_model_overall = best_model
                    best_model_name = model_name

            else:
                self.logger.warning(f"No valid configuration found for {model_name}.")

        # Evaluate the overall best model
        if best_model_overall:
            self.logger.info(
                f"Evaluating the best model ({best_model_name}) on the test set..."
            )
            self.evaluate(X_train, y_train, X_test, y_test, best_model_overall)
