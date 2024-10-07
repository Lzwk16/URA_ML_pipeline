import logging
import logging.config
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.ura_pipeline.utils import count_null_values, load_config


class DataProcessor:
    def __init__(self, config_file, project_data, transaction_data, cfg=None):
        # load_logging_config(cfg)
        self.config = load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.project_data = project_data
        self.transaction_data = transaction_data
        self.target_col = self.config["data"]["target_col"]
        self.primary_key = self.config["data"]["primary_key"]
        self.foreign_key = self.config["data"]["foreign_key"]
        self.merge_strategy = self.config["data"]["merge_strategy"]
        self.property_col = self.config["data"]["property"]["column_name"]
        self.property_type = self.config["data"]["property"]["filter_property_type"]
        self.impute_cols = self.config["data"]["impute_cols"]
        self.drop_cols = self.config["data"]["drop_cols"]
        self.knn_n_neighbors = self.config["data"]["KNNIMPUTER"]
        self.current_year = datetime.now().year

    def _clean_ura_dataset(
        self,
        data,
    ):
        """
        Clean the URA dataset by imputing missing values and filtering the dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            The URA dataset to be cleaned.

        Returns
        -------
        pd.DataFrame
            The cleaned URA dataset.
        """
        if data.isnull().sum().any() != 0:
            self.logger.info(f"Imputing missing values in the dataset.")
            filter_data = data[self.config["data"]["impute_cols"]]
            non_null_data = filter_data[filter_data[self.impute_cols].notnull()]
            knn_imputer = KNNImputer(n_neighbors=self.knn_n_neighbors)
            self.logger.info("Initialized KNNImputer.")
            self.logger.info(
                f"Using {self.knn_n_neighbors} N NEAREST NEIGHBOURS for KNN Imputer."
            )
            knn_imputer.fit(non_null_data)
            imputed_data = knn_imputer.transform(filter_data)
            mask_x = data["x_coordinate"].isnull()
            mask_y = data["y_coordinate"].isnull()

            # Assign the imputed coordinates back to data
            data.loc[mask_x, "x_coordinate"] = imputed_data[mask_x, 0]
            data.loc[mask_y, "y_coordinate"] = imputed_data[mask_y, 1]
            self.logger.info("Replaced NaN values with imputed values.")

        data = data[data[self.property_col] == self.property_type]

        data = data.drop_duplicates()

        # Count null values after cleaning
        null_counts = count_null_values(data)

        # Raise an error if there are any null values
        if null_counts.sum() > 0:
            self.logger.warning(
                f"Null values remaining after cleaning: {null_counts[null_counts > 0]}"
            )

        return data

    def _feature_engineering(self, data):
        """
        Perform feature engineering on the given dataset.

        The following features are created:
        - year: The year of the transaction date.
        - month: The month of the transaction date.
        - remaining_lease: The remaining lease years, calculated as
            99 - (current_year - lease_start_year).
        - middle_story: The middle story of the floor range, calculated as
            (floor_range[0] + floor_range[-1]) / 2.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to perform feature engineering on.

        Returns
        -------
        pd.DataFrame
            The dataset with the new features added.
        """
        data = data.copy()
        data[["year", "month"]] = data["transaction_date"].apply(
            lambda x: pd.Series(self._extract_month_year(x))
        )
        lease_start_years = data["tenure"].apply(self._extract_lease_start_year)
        data.loc[:, "remaining_lease"] = 99 - (self.current_year - lease_start_years)
        data.loc[:, "middle_story"] = data["floor_range"].apply(
            self._extract_middle_story
        )

        return data

    def _extract_month_year(self, transaction_date):
        """
        Extracts the year and month from a transaction date string in the format "YYYY-MM".

        Parameters
        ----------
        transaction_date : str
            The transaction date string in the format "YYYY-MM".

        Returns
        -------
        year : str
            The year of the transaction date.
        month : str
            The month of the transaction date.
        """
        year, month = transaction_date.split("-")
        return year, month

    def _extract_lease_start_year(self, tenure):
        """
        Extracts the lease start year from the tenure string.

        If the tenure string does not contain "commencing from", the current year is returned.

        Parameters
        ----------
        tenure : str
            Tenure string.

        Returns
        -------
        int
            Lease start year.

        """
        if "commencing from" in tenure:
            return int(tenure.split("commencing from")[-1].strip())
        return self.current_year

    def _extract_middle_story(self, floor_range):
        """
        Extracts the middle story of a floor range.

        The middle story is defined as the average of the lower and upper bounds of the floor range.

        Args:
            floor_range (str): The floor range, e.g. "01-05".

        Returns:
            int: The middle story of the floor range.
        """
        lower, upper = map(int, floor_range.split("-"))
        middle = (lower + upper) // 2
        return middle

    def _data_splitter(self, data):
        """
        Split the data into training and test sets using Scikit-Learn's train_test_split
        function.

        Args:
            data (pd.DataFrame): The dataframe containing the data to be split.

        Returns:
            tuple: A tuple of two dataframes - the training dataframe and the test dataframe.
        """
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        return train, test

    def _merge_data(self, transaction_df, project_df):
        """
        Merge the transaction and project dataframes on the project_id column.

        Args:
            transaction_df (pd.DataFrame): The dataframe containing transaction data.
            project_df (pd.DataFrame): The dataframe containing project data.

        Returns:
            pd.DataFrame: The merged dataframe of transactions and project data.
        """
        merged_df = pd.merge(
            left=transaction_df,
            right=project_df,
            how=self.merge_strategy,
            left_on=self.foreign_key,
            right_on=self.primary_key,
        )

        # Drop the columns in place
        columns_to_drop = [
            col for col in merged_df.columns if col.endswith("_x") or col.endswith("_y")
        ]
        merged_df.drop(columns=columns_to_drop, inplace=True)
        return merged_df

    def run(self):
        """
        Runs the entire data processing pipeline.

        This method takes in project and transaction data and outputs the final
        X and y datasets for both the train and test sets.

        The steps involved in the pipeline are:

        1. Merging project and transaction data.
        2. Data splitting into train and test sets.
        3. Data cleaning and feature engineering.
        4. Splitting data into X and y variables for both train and test.
        5. Returning the final X and y sets.

        Returns
        -------
        tuple
            A tuple of four pandas DataFrames: X_train, y_train, X_test, y_test.
        """
        self.logger.info("Initializing DataProcessor ...")

        # Step 1: Merging project and transaction data
        self.logger.info("Merging project and transaction data...")
        merged_data = self._merge_data(self.transaction_data, self.project_data)

        self.logger.info(
            f"Market segment nulls in merged data: {merged_data['market_segment'].isnull().sum()}"
        )

        # Step 2: Data Splitting
        self.logger.info("Splitting merged data into train and test sets.")
        train_data, test_data = self._data_splitter(merged_data)

        self.logger.info("Data splitting complete.")

        # Step 3: Data Cleaning and Feature Engineering
        self.logger.info("Performing data cleaning and feature engineering...")

        datasets = {"train": train_data, "test": test_data}

        cleaned_datasets = {}

        for dataset_type, dataset in datasets.items():
            # Cleaning data
            self.logger.info(f"Cleaning {dataset_type} data...")
            cleaned_data = self._clean_ura_dataset(dataset)

            # Feature engineering
            self.logger.info(
                f"Performing feature engineering on {dataset_type} data..."
            )
            engineered_data = self._feature_engineering(cleaned_data)

            engineered_data = engineered_data.drop(columns=self.drop_cols)

            cleaned_datasets[dataset_type] = engineered_data

        self.logger.info("Data cleaning and feature engineering completed.")

        # Step 4: Split into X and y variables for both train and test
        self.logger.info("Splitting data into X and y variables.")

        final_data = {}

        for dataset_type, dataset in cleaned_datasets.items():
            X = dataset.drop(columns=self.target_col)
            y = dataset[self.target_col]
            final_data[dataset_type] = (X, y)

        X_train, y_train = final_data["train"]
        X_test, y_test = final_data["test"]

        self.logger.info("DataProcessor completed.")

        # Step 5: Return the final X and y sets
        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Used for test run and debugging
    logging.info("Starting DataLoader...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    os.chdir(root)
    print(f"Current working directory: {os.getcwd()}")
    project_df = pd.read_csv("data/projects.csv")
    transaction_df = pd.read_csv("data/transactions.csv")
    loader = DataProcessor(
        config_file="conf/base/config.yaml",
        project_data=project_df,
        transaction_data=transaction_df,
        cfg="conf/base/logging.yaml",
    )
    X_train, y_train, X_test, y_test = loader.run()
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
