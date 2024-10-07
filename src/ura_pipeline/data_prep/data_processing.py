import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ura_pipeline.utils import count_null_values, load_config


class DataProcessor:
    def __init__(self, config_file, project_data, transaction_data):
        self.config = load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.project_data = project_data
        self.transaction_data = transaction_data
        self.target_col = self.config["data"]["target_col"]
        self.property_col = self.config["data"]["property"]["column_name"]
        self.property_type = self.config["data"]["property"]["filter_property_type"]
        self.impute_cols = self.config["data"]["impute_cols"]
        self.drop_cols = self.config["data"]["drop_cols"]
        self.knn_n_neighbors = self.config["data"]["KNNIMPUTER"]
        self.current_year = datetime.now().year
    
    def _clean_ura_dataset(self, data, is_project_data=False, is_transaction_data=False):
        if is_project_data and data.isnull().sum().any() != 0:
            filter_data = data[self.config["data"]["impute_cols"]]
            non_null_data = filter_data[filter_data[self.impute_cols].notnull()]
            knn_imputer = KNNImputer(n_neighbors=self.knn_n_neighbors)
            self.logger.info("Initialized KNNImputer.")
            self.logger.info(
                f"Using {self.knn_n_neighbors} N NEAREST NEIGHBOURS for KNN Imputer."
            )
            knn_imputer.fit(non_null_data)
            imputed_data = knn_imputer.transform(filter_data)
            mask_x = data['x_coordinate'].isnull()
            mask_y = data['y_coordinate'].isnull()

            # Assign the imputed coordinates back to data
            data.loc[mask_x, 'x_coordinate'] = imputed_data[mask_x, 0]
            data.loc[mask_y, 'y_coordinate'] = imputed_data[mask_y, 1]
            self.logger.info("Replaced NaN values with imputed values.")
        
        if is_transaction_data:
            data = data[data[self.property_col] == self.property_type]
        
        data.drop_duplicates(inplace=True)
        
        # Count null values after cleaning
        null_counts = count_null_values(data)

        # Raise an error if there are any null values
        if null_counts.sum() > 0:
            self.logger.warning(f"Null values remaining after cleaning: {null_counts[null_counts > 0]}")
        
        return data
    
    def _feature_engineering(self, data):
        data[['year', 'month']] = data['transaction_date'].apply(
            lambda x: pd.Series(self._extract_month_year(x))
        )
        lease_start_years = data['tenure'].apply(self._extract_lease_start_year)
        data['remaining_lease'] = 99 - (self.current_year - lease_start_years)
        data['middle_story'] = data['floor_range'].apply(self._extract_middle_story)
        return data
    
    
    def _extract_month_year(self, transaction_date):
        year, month = transaction_date.split('-')
        return year, month
    
    
    def _extract_lease_start_year(self, tenure):
        if 'commencing from' in tenure:
            return int(tenure.split('commencing from')[-1].strip())
        return self.current_year
    
    def _extract_middle_story(self, floor_range):
        lower, upper = map(int, floor_range.split('-'))
        middle = (lower + upper) // 2
        return middle
    
    def _data_splitter(self, data):
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        return train, test
    
    def _merge_final_data(self, transaction_df, project_df):
        merged_df = pd.merge(
            left=transaction_df,
            right=project_df,
            how='left', 
            left_on='project_id', 
            right_on='id'
        )
        
        # Drop the columns in place
        columns_to_drop = [
            col for col in merged_df.columns 
            if col.endswith("_x") or col.endswith("_y")
        ]
        merged_df.drop(columns=columns_to_drop, inplace=True)
        return merged_df
    
    def _categorical_encode(self, train_data, test_data, categorical_cols):
        df_train_copy = train_data.copy()
        df_test_copy = test_data.copy()
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore')

        # Fit and transform the categorical columns
        encoded_train_data = encoder.fit_transform(df_train_copy[categorical_cols]).toarray()
        encoded_test_data = encoder.transform(df_test_copy[categorical_cols]).toarray()

        # Create DataFrames from the encoded columns
        encoded_train_df = pd.DataFrame(
            encoded_train_data, 
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        encoded_test_df = pd.DataFrame(
            encoded_test_data, 
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        
        processed_train_df = pd.concat(
            [df_train_copy.drop(columns=categorical_cols), encoded_train_df],
            axis=1
        )

        processed_test_df = pd.concat(
            [df_test_copy.drop(columns=categorical_cols), encoded_test_df],
            axis=1
        )
        
        return processed_train_df, processed_test_df
    
    def run(self):
        # Step 1: Data Splitting
        self.logger.info("Initializing DataProcessor ...")
        self.logger.info("Splitting data into train and test sets.")
        project_train, project_test = self._data_splitter(self.project_data)
        transaction_train, transaction_test = self._data_splitter(self.transaction_data)
        
        self.logger.info("Data splitting complete.")
        datasets = [
            (project_train, project_test, True, False),  # True means project data (KNN imputation needed)
            (transaction_train, transaction_test, False, True)  # False means not project data (no KNN imputation)
        ]
        self.logger.info("Performing data cleaning...")
        # Step 2: Data Cleaning (KNN Imputation for project data)
        cleaned_datasets = []
        for (train_data, test_data, 
            is_project_data, is_transaction_data) in datasets:
            
            cleaned_train = self._clean_ura_dataset(
                train_data, 
                is_project_data=is_project_data, 
                is_transaction_data=is_transaction_data
            )
            
            cleaned_test = self._clean_ura_dataset(
                test_data, 
                is_project_data=is_project_data, 
                is_transaction_data=is_transaction_data
            )
            
            cleaned_datasets.append((cleaned_train, cleaned_test))
        self.logger.info("Cleaning completed...")
        
        project_train, project_test = cleaned_datasets[0]
        transaction_train, transaction_test = cleaned_datasets[1]
        
        
        # Step 3: Feature Engineering (with loop)
        self.logger.info("Performing feature engineering...")
        feature_engineered_datasets = []
        for train_data, test_data in [(transaction_train, transaction_test)]:
            engineered_train = self._feature_engineering(train_data)
            engineered_test = self._feature_engineering(test_data)
            feature_engineered_datasets.append((engineered_train, engineered_test))
        self.logger.info("Feature engineering completed...")
        
        transaction_train, transaction_test = feature_engineered_datasets[0]
        
        # Step 4: Merging project and transaction datasets (with loop)
        
        self.logger.info("Combining datasets...")
        datasets = [
            (transaction_train, project_train),
            (transaction_test, project_test)
        ]

        merged_datasets = []  # Initialize list to hold merged datasets

        for transaction_data, project_data in datasets:
            self.logger.info(f"transaction_data cols: {transaction_data.columns}")
            self.logger.info(f"project_data cols: {project_data.columns}")

            # Merge the datasets
            merged_data = self._merge_final_data(transaction_data, project_data)
            
            # Log the list of columns after merging
            self.logger.info(f"List of columns after merging: {merged_data.columns}")
            
            # Drop unnecessary columns
            merged_data.drop(columns=self.drop_cols, inplace=True)
            
            # Append the merged dataset to the list
            merged_datasets.append(merged_data)

        self.logger.info("Merging completed!")
        train_data, test_data = merged_datasets[0], merged_datasets[1]
        
        # Step 5: Split into X and y variables
        
        self.logger.info("Performing encoding...")
        self.logger.info(f"training data shape before encoding: {train_data.shape}")
        self.logger.info(f"test data shape before encoding: {train_data.shape}")
        X_train = train_data.drop(columns=self.target_col)  
        y_train = train_data[self.target_col]
        
        X_test = test_data.drop(columns=self.target_col)
        y_test = test_data[self.target_col]
        
        # Step 6: Categorical Encoding
        categorical_cols = X_train.select_dtypes(include=['object']).columns 
        X_train_encoded, X_test_encoded = self._categorical_encode(X_train, X_test, categorical_cols)
        self.logger.info("Encoding completed...")
        self.logger.info(f"training data shape after encoding: {X_train_encoded.shape}")
        self.logger.info(f"test data shape after encoding: {X_test_encoded.shape}")
        self.logger.info(f"columns in train after encode: {X_train_encoded.columns}")
        self.logger.info(f"columns in test after encode: {X_test_encoded.columns}")
        self.logger.info("DataProcessor completed!")
        # Step 7: Return the final X and y sets
        return X_train_encoded, y_train, X_test_encoded, y_test
    

if __name__ == "__main__":
    logging.info("Starting DataLoader...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    os.chdir(root)
    print(f"Current working directory: {os.getcwd()}")
    project_df = pd.read_csv('data/projects.csv')
    transaction_df = pd.read_csv('data/transactions.csv')
    loader = DataProcessor(
        config_file="conf/base/config.yaml", 
        project_data=project_df, 
        transaction_data=transaction_df
    )
    X_train, y_train, X_test, y_test = loader.run()