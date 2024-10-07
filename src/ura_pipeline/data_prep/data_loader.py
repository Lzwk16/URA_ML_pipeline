import logging
import os
import sys

import pandas as pd
import psycopg2
import yaml
from dotenv import load_dotenv

from ura_pipeline.utils import load_config


# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
class DataLoader:
    """Fetch data from PostgreSQL database and save to CSV files."""
    
    def __init__(self, config_file):
        
        # Load environment variables from .env file
        load_dotenv('.env')
        self.config = load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.path = self.config['data']['data_path']
        
        # Set up the output directory for CSV files
        self.output_dir = os.path.join(os.getcwd(), self.path)
        self.logger.info(f"Output directory: {self.output_dir}")
        if not os.path.exists(self.output_dir):
            self.logger.info(f"Creating directory at: {self.output_dir}")
            os.makedirs(self.output_dir)
    
    def connect_db(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            conn_params = {
                'host': self.config['database']['host'],
                'port': self.config['database']['port'],
                'database': self.config['database']['name'],
                'user': os.getenv('POSTGRES_USER'),
                'password': os.getenv('POSTGRES_PASSWORD'),
            }
            conn = psycopg2.connect(**conn_params)
            self.logger.info("Connected to the PostgreSQL database!")
            return conn
        except psycopg2.Error as e:
            raise ConnectionError(f"Unable to connect to the database: {e}")

    def fetch_data_from_db(self, query):
        """Fetch data from the PostgreSQL database using the given SQL query."""
        conn = self.connect_db()
        try:
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def save_to_csv(self, df, filename):
        """Save the DataFrame to a CSV file."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved data to {filepath}")

    def run(self):
        """Main function to query data and save to CSV files."""
        # Define the SQL queries to fetch data from each table
        projects_query = "SELECT * FROM projects;"
        transactions_query = "SELECT * FROM private_property_transactions;"

        # Fetch project data
        projects_df = self.fetch_data_from_db(projects_query)
        if projects_df is not None and not projects_df.empty:
            self.save_to_csv(projects_df, 'projects.csv')

        # Fetch transaction data
        transactions_df = self.fetch_data_from_db(transactions_query)
        if transactions_df is not None and not transactions_df.empty:
            self.save_to_csv(transactions_df, 'transactions.csv')

if __name__ == "__main__":
    logging.info("Starting DataLoader...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    os.chdir(root)
    print(f"Current working directory: {os.getcwd()}")
    loader = DataLoader("conf/base/config.yaml")
    loader.run()
