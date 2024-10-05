import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Database credentials
db_name = os.getenv('POSTGRES_DB')
user_name = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')

# Set up the output directory for CSV files
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

def fetch_data_from_db(query):
    """Fetch data from the PostgreSQL database using the given SQL query."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5433",  # Ensure this matches your Docker mapping
            database=db_name,
            user=user_name,
            password=password
        )
        print("Connected to the PostgreSQL database!")
    except psycopg2.Error as e:
        raise ConnectionError(f"Unable to connect to the database: {e}")

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def save_to_csv(df, filename):
    """Save the DataFrame to a CSV file."""
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved data to {filepath}")


def run():
    """Main function to query data and save to CSV files."""
    # Define the SQL queries to fetch data from each table
    projects_query = """
        SELECT * FROM projects;
    """
    
    transactions_query = """
        SELECT * FROM private_property_transactions;
    """

    # Fetch project data
    projects_df = fetch_data_from_db(projects_query)
    if projects_df is not None and not projects_df.empty:
        save_to_csv(projects_df, 'projects.csv')

    # Fetch transaction data
    transactions_df = fetch_data_from_db(transactions_query)
    if transactions_df is not None and not transactions_df.empty:
        save_to_csv(transactions_df, 'transactions.csv')


if __name__ == "__main__":
    run()
