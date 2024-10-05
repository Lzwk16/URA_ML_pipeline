import os
from datetime import datetime

import psycopg2
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Base API URL
api_url_base = "https://www.ura.gov.sg/uraDataService/invokeUraDS?service=PMI_Resi_Transaction&batch="

# Database credentials
db_name = os.getenv('POSTGRES_DB')
user_name = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')

# API credentials
access_key = os.getenv('ACCESS_KEY')
token = os.getenv('TOKEN')

# Set up headers for the API request
headers = {
    "AccessKey": access_key,
    "Token": token,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def fetch_data_for_batch(batch_no):
    """Fetch data from the API for a specific batch number."""
    api_url = api_url_base + str(batch_no)
    
    try:
        response = requests.get(api_url, headers=headers, verify=True)
        if response.status_code == 200:
            try:
                data = response.json()
                results = data.get('Result', [])
                if results:
                    print(f"Batch {batch_no} - Sample project data:", results[0])  # Print the first project
                    print(f"Batch {batch_no} - Sample transaction data:", results[0].get("transaction", []))  # Print first transaction
                else:
                    print(f"Batch {batch_no}: No projects found in results")
                return results
            except ValueError as e:
                raise RuntimeError(f"Batch {batch_no}: Error parsing JSON: {e}")
    
    except requests.RequestException as e:
        raise ConnectionError(f"Batch {batch_no}: Request failed: {e}")

def insert_data_into_db(results):
    """Insert project and transaction data into the PostgreSQL database."""
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

    cursor = conn.cursor()

    # Iterate through each project in results
    for project in results:
        cursor.execute("""
            INSERT INTO projects (project_name, market_segment, street, x_coordinate, y_coordinate)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            project["project"],
            project["marketSegment"],
            project["street"],
            project.get("x", None),
            project.get("y", None)
        ))

        project_id = cursor.fetchone()[0]  # Get the generated project ID
        print(f"Inserted project with ID: {project_id}")

        # Iterate through each transaction for the current project
        for transaction in project["transaction"]:
            # Convert contractDate to a date format
            month = transaction['contractDate'][:2]
            year = "20" + transaction['contractDate'][2:]  # Assuming all years are in the 2000s
            transaction_date = datetime.strptime(f"{year}-{month}-01", "%Y-%m-%d").date()

            cursor.execute("""
                INSERT INTO private_property_transactions (
                    project_id, 
                    transaction_date, 
                    area, 
                    price, 
                    property_type, 
                    tenure, 
                    type_of_area, 
                    floor_range, 
                    type_of_sale, 
                    district, 
                    no_of_units
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                project_id,
                transaction_date,
                float(transaction['area']),
                float(transaction['price']),
                transaction['propertyType'],
                transaction['tenure'],
                transaction['typeOfArea'],
                transaction['floorRange'],
                transaction['typeOfSale'],
                transaction['district'],
                int(transaction['noOfUnits']),
            ))

    conn.commit()
    print("Data inserted successfully!")

    # Close the connection
    cursor.close()
    conn.close()


def run():
    """Main function to download data from all batches (1-4) and insert into the database."""
    for batch_no in range(1, 5):  # Loop through batch numbers 1 to 4
        print(f"Fetching data for batch {batch_no}...")
        results = fetch_data_for_batch(batch_no)
        if results:
            print(f"Inserting data for batch {batch_no} into the database...")
            insert_data_into_db(results)
        else:
            print(f"No data to insert for batch {batch_no}.")


if __name__ == "__main__":
    run()