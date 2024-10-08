import logging
import os
import sys
from datetime import datetime

import psycopg2
import requests
import yaml
from dotenv import load_dotenv

from src.ura_pipeline.utils import load_config

# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")


class DataScraper:
    """Fetch data from API and insert it into the PostgreSQL database."""

    def __init__(self, config_file):

        # Load environment variables from .env file
        load_dotenv(".env")
        self.config = load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.headers = {
            "AccessKey": os.getenv("ACCESS_KEY"),
            "Token": os.getenv("TOKEN"),
            "User-Agent": self.config["ura_api"]["user_agent"],
        }

    def fetch_data_for_batch(self, batch_no):
        """
        Fetches data from the API for a given batch number and extracts the results

        Parameters
        ----------
        batch_no: int
            The batch number for which to fetch data

        Returns
        -------
        list
            A list of dictionaries containing the private property project and transaction data
        """
        api_url = f"{self.config['ura_api']['endpoint']}{batch_no}"

        try:
            response = requests.get(
                api_url,
                headers=self.headers,
                verify=self.config["ura_api"]["verify_ssl"],
            )
            if response.status_code == 200:
                try:
                    data = response.json()
                    results = data.get("Result", [])
                    self.logger.info(f"results: {results}")
                    if results:
                        # print the first private property project
                        self.logger.info(
                            f"Batch {batch_no} - Sample project data:", results[0]
                        )
                        # print first transaction
                        self.logger.info(
                            f"Batch {batch_no} - Sample transaction data:",
                            results[0].get("transaction", []),
                        )
                    else:
                        self.logger.info(
                            f"Batch {batch_no}: No projects found in results"
                        )
                    return results
                except ValueError as e:
                    raise RuntimeError(f"Batch {batch_no}: Error parsing JSON: {e}")

        except requests.RequestException as e:
            raise ConnectionError(f"Batch {batch_no}: Request failed: {e}")

    def insert_data_into_db(self, results):
        """
        Insert private property project and transaction data into the PostgreSQL database.

        Parameters
        ----------
        results : list
            A list of dictionaries containing the private property project and transaction data

        Raises
        ------
        ConnectionError
            If there is an error connecting to the PostgreSQL database
        """
        try:
            conn_params = {
                "host": self.config["database"]["host"],
                "port": self.config["database"]["port"],
                "database": self.config["database"]["name"],
                "user": os.getenv("POSTGRES_USER"),
                "password": os.getenv("POSTGRES_PASSWORD"),
            }
            with psycopg2.connect(**conn_params) as conn:
                self.logger.info("Connected to the PostgreSQL database!")
                with conn.cursor() as cursor:
                    for project in results:
                        cursor.execute(
                            """
                            INSERT INTO projects (
                                project_name, 
                                market_segment, 
                                street, 
                                x_coordinate, 
                                y_coordinate
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                        """,
                            (
                                project["project"],
                                project["marketSegment"],
                                project["street"],
                                project.get("x", None),
                                project.get("y", None),
                            ),
                        )
                        project_id = cursor.fetchone()[0]
                        self.logger.info(f"Inserted project with ID: {project_id}")

                        for transaction in project["transaction"]:
                            month = transaction["contractDate"][:2]
                            year = "20" + transaction["contractDate"][2:]
                            transaction_date = f"{year}-{month}"

                            cursor.execute(
                                """
                                INSERT INTO private_property_transactions (
                                    project_id, 
                                    transaction_date, 
                                    area, 
                                    price,
                                    nett_price, 
                                    property_type, 
                                    tenure, 
                                    type_of_area, 
                                    floor_range, 
                                    type_of_sale, 
                                    district, 
                                    no_of_units
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                                (
                                    project_id,
                                    transaction_date,
                                    float(transaction["area"]),
                                    float(transaction["price"]),
                                    float(transaction.get("nettPrice", 0)),
                                    transaction["propertyType"],
                                    transaction["tenure"],
                                    transaction["typeOfArea"],
                                    transaction["floorRange"],
                                    transaction["typeOfSale"],
                                    transaction["district"],
                                    int(transaction["noOfUnits"]),
                                ),
                            )
                    conn.commit()
                    self.logger.info("Data inserted successfully!")
        except psycopg2.Error as e:
            raise ConnectionError(f"Unable to connect to the database: {e}")

    def run(self):
        """
        Fetches data from the API for all batches specified in the config and
        inserts the data into the database.
        """
        for batch_no in range(
            self.config["batches"]["range"][0], self.config["batches"]["range"][1] + 1
        ):
            self.logger.info(f"Fetching data for batch {batch_no}...")
            results = self.fetch_data_for_batch(batch_no)
            if results:
                self.logger.info(
                    f"Inserting data for batch {batch_no} into the database..."
                )
                self.insert_data_into_db(results)
            else:
                self.logger.info(f"No data to insert for batch {batch_no}.")


if __name__ == "__main__":
    logging.info("Starting DataScraper...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    os.chdir(root)
    pipeline = DataScraper("conf/base/config.yaml")
    pipeline.run()
