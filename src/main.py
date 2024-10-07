import logging
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ura_pipeline.data_prep.data_loader import DataLoader
from src.ura_pipeline.data_prep.data_processing import DataProcessor
from src.ura_pipeline.database.data_scraper import DataScraper
from src.ura_pipeline.model.train_model import ModelTrainer
from src.ura_pipeline.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main(cfg, model_cfg):
    """
    Main entry point for the script.

    This function fetches data from the API using the config provided in `cfg`,
    processes the data using the config provided in `model_cfg`, and trains a
    model using the processed data.

    Parameters
    ----------
    cfg : str
        The path to the config file for fetching and processing data.
    model_cfg : str
        The path to the config file for training the model.
    """
    start = time.time()

    # Fetch data from API and save to CSV files
    scraper = DataScraper(cfg)
    loader = DataLoader(cfg)
    logger.info("Fetching data from API and saving to CSV files...")
    scraper.run()
    projects_path, transactions_path = loader.run()
    logger.info("Data fetched and saved to CSV files.")

    # Load data from CSV files
    logger.info(f"loading data from {projects_path} and {transactions_path}")
    projects_df = pd.read_csv(projects_path)
    transactions_df = pd.read_csv(transactions_path)

    # Process data and split into train and test sets
    data_processor = DataProcessor(cfg, projects_df, transactions_df)
    X_train, y_train, X_test, y_test = data_processor.run()

    model_trainer = ModelTrainer(model_cfg)

    # # Train the model
    model_trainer.run(X_train, y_train, X_test, y_test)
    end = time.time()
    print(f"Total time taken: {end - start}")


if __name__ == "__main__":
    main(cfg="./conf/base/config.yaml", model_cfg="./conf/base/train_model.yaml")
