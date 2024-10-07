import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def count_null_values(
        raw_data: pd.DataFrame,
    ) -> pd.Series:
        """
        Counts the number of null values in each column of the dataframe.

        Args:
            raw_data (pd.DataFrame): The input dataframe.

        Returns:
            pd.Series: A series with the count of null values for each column.
        """
        null_counts = raw_data.isnull().sum()
        has_nulls = False

        for col, count in null_counts.items():
            if count > 0:
                logger.warning(f"Column '{col}' has {count} null value(s)")
                has_nulls = True

        if not has_nulls:
            logger.info("Dataframe has no null value(s)")

        return null_counts