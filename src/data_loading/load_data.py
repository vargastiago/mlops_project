import logging

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger('src.data_loading.load_data')


def fetch_data() -> pd.DataFrame:
    logger.info('Fetching data...')
    dataset = load_breast_cancer()

    # Feature columns
    data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

    # Introduce random NaN values
    np.random.seed(42)
    for col in data.columns:
        mask = np.random.random(len(data)) < 0.05
        data.loc[mask, col] = np.nan

    # Target column
    data['target'] = dataset.target

    return data


def save_data(data: pd.DataFrame) -> None:
    output_path = 'data/raw/raw.csv'
    logger.info(f'Saving raw data to {output_path}')
    data.to_csv(output_path, index=False)


def main() -> None:
    raw_data = fetch_data()
    save_data(raw_data)
    logger.info('Data loading completed')


if __name__ == '__main__':
    main()
