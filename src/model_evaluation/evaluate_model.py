import logging
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger('src.model_evaluation.evaluate_model')


def load_model() -> tf.keras.Model:
    model_path = 'models/model.keras'
    model = tf.keras.models.load_model(model_path)
    return model


def load_encoder() -> OneHotEncoder:
    encoder_path = 'artifacts/[target]_one_hot_encoder.joblib'
    encoder = joblib.load(encoder_path)
    return encoder


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    data_path = 'data/processed/test_processed.csv'
    logger.info(f'Loading test data from {data_path}')
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y


def evaluate_model(model: tf.keras.Model, X: pd.DataFrame, y_true: pd.Series) -> None:
    # Generate model predictions
    y_pred_proba = model.predict(X)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate evaluation metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    evaluation = {'classification report': report, 'confusion matrix': cm}

    # Log metrics
    logger.info(f'Classification Report:\n{classification_report(y_true, y_pred)}')
    evaluation_path = 'metrics/evaluation.json'
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    model = load_model()
    X, y = load_test_data()
    evaluate_model(model, X, y)
    logger.info('Model evaluation completed')


if __name__ == '__main__':
    main()
