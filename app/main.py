import io
import logging
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model

logger = logging.getLogger('app.main')


class ModelService:
    def __init__(self) -> None:
        logger.info('Loading artefacts from local project folder')

        artifacts_dir = 'artifacts'
        models_dir = 'models'

        features_imputer_path = os.path.join(
            artifacts_dir, '[features]_mean_imputer.joblib'
        )
        features_scaler_path = os.path.join(artifacts_dir, '[features]_scaler.joblib')
        target_encoder_path = os.path.join(
            artifacts_dir, '[target]_one_hot_encoder.joblib'
        )

        model_path = os.path.join(models_dir, 'model.keras')

        self.features_imputer = joblib.load(features_imputer_path)
        self.features_scaler = joblib.load(features_scaler_path)
        self.target_encoder = joblib.load(target_encoder_path)
        self.model = load_model(model_path)

        logger.info('Successfully loaded all artifacts')

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        y_pred = self.model.predict(X_scaled)
        y_decoded = self.target_encoder.inverse_transform(y_pred)

        return pd.DataFrame({'Predicition': y_decoded.ravel()}, index=features.index)


def create_routes(app: Flask) -> None:
    @app.route('/')
    def index() -> str:
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload() -> str:
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return render_template('index.html', error='Please upload a CSV file')

        try:
            content = file.read().decode('utf-8')
            features = pd.read_csv(io.StringIO(content))

            # Validate column names against breast cancer dataset
            expected_features = load_breast_cancer().feature_names
            missing_cols = [
                col for col in expected_features if col not in features.columns
            ]
            if missing_cols:
                return render_template(
                    'index.html',
                    error=f'Missing required columns {', '.join(missing_cols)}',
                )

            features = features[expected_features]

            predictions = app.model_service.predict(features)
            result = predictions.to_string()

            return render_template('index.html', predictions=result)
        except Exception as e:
            logger.error(f'Error processing file: {e}', exc_info=True)
            return render_template(
                'index.html', error=f'Error processing file: {str(e)}'
            )


app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info('Application initialized with model service and routes')


def main() -> None:
    app.run(host='0.0.0.0', port=5001)


if __name__ == '__main__':
    main()
