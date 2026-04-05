# Breast Cancer Classifier

Projeto de MLOps para classificação binária de câncer de mama com TensorFlow/Keras, Flask e um pipeline simples de preparação, treino, avaliação e inferência.

## Visão geral

O repositório organiza um fluxo de ponta a ponta para treinar e servir um modelo de rede neural usando o dataset `breast cancer` do `scikit-learn`. O pipeline executa as seguintes etapas:

1. Geração dos dados brutos a partir do dataset público, com inserção controlada de valores ausentes.
2. Separação entre treino e teste.
3. Imputação de valores faltantes com a média das colunas.
4. Normalização das features com `StandardScaler`.
5. Treino de uma rede neural densa para classificação binária.
6. Avaliação do modelo no conjunto de teste.
7. Disponibilização do modelo em uma aplicação web Flask para inferência via upload de CSV.

## Objetivo

O objetivo do projeto é demonstrar uma estrutura prática de MLOps em um cenário pequeno, mas realista, com artefatos persistidos em disco, configuração centralizada e uma aplicação web para consumo do modelo treinado.

## Dataset

O dataset base é o `breast cancer` carregado por `sklearn.datasets.load_breast_cancer`. Durante a etapa de ingestão, o script adiciona valores nulos aleatórios em cerca de 5% das células de cada feature para simular dados imperfeitos e exercitar a etapa de imputação.

## Pipeline do projeto

### 1. Ingestão de dados

O script [src/data_loading/load_data.py](src/data_loading/load_data.py) carrega o dataset original e salva o arquivo bruto em `data/raw/raw.csv`.

### 2. Pré-processamento

O script [src/data_preprocessing/preprocess_data.py](src/data_preprocessing/preprocess_data.py) lê os dados brutos, divide em treino e teste com os parâmetros definidos em [params.yaml](params.yaml), preenche valores ausentes com `SimpleImputer(strategy='mean')` e salva:

* `data/preprocessed/train_preprocessed.csv`
* `data/preprocessed/test_preprocessed.csv`
* `artifacts/[features]_mean_imputer.joblib`

### 3. Engenharia de features

O script [src/feature_engineering/engineer_features.py](src/feature_engineering/engineer_features.py) normaliza as variáveis numéricas com `StandardScaler` e salva:

* `data/processed/train_processed.csv`
* `data/processed/test_processed.csv`
* `artifacts/[features]_scaler.joblib`

### 4. Treino do modelo

O script [src/model_training/train_model.py](src/model_training/train_model.py) prepara os dados, aplica `OneHotEncoder` ao alvo, cria uma rede neural com camadas densas e dropout, treina com early stopping e salva:

* `models/model.keras`
* `artifacts/[target]_one_hot_encoder.joblib`
* `metrics/training.json`

### 5. Avaliação

O script [src/model_evaluation/evaluate_model.py](src/model_evaluation/evaluate_model.py) executa inferência no conjunto de teste e gera métricas em `metrics/evaluation.json`, incluindo `classification_report` e matriz de confusão.

### 6. Aplicação web

A aplicação Flask em [app/main.py](app/main.py) expõe uma interface simples em `http://localhost:5001`. O usuário faz upload de um CSV com as mesmas colunas esperadas pelo dataset original e recebe as previsões geradas pelo modelo.

## Estrutura do repositório

* [app/](app)
* [src/data_loading/](src/data_loading)
* [src/data_preprocessing/](src/data_preprocessing)
* [src/feature_engineering/](src/feature_engineering)
* [src/model_training/](src/model_training)
* [src/model_evaluation/](src/model_evaluation)
* [artifacts/](artifacts)
* [data/](data)
* [metrics/](metrics)
* [models/](models)

## Principais arquivos gerados

* `data/raw/raw.csv`: dados brutos com valores ausentes simulados.
* `data/preprocessed/*.csv`: dados após imputação e split.
* `data/processed/*.csv`: dados escalados e prontos para treino e avaliação.
* `artifacts/*.joblib`: transformações reutilizadas em treino e inferência.
* `models/model.keras`: modelo treinado.
* `metrics/training.json` e `metrics/evaluation.json`: métricas do experimento.

## Requisitos

* Python 3.12 ou superior
* Dependências listadas em [pyproject.toml](pyproject.toml)
* Opcional: Docker, para execução em container

As principais bibliotecas utilizadas são `Flask`, `gunicorn`, `numpy`, `pandas`, `PyYAML`, `scikit-learn` e `tensorflow`.

## Instalação

```bash
pip install .
```

Se preferir trabalhar em um ambiente isolado, crie e ative um virtualenv antes da instalação.

## Como executar o pipeline

Execute as etapas na ordem abaixo para recriar todos os artefatos do projeto:

```bash
python -m src.data_loading.load_data
python -m src.data_preprocessing.preprocess_data
python -m src.feature_engineering.engineer_features
python -m src.model_training.train_model
python -m src.model_evaluation.evaluate_model
```

Os parâmetros de split e treino ficam centralizados em [params.yaml](params.yaml).

## Como executar a aplicação web

Depois de gerar os artefatos, inicie a API Flask:

```bash
python -m app.main
```

Em seguida, abra `http://localhost:5001`.

### Formato esperado do arquivo de entrada

O upload deve ser um arquivo CSV contendo as mesmas colunas de features do dataset de câncer de mama do `scikit-learn`. O endpoint valida a presença dessas colunas antes de executar a predição.

## Execução com Docker

O projeto também pode ser executado via Docker:

```bash
docker build -t ml-classifier .
docker run --rm -p 5001:5001 ml-classifier
```

## Observações

* O projeto persiste artefatos em disco para facilitar reuso entre treino, avaliação e inferência.
* A aplicação assume que os artefatos de `artifacts/` e `models/` já existem.
